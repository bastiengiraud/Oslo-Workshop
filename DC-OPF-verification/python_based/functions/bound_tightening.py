"""
based on E2_neural_network_scalability_milp.m

% First: ReLU stability: eliminate inactive and fix active ReLUs
% Second: Interval Arithmetic to tighten bounds
% Third: LP relaxation to tighten bounds
% Fourth: Full MILP to tighten bounds

"""

import numpy as np
import pandapower as pp
import pandapower.converter as pc
import pandas as pd
import scipy.io
import gurobipy as gp
from gurobipy import Model, GRB
import gurobi_ml
import os

from statistical_bound import DataLoader, ReluStability

os.environ['GRB_LICENSE_FILE'] = 'C:/Users/bagir/gurobi.lic'

# try to load nn with gurobi
#gurobi_ml.add_mlp_regressor_constr()



class ReluBoundTightening:
    def __init__(self, nn_path, layers, data_loader, nn_relu_stability):
        self.nn_path = nn_path
        self.layers = layers
        self.input_nn = data_loader.input_nn
        self.output_nn = data_loader.output_nn
        self.w_hidden = data_loader.w_hidden
        self.w_input = data_loader.w_input
        self.w_output = data_loader.w_output
        self.biases = data_loader.biases
        
        self.nb_samples = self.input_nn.shape[0]  # Number of samples
        self.ReLU_layers = len(self.w_hidden) + 1  # Number of layers with ReLU activation
        self.nb_neurons = self.w_hidden[0].shape[1]  # Number of neurons in each layer (assuming square layers)
        
        self.ReLU_always_active = nn_relu_stability.ReLU_always_active
        self.ReLU_always_inactive = nn_relu_stability.ReLU_always_inactive
        
        # apply interval arithmetic
        self.interval_arithmetic()
        
        # apply optimization based bound tightening twice
        self.obbt()
        

        
    def interval_arithmetic(self):
        # initialize large bounds
        self.zk_hat_max = np.ones((self.nb_neurons,self.ReLU_layers))*(10000) 
        self.zk_hat_min = np.ones((self.nb_neurons,self.ReLU_layers))*(-10000) 

        # use interval arithmetic to compute tighter bounds, initial input bounds. these are your initial input bounds (0 < pd < 1).
        u_init = np.ones((self.input_nn.shape[1])) 
        l_init = np.zeros((self.input_nn.shape[1]))  

        self.zk_hat_max[:, 0] = (np.maximum(self.w_input, 0) @ u_init + np.minimum(self.w_input, 0) @ l_init + self.biases[0])
        self.zk_hat_min[:, 0] = (np.minimum(self.w_input, 0) @ u_init + np.maximum(self.w_input, 0) @ l_init + self.biases[0])

        # Compute the rest of the layers
        for j in range(self.ReLU_layers - 1):  
            self.zk_hat_max[:, j + 1] = (np.maximum(self.w_hidden[j], 0) @ np.maximum(self.zk_hat_max[:, j], 0) + np.minimum(self.w_hidden[j], 0) @ np.maximum(self.zk_hat_min[:, j], 0) + self.biases[j + 1])
            self.zk_hat_min[:, j + 1] = (np.minimum(self.w_hidden[j], 0) @ np.maximum(self.zk_hat_max[:, j], 0) + np.maximum(self.w_hidden[j], 0) @ np.maximum(self.zk_hat_min[:, j], 0) + self.biases[j + 1])
        
        self.zk_hat_max_cur = self.zk_hat_max.copy()
        self.zk_hat_min_cur = self.zk_hat_min.copy() 
    
        
    def obbt(self):

        for run in range(1, 3):
            lp_tightening = True
            
            # tighten the bounds of each individual RELU, MILP_tightening
            if lp_tightening == True: #build relaxed/MILP optimization problem
                size_input = self.input_nn.shape[1]
                size_output = self.output_nn.shape[1]
                
                if run == 1:
                    lp_relax = True # first solve LP relaxation
                else:
                    lp_relax = False # second solve full MILP formulation
                                
            # Create Gurobi model
            model = Model("NN_Optimization")
            model.setParam("OutputFlag", 0)  # Suppress Gurobi output by setting to 0

            # Variables
            pd_NN = model.addMVar((size_input, 1), lb = 0.0, ub = 1.0, vtype=GRB.CONTINUOUS, name="pd_NN")
            # pg_pred = model.addMVar((size_output, 1), vtype=GRB.CONTINUOUS, name="pg_pred") # do i put an ub on pg_pred? then never violated?
            zk_hat = model.addMVar((self.nb_neurons, self.ReLU_layers), lb = self.zk_hat_min_cur, ub = self.zk_hat_max_cur, name="zk_hat")
            zk = model.addMVar((self.nb_neurons, self.ReLU_layers), lb = -GRB.INFINITY, ub = GRB.INFINITY, name="zk")
            
            if lp_relax:
                ReLU = model.addMVar((self.nb_neurons, self.ReLU_layers), lb = 0.0, ub = 1.0, vtype=GRB.CONTINUOUS, name="ReLU")
            else:
                ReLU = model.addMVar((self.nb_neurons, self.ReLU_layers), vtype=GRB.BINARY, name="ReLU")

            # Constraints
            constr_tightening = []     

            # Input layer constraints            
            constr_tightening.append(model.addConstr(zk_hat[:, 0] == self.w_input[:, :] @ pd_NN[:, 0] + self.biases[0][:]))

            # Hidden layer constraints
            for layer in range(1, self.ReLU_layers):
                constr_tightening.append(model.addConstr(zk_hat[:, layer] == self.w_hidden[layer - 1][:, :] @ zk[:, layer - 1] + self.biases[layer][:]))                   
                    
            # Output layer constraints
            # constr_tightening.append(model.addConstr(pg_pred[:, 0] == self.w_output[:, :] @ zk[:, 2] + self.biases[3][:])) 

            # ReLU stability and bounds
            for k in range(self.ReLU_layers):
                for m in range(self.nb_neurons):
                    constr_tightening_loop = []
                    if m > 1:
                        for constr in constr_tightening_loop:
                            model.removeConstr(constr)
                            model.update()
                    
                    for i in range(self.ReLU_layers):
                        for jj in range(self.nb_neurons):
                            if self.ReLU_always_active[i, jj] == 1:
                                constr_tightening_loop.append(model.addConstr(zk[jj, i] == zk_hat[jj, i]))
                            elif self.ReLU_always_inactive[i, jj] == 1:
                                constr_tightening_loop.append(model.addConstr(zk[jj, i] == 0))
                            else: # rewrite the max function
                                constr_tightening_loop.append(model.addConstr(zk[jj, i] <= zk_hat[jj, i] - self.zk_hat_min_cur[jj, i] * (1 - ReLU[jj, i])))
                                constr_tightening_loop.append(model.addConstr(zk[jj, i] >= zk_hat[jj, i]))
                                constr_tightening_loop.append(model.addConstr(zk[jj, i] <= self.zk_hat_max_cur[jj, i] * ReLU[jj, i]))
                                constr_tightening_loop.append(model.addConstr(zk[jj, i] >= 0))
                    
                    # Solve for lower bound
                    model.setObjective(zk_hat[m, k], GRB.MINIMIZE)
                    model.optimize()
                    
                    # model.Status = 2 (optimal), model.Status = 3 (infeasible), model.Status = 5 (unbounded)
                    issues = {2 : 'optimal', 3 : 'infeasible', 5 : 'unbounded'}
                    
                    if model.Status != GRB.OPTIMAL:
                        raise ValueError("Some issue with solving MILP 1: ", issues[model.Status])
                    else:
                        self.zk_hat_min_cur[m, k] = min(zk_hat[m, k].x, self.zk_hat_max_cur[m, k] - 1e-3) 

                    # Solve for upper bound
                    model.update()
                    model.setObjective(-zk_hat[m, k], GRB.MINIMIZE)
                    model.optimize()

                    if model.Status != GRB.OPTIMAL:
                        raise ValueError("Some issue with solving MILP 2: ", issues[model.Status])
                    else:
                        self.zk_hat_max_cur[m, k] = max(zk_hat[m, k].x, self.zk_hat_min_cur[m, k] + 1e-3)
        
            # this shows by how much we reduced the bounds 
            mean_value = np.mean((self.zk_hat_max_cur + 1e-3 - self.zk_hat_min_cur) / (self.zk_hat_max + 1e-3 - self.zk_hat_min)) # prevents elementwise division by very small number
            self.zk_hat_max = self.zk_hat_max_cur.copy()
            self.zk_hat_min = self.zk_hat_min_cur.copy()
            print("reduction: ", (1 - mean_value)*100, "%")

        print("bound tightening completed!")
        
        scipy.io.savemat(os.path.join(self.nn_path, 'zk_hat_min.mat'), {'zk_hat_min': self.zk_hat_min})
        scipy.io.savemat(os.path.join(self.nn_path, 'zk_hat_max.mat'), {'zk_hat_max': self.zk_hat_max})

        
if __name__ == "__main__":
    
    # Get the current working directory
    parent_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

    # Define the cases
    case_name = 'case39_DCOPF'
    case_iter = '1'
    case_path = os.path.join(parent_directory, "python_based", "test_networks", "network_modified")

    # define the neural network
    nn_path = os.path.join(parent_directory, "python_based", "trained_nns", case_name, case_iter)
    
    # load dataset (test, train, all)
    data_split = "all"
    data_loader = DataLoader(nn_path, data_split)
    nn_relu_stability = ReluStability(nn_path, 3, data_loader)

    # load nn and determine relu stability
    nn_tightening = ReluBoundTightening(nn_path, 3, data_loader, nn_relu_stability)



# Time_Scalability(c,iter)=toc(tElapsed);


