"""
based on E3_worst_case_constraint_violations_milp.m & copmute_worst_case_constraint_violations.m

"""

import numpy as np
import pandapower as pp
import scipy.io
import gurobipy as gp
from gurobipy import Model, GRB

import os

from statistical_bound import PrepareDCOPFData, DataLoader, NeuralNetworkPrediction
os.environ['GRB_LICENSE_FILE'] = 'C:/Users/bagir/gurobi.lic'

class MILPModel:
    def __init__(self, name='MILP_Model'):
        self.model = Model(name)
        self.model.setParam("OutputFlag", 0)  # Suppress output by default
        self.model.reset()
        

class PowerSystemModel:
    def __init__(self, mpc, milp, input_nn, output_nn, line_flow_formulation):
        self.milp = milp
        self.model = milp.model
        self.mpc = mpc
        self.input_nn = input_nn
        self.output_nn = output_nn
        self.line_flow_formulation = line_flow_formulation
        
        self.add_dcopf_variables()
        self.add_dcopf_constraints()
    
    def add_dcopf_variables(self):
        """
        if you don't assign bounds to variables, gurobipy will automatically assume a lb of 0 and an ub of inf
        
        """
        delta = 0.0
        self.pd_NN = self.model.addMVar((self.input_nn.shape[1], 1), lb = 0.0 + delta, ub = 1.0 - delta, vtype=GRB.CONTINUOUS, name="pd_NN")
        self.pg_pred = self.model.addMVar((self.output_nn.shape[1], 1), lb = -GRB.INFINITY, ub = GRB.INFINITY, vtype=GRB.CONTINUOUS, name="pg_pred") # [p.u.]
        self.theta = self.model.addMVar((self.mpc.nb), lb = -GRB.INFINITY, ub = GRB.INFINITY, vtype=GRB.CONTINUOUS, name="theta") #  
        self.pline = self.model.addMVar((self.mpc.nl + self.mpc.nl_tf), lb = -GRB.INFINITY, ub = GRB.INFINITY, vtype=GRB.CONTINUOUS, name="pline") # [p.u.]
        self.pg_slack = self.model.addMVar((1, 1), lb = -GRB.INFINITY, ub = GRB.INFINITY, vtype=GRB.CONTINUOUS, name="pg_slack") # [p.u.]
        
        self.M_g_nsb = np.delete(np.delete(self.mpc.M_g, self.mpc.gen_slack, axis=1), self.mpc.slack, axis = 0)
        self.M_d_nsb = np.delete(self.mpc.M_d, self.mpc.slack, axis=0)


    def add_dcopf_constraints(self):
        if self.line_flow_formulation == 'dcpf':
            # DC power balance including slack bus
            self.model.addConstr(self.mpc.M_g @ gp.vstack((self.pg_slack, (self.pg_pred * self.mpc.pg_delta.reshape(-1, 1) / self.mpc.baseMVA))) 
                                - self.mpc.M_d @  ((self.pd_NN * self.mpc.pd_delta.reshape(-1, 1) + self.mpc.pd_min.reshape(-1, 1)) / self.mpc.baseMVA)
                                == self.mpc.B @ self.theta.reshape(-1, 1), "power_balance")
            
            # active power line flows
            self.model.addConstr(self.pline == (self.mpc.Bline @ self.theta), "line_flow")
            
            # Slack bus constraint
            self.model.addConstr(self.theta[self.mpc.slack] == 0)
                       
        elif self.line_flow_formulation == 'ptdf':
            # Power balance constraint on slack bus
            self.model.addConstr(self.pg_slack == sum(((self.pd_NN * self.mpc.pd_delta.reshape(-1, 1) + self.mpc.pd_min.reshape(-1, 1)) / self.mpc.baseMVA)) 
                - sum(self.pg_pred * self.mpc.pg_delta.reshape(-1, 1) / self.mpc.baseMVA)
            )
        
        

class NeuralNetwork:
    def __init__(self, milp, input_vars, output_vars, data_loader, zk_hat_min, zk_hat_max, ReLU_always_active, ReLU_always_inactive):
        self.model = milp.model
        self.input_vars = input_vars # pd_NN
        self.output_vars = output_vars # pg_pred
        self.zk_hat_min = zk_hat_min
        self.zk_hat_max = zk_hat_max
        self.ReLU_always_active = ReLU_always_active
        self.ReLU_always_inactive = ReLU_always_inactive
        self.w_input = data_loader.w_input
        self.w_output = data_loader.w_output
        self.w_hidden = data_loader.w_hidden
        self.biases = data_loader.biases
        self.ReLU_layers = 3
        self.nb_neurons = self.w_hidden[0].shape[1]
        self.size_input = data_loader.input_nn.shape[1]
        self.size_output = data_loader.output_nn.shape[1]
    
        self.define_variables()
        self.setup_nn_constraints()
        
    def define_variables(self):
        # define variables
        self.zk_hat = self.model.addMVar((self.nb_neurons, self.ReLU_layers), lb = self.zk_hat_min, ub = self.zk_hat_max, vtype=GRB.CONTINUOUS, name="zk_hat") # , lb = self.zk_hat_min, ub = self.zk_hat_max
        self.zk = self.model.addMVar((self.nb_neurons, self.ReLU_layers), lb = 0, ub = GRB.INFINITY, vtype=GRB.CONTINUOUS, name="zk") # lb = 0, ub = 1000, 
        self.ReLU = self.model.addMVar((self.nb_neurons, self.ReLU_layers), vtype=GRB.BINARY, name="ReLU")

    def setup_nn_constraints(self):       
        # Input layer constraints
        self.model.addConstr(self.zk_hat[:, 0] == self.w_input[:, :] @ self.input_vars[:, 0] + self.biases[0])
        
        # Hidden layer constraints
        for layer in range(1, self.ReLU_layers):
            self.model.addConstr(self.zk_hat[:, layer] == self.w_hidden[layer - 1][:, :] @ self.zk[:, layer - 1] + self.biases[layer])
                
        # Output layer constraints
        self.model.addConstr(self.output_vars[:, 0] == self.w_output[:, :] @ self.zk[:, 2] + self.biases[3])

        # Relu stability
        for i in range(self.ReLU_layers):
            for jj in range(self.nb_neurons):
                if self.ReLU_always_active[i, jj] == 1:
                    self.model.addConstr(self.zk[jj, i] == self.zk_hat[jj, i])
                elif self.ReLU_always_inactive[i, jj] == 1:
                    self.model.addConstr(self.zk[jj, i] == 0)
                else: # rewrite the max function
                    self.model.addConstr(self.zk[jj, i] <= self.zk_hat[jj, i] - self.zk_hat_min[jj, i] * (1 - self.ReLU[jj, i]))
                    self.model.addConstr(self.zk[jj, i] >= self.zk_hat[jj, i])
                    self.model.addConstr(self.zk[jj, i] <= self.zk_hat_max[jj, i] * self.ReLU[jj, i])
                    self.model.addConstr(self.zk[jj, i] >= 0)


class WorstCaseViolationAnalyzer:
    def __init__(self, milp, mpc, power_system, neural_network, nn_prediction):
        self.milp = milp
        self.mpc = mpc
        self.power_system = power_system
        self.neural_network = neural_network
        self.nn_prediction = nn_prediction
        
        # placeholders gen up violations
        self.v_g_wc_max = np.zeros(self.mpc.ng + 1)
        self.v_g_time_max = np.zeros(self.mpc.ng + 1)
        self.pl_values = np.zeros((2 * self.mpc.ng + 1 + self.mpc.nl + self.mpc.nl_tf, self.mpc.nloads))
        
        # placeholders gen down violations
        self.v_g_wc_min = np.zeros(self.mpc.ng + 1)
        self.v_g_time_min = np.zeros(self.mpc.ng + 1)
        
        # placeholders line violations
        self.v_line_wc = np.zeros((self.mpc.nl + self.mpc.nl_tf))
        self.v_line_time = np.zeros((self.mpc.nl + self.mpc.nl_tf))
        
        self.v_info = {}
        
        # compute wc violations
        self.generator_up_violations()
        self.generator_down_violations()
        self.wc_gen_violation()
        self.wc_line_violation()
        
        # BestBdStop sets an objective on the dual bound
        # BestObjStop sets an objective on the primal bound 

    def generator_up_violations(self):
        #self.milp.model.reset()
        print("Solving MILP for PGMAX Violations")
        for i in range(self.mpc.ng+1):
            self.milp.model.setParam("BestBdStop", 0.0)
            
            # specify objective. 
            if i == self.mpc.ng: # slack bus
                obj = (self.power_system.pg_slack[0, 0] * self.mpc.baseMVA - self.mpc.pg_slack_max[0])
            else:
                obj = (self.power_system.pg_pred[i, 0] * self.mpc.pg_max[i] - self.mpc.pg_max[i])
            self.milp.model.setObjective(obj, GRB.MAXIMIZE)
            # self.milp.model.setParam("DualReductions", 0)
            self.milp.model.optimize() 
    
            # Extract solution
            self.v_g_time_max[i] = self.milp.model.Runtime
            try:
                # Check if the first variable has an assigned value
                if hasattr(self.power_system.pd_NN[0, 0], "x"):
                    self.pl_values[i, :] = [self.power_system.pd_NN[j, 0].x for j in range(self.mpc.nloads)]
                else:
                    print(f"Generator {i}, Model is not optimal. Cannot access variable values. Status: {self.milp.model.status}")
                    self.pl_values[i, :] = np.nan  # Fill with NaN
                    print("Optimal Objective Value:", self.milp.model.ObjVal)
                    print("Dual Bound (Best Bound):", self.milp.model.ObjBound)
                    print("MIP Gap:", self.milp.model.MIPGap)
            except (AttributeError, gp.GurobiError) as e:
                print(f"Generator {i}, Error accessing variable values: {str(e)}, model status: {self.milp.model.status}")
                self.pl_values[i, :] = np.nan  # Fill with NaN in case of an error
                print("Optimal Objective Value:", self.milp.model.ObjVal)
                print("Dual Bound (Best Bound):", self.milp.model.ObjBound)
                print("MIP Gap:", self.milp.model.MIPGap)
            
            # Check if the MILP solution is not optimal
            if self.milp.model.status != GRB.OPTIMAL:  

                # Attempt to rerun the optimization with a conservative presolve strategy
                self.milp.model.presolve()
                self.milp.model.optimize()
            

                # If the issue persists after re-optimization, abort with a detailed error message
                if self.milp.model.status not in [GRB.OPTIMAL, GRB.INFEASIBLE, GRB.USER_OBJ_LIMIT]:  
                    raise Exception(
                        f"Generator {i}: Issue with solving MILP PGMAX. "
                        f"Solver Status: {self.milp.model.status}, "
                        f"Objective Value: {getattr(self.milp.model, 'objVal', 'N/A')}, "
                        f"MIP Gap: {getattr(self.milp.model, 'MIPGap', 'N/A')}"
                                                                )

            if self.milp.model.Status == GRB.OPTIMAL:
                self.v_g_wc_max[i] = obj.getValue()  # getValue() for the Gurobi variable value
                self.milp.model.setParam("BestBdStop", obj.getValue()) # early stopping at objective value

                # Double check that the computed neural network input leads to the worst-case violation
                mpc_test = self.mpc.mpc
                
                pd_NN_values = np.array([var.X for var in (self.power_system.pd_NN)])
                pg_pred_nn = self.nn_prediction.nn_prediction_all_ReLUs(pd_NN_values[:,0])
                pg_pred_values = np.array([var.X for var in (self.power_system.pg_pred)])
                    
                if np.sum(np.abs(pg_pred_nn - pg_pred_values[:,0])) > 1e-3:
                    print(f'Generator {i}, Mismatch in neural network prediction -- PGMAX: {np.sum(np.abs(pg_pred_nn - pg_pred_values[:,0]))}')
                    #raise Exception(f'Generator {i}, Mismatch in neural network prediction -- PGMAX')

                mpc_test.load['p_mw'] = pd_NN_values[:,0] * self.mpc.pd_delta + self.mpc.pd_min
                mpc_test.gen['p_mw'] = self.mpc.pg_delta * pg_pred_nn
                
                pp.rundcpp(mpc_test)
                if i == self.mpc.ng:
                    pg_viol_max = np.array(mpc_test.res_ext_grid['p_mw'][0] - self.mpc.pg_slack_max[0])
                else:
                    pg_viol_max = np.array(mpc_test.res_gen['p_mw'][i] - self.mpc.pg_max[i])
                
                if np.abs(pg_viol_max - self.v_g_wc_max[i]) > 1e-2:
                    print(f'Generator {i}, Mismatch in worst-case violation -- PGMAX. Violation dcpf: ', pg_viol_max, 'Violation optimization: ', self.v_g_wc_max[i]) #raise Exception

                if self.milp.model.MIPGap > 10e-1:
                    raise Exception(f'Generator {i}, MILP gap larger than 10e-2')
                
            elif self.milp.model.Status != GRB.OPTIMAL: 
                self.v_g_wc_max[i] = self.milp.model.objBound - 1e-4
                
            # print(f"worst-case up violation for generator {i}: ", self.v_g_wc_max[i])
       
    
    def generator_down_violations(self):
        # use warm start solution from generator_up to help the solver. Otherwise the MIPgap is too large, and it doesn't solve.
        # self.milp.model.reset() # so don't reset model but keep previous solution; comment this line out
        print("Solving MILP for PGMIN Violations")
        for i in range(self.mpc.ng + 1):
            self.milp.model.setParam("BestBdStop", 0.0)
            
            # specify objective function
            if i == self.mpc.ng: # slack bus
                obj = (self.mpc.pg_slack_min - self.power_system.pg_slack[0, 0] * self.mpc.baseMVA)
            else:
                obj = (self.mpc.pg_min[i] - self.power_system.pg_pred[i, 0] * self.mpc.pg_max[i])
            self.milp.model.setObjective(obj, GRB.MAXIMIZE)
            self.milp.model.optimize()
               
            # Extract solution
            self.v_g_time_min[i] = self.milp.model.Runtime
            try:
                # Check if the first variable has an assigned value
                if hasattr(self.power_system.pd_NN[0, 0], "x"):
                    self.pl_values[self.mpc.ng + i, :] = [self.power_system.pd_NN[j, 0].x for j in range(self.mpc.nloads)]
                else:
                    print(f"Generator {i}, Model is not optimal. Cannot access variable values. Status: {self.milp.model.status}")
                    self.pl_values[self.mpc.ng + i, :] = np.nan  # Fill with NaN
                    print("Optimal Objective Value:", self.milp.model.ObjVal)
                    print("Dual Bound (Best Bound):", self.milp.model.ObjBound)
                    print("MIP Gap:", self.milp.model.MIPGap)
            except (AttributeError, gp.GurobiError) as e:
                print(f"Generator {i}, Error accessing variable values: {str(e)}, model status: {self.milp.model.status}")
                self.pl_values[self.mpc.ng + i, :] = np.nan  # Fill with NaN in case of an error
                print("Optimal Objective Value:", self.milp.model.ObjVal)
                print("Dual Bound (Best Bound):", self.milp.model.ObjBound)
                print("MIP Gap:", self.milp.model.MIPGap)
            
            # Check if the MILP solution is not optimal
            if self.milp.model.status != GRB.OPTIMAL:  

                # Attempt to rerun the optimization with a conservative presolve strategy
                self.milp.model.presolve()
                self.milp.model.optimize()

                # If the issue persists after re-optimization, abort with a detailed error message
                if self.milp.model.status not in [GRB.OPTIMAL, GRB.INFEASIBLE, GRB.USER_OBJ_LIMIT]:  
                    raise Exception(
                        f"Generator {i}: Issue with solving MILP PGMIN. "
                        f"Solver Status: {self.milp.model.status}, "
                        f"Objective Value: {getattr(self.milp.model, 'objVal', 'N/A')}, "
                        f"MIP Gap: {getattr(self.milp.model, 'MIPGap', 'N/A')}"
                                                                )

            if self.milp.model.Status == GRB.OPTIMAL: # 15 is stopping due to reaching user specified objective: the objective bound or best bound!
                self.v_g_wc_min[i] = obj.getValue()  
                self.milp.model.setParam("BestBdStop", obj.getValue()) # early stopping at objective value

                # Double check that the computed neural network input leads to the worst-case violation
                mpc_test = self.mpc.mpc
                pd_NN_values = np.array([var.X for var in (self.power_system.pd_NN)])
                pg_pred_nn = self.nn_prediction.nn_prediction_fixed_ReLUs(pd_NN_values[:,0])
                pg_pred_values = np.array([var.X for var in (self.power_system.pg_pred)])
                    
                if np.sum(np.abs(pg_pred_nn - pg_pred_values[:,0])) > 10-4:
                    raise Exception(f'Generator {i}, Mismatch in neural network prediction -- PGMIN')

                mpc_test.load['p_mw'] = pd_NN_values[:,0] * self.mpc.pd_delta + self.mpc.pd_min
                mpc_test.gen['p_mw'] = self.mpc.pg_delta * pg_pred_nn
                
                pp.rundcpp(mpc_test)
                if i == self.mpc.ng:
                    pg_viol_min = np.array(self.mpc.pg_slack_min[0] - mpc_test.res_ext_grid['p_mw'][0])
                else:
                    pg_viol_min = np.array(self.mpc.pg_min[i] - mpc_test.res_gen['p_mw'][i])
                
                if np.abs(pg_viol_min - self.v_g_wc_min[i]) > 10e-4:
                    print(f'Generator {i}, Mismatch in worst-case violation -- PGMIN. Violation dcpf: ', pg_viol_min, 'Violation optimization: ', self.v_g_wc_min[i]) # raise Exception

                if self.milp.model.MIPGap > 1e-2:
                    raise Exception(f'Generator {i}, MILP gap larger than 1e-2')
                
            elif self.milp.model.Status != GRB.OPTIMAL: 
                self.v_g_wc_min[i] = self.milp.model.objBound - 1e-4
                
            # print(f"worst-case down violation for generator {i}: ", self.v_g_wc_min[i])
                
    def wc_gen_violation(self):
        # Generator violation
        self.v_info["v_g_time"] = np.sum(self.v_g_time_min + self.v_g_time_max)

        # Identify whether upper or lower generator bound violations are larger
        if np.max(self.v_g_wc_min) > np.max(self.v_g_wc_max):
            self.v_info["v_g_wc"] = np.max(self.v_g_wc_min)
            self.v_info["v_g_ID"] = np.argmax(self.v_g_wc_min)
        else:
            self.v_info["v_g_wc"] = np.max(self.v_g_wc_max)
            self.v_info["v_g_ID"] = np.argmax(self.v_g_wc_max)

    def wc_line_violation(self):
        # self.milp.model.reset()
        print('Solving MILP for PLINE Violations')
        for i in range(self.mpc.nl + self.mpc.nl_tf):
            solved = False # some instances have issues with the GUROBI presolve;
            for runs in range(2):  # if solving with presolve fails we re-run without presolve and check again
                if not solved:
                    self.milp.model.setParam("BestBdStop", 0.0)
                    self.milp.model.Params.TimeLimit = 10
                    
                    if runs == 0:
                        # automatic presolve: this speeds up the MILP in most cases, in very few it malfunctions
                        self.milp.model.setParam('Presolve', -1)  # Automatic presolve
                    else:
                        self.milp.model.setParam('Presolve', 0)  # Disable presolve
                        
                    # Remove old constraints related to z if they exist
                    for constr in ["abs", "mva"] : # ["abs_pos", "abs_neg"]:
                        try:
                            self.milp.model.remove(self.milp.model.getConstrByName(constr))
                        except:  
                            pass  
                    
                    if self.power_system.line_flow_formulation == 'ptdf':
                        # all line flows:
                        line_flows = self.milp.model.addMVar((self.mpc.nl + self.mpc.nl_tf), lb = -GRB.INFINITY, ub = GRB.INFINITY, vtype=GRB.CONTINUOUS, name="all_line_flows")                    
                        self.milp.model.addConstr( line_flows == self.mpc.ptdf @ (self.power_system.M_g_nsb @ (self.power_system.pg_pred * self.mpc.pg_delta.reshape(-1, 1) / self.mpc.baseMVA)
                                    - self.power_system.M_d_nsb @  ((self.power_system.pd_NN * self.mpc.pd_delta.reshape(-1, 1) + self.mpc.pd_min.reshape(-1, 1)) / self.mpc.baseMVA) ) )
                    elif self.power_system.line_flow_formulation == 'dcpf':
                        line_flows = self.power_system.pline
                    
                    # Auxiliary variable for the absolute value of pline[i] * baseMVA
                    line_flow_abs = self.milp.model.addVar(lb=0, ub = GRB.INFINITY, name="line_flow_abs")
                    line_flow_mva = self.milp.model.addVar(lb=-GRB.INFINITY, ub= GRB.INFINITY, vtype=GRB.CONTINUOUS, name="line_flow_mva")
                    
                    self.milp.model.addConstr(line_flow_mva == line_flows[i] * self.mpc.baseMVA, name="scale")
                    self.milp.model.addConstr(line_flow_abs == gp.abs_(line_flow_mva), name= "abs")
                    
                    if i < self.mpc.nl:
                        obj = (line_flow_abs - self.mpc.rate_a[i])
                    else:
                        obj = (line_flow_abs - self.mpc.rate_a_tf[i - self.mpc.nl])

                    # Set the objective to maximize the violation 
                    self.milp.model.setObjective(obj, GRB.MAXIMIZE)
                    self.milp.model.optimize()
                    
                    # extract solution
                    self.v_line_time[i] = self.milp.model.Runtime                        
                    try:
                        # Check if the first variable has an assigned value
                        if hasattr(self.power_system.pd_NN[0, 0], "x"):
                            self.pl_values[2 * self.mpc.ng + i, :] = [self.power_system.pd_NN[j, 0].x for j in range(self.mpc.nloads)]
                        else:
                            print(f"Line {i}, Model is not optimal. Cannot access variable values. Status: {self.milp.model.status}")
                            self.pl_values[2 * self.mpc.ng + i, :] = np.nan  # Fill with NaN
                            print("Optimal Objective Value:", self.milp.model.ObjVal)
                            print("Dual Bound (Best Bound):", self.milp.model.ObjBound)
                            print("MIP Gap:", self.milp.model.MIPGap)
                    except (AttributeError, gp.GurobiError) as e:
                        print(f"Line {i}, Error accessing variable values: {str(e)}, model status: {self.milp.model.status}")
                        self.pl_values[2 * self.mpc.ng + i, :] = np.nan  # Fill with NaN in case of an error
                        print("Optimal Objective Value:", self.milp.model.ObjVal)
                        print("Dual Bound (Best Bound):", self.milp.model.ObjBound)
                        print("MIP Gap:", self.milp.model.MIPGap)
                    
                    # Check if the MILP solution is not optimal
                    if self.milp.model.status != GRB.OPTIMAL:  

                        # Attempt to rerun the optimization with a conservative presolve strategy
                        self.milp.model.presolve()
                        self.milp.model.optimize()

                        # If the issue persists after re-optimization, abort with a detailed error message
                        if self.milp.model.status not in [GRB.OPTIMAL, GRB.INFEASIBLE, GRB.USER_OBJ_LIMIT]:  
                            print(
                                f"Line {i}: Issue with solving MILP PLINE. "
                                f"Solver Status: {self.milp.model.status}, "
                                f"Objective Value: {getattr(self.milp.model, 'objVal', 'N/A')}, "
                                f"MIP Gap: {getattr(self.milp.model, 'MIPGap', 'N/A')}"
                                                                        )
                    
                    if self.milp.model.status == GRB.OPTIMAL:
                                            
                        # Double check that the computed neural network input leads to the worst-case violation
                        mpc_test = self.mpc.mpc
                        pd_NN_values = np.array([var.X for var in (self.power_system.pd_NN)])
                        pg_pred_nn = self.nn_prediction.nn_prediction_fixed_ReLUs(pd_NN_values[:,0])
                        pg_pred_values = np.array([var.X for var in (self.power_system.pg_pred)])
                        
                        if np.sum(np.abs(pg_pred_nn - pg_pred_values[:,0])) > 1e-2:
                            if runs == 0:
                                print(f"Branch {i}, With Presolve: Mismatch in neural network prediction -- PLINE by: ", np.sum(np.abs(pg_pred_nn - pg_pred_values[:,0])) )
                            else:
                                print(f"Branch {i}, Without Presolve: Mismatch in neural network prediction -- PLINE: ", np.sum(np.abs(pg_pred_nn - pg_pred_values[:,0])) )
                            solved = False
                        else:
                            solved = True
                            
                        if solved:
                            mpc_test.load['p_mw'] = pd_NN_values[:,0] * self.mpc.pd_delta + self.mpc.pd_min
                            mpc_test.gen['p_mw'] = self.mpc.pg_delta * pg_pred_nn
                            
                            pp.rundcpp(mpc_test)
                            if i < self.mpc.nl:
                                pline_viol_max = max(np.array(abs(mpc_test.res_line["p_to_mw"])[i] - self.mpc.rate_a[i]), 0)
                            else:
                                pline_viol_max = max(np.array(abs(mpc_test.res_trafo["p_hv_mw"])[i - self.mpc.nl] - self.mpc.rate_a_tf[i - self.mpc.nl]), 0)
                            
                            if abs(pline_viol_max - (obj.getValue())) > 10e-3 and pline_viol_max > 0 and obj.getValue() > 0:
                                if runs == 0:
                                    print(f"Branch {i}, With Presolve: Mismatch in worst-case violation -- PLINE (Run: {runs}), Violation dcpf: {pline_viol_max}, Violation optimization: {obj.getValue()}")
                                else:
                                    print(f"Branch {i}, Without Presolve: Mismatch in worst-case violation -- PLINE (Run: {runs}), Violation dcpf: {pline_viol_max}, Violation optimization: {obj.getValue()}")
                                solved = False
                            
                            if self.milp.model.MIPGap > 10e-2:
                                raise Exception("MILP gap larger than 10^-2")
                            
                            self.v_line_wc[i] = obj.getValue()
                            #self.milp.model.setParam("BestBdStop", -1 * v.x)
                    
                    elif self.milp.model.status != GRB.OPTIMAL:
                        solved = True
                        self.v_line_wc[i] = obj.getValue() #self.milp.model.objBound - 1e-4
                      
                    # print(f"worst-case violation for line {i}: ", self.v_line_wc[i])  

            
        self.v_info["v_line_time"] = np.sum(self.v_line_time)
        self.v_info["v_line_wc"] = np.max(self.v_line_wc)
        self.v_info["v_line_ID"] = np.argmax(self.v_line_wc)



class WorstCaseAnalyzer:
    """
    Simplifies the process of running worst-case violation analysis.
    """

    def __init__(self, case_name, case_path, path_input, dataset_type):
        """
        Initializes the required models and prepares data.

        Parameters:
        - case_name: str, name of the power system case
        - case_path: str, path to the case file
        - path_input: str, path to input data
        - dataset_type: str, type of dataset used
        """
        # Load tightened ReLU bounds
        self.zk_hat_min = scipy.io.loadmat(f'{path_input}/zk_hat_min.mat')['zk_hat_min']
        self.zk_hat_max = scipy.io.loadmat(f'{path_input}/zk_hat_max.mat')['zk_hat_max']

        # Load ReLU stability (active/inactive ReLUs)
        self.ReLU_always_inactive = scipy.io.loadmat(f'{path_input}/ReLU_always_inactive.mat')['ReLU_always_inactive']
        self.ReLU_always_active = scipy.io.loadmat(f'{path_input}/ReLU_always_active.mat')['ReLU_always_active']
        
        # Load network data
        self.mpc = PrepareDCOPFData(case_name, case_path)
        self.data_loader = DataLoader(path_input, dataset_type)
        self.nn_prediction = NeuralNetworkPrediction(self.data_loader, self.ReLU_always_active, self.ReLU_always_inactive)

        # Initialize models
        self.milp = MILPModel()
        self.power_system = PowerSystemModel(self.mpc, self.milp, self.data_loader.input_nn, self.data_loader.output_nn, 'dcpf')
        self.neural_network = NeuralNetwork(
            self.milp, self.power_system.pd_NN, self.power_system.pg_pred, self.data_loader,
            self.zk_hat_min, self.zk_hat_max, self.ReLU_always_active, self.ReLU_always_inactive
        )

        # Create the analyzer
        self.analyzer = WorstCaseViolationAnalyzer(self.milp, self.mpc, self.power_system, self.neural_network, self.nn_prediction)

    def run_analysis(self):
        """
        Runs worst-case violation analysis and prints formatted results.
        """
        wc_results = self.analyzer.v_info  # Retrieve worst-case violation info

        print("\nWorst-Case Summary:")
        print("-" * 50)
        for key, value in wc_results.items():
            if isinstance(value, float):  # Format floats with 3 decimal places
                print(f"{key.replace('_', ' '):<30}: {value:>10.3f}")
            else:  # Print integers as they are
                print(f"{key.replace('_', ' '):<30}: {value:>10}")


# Run the analysis with a single function call
if __name__ == "__main__":
    
    # Get the current working directory
    parent_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

    # Define the cases
    case_name = 'case118_DCOPF'
    case_iter = '1'
    case_path = os.path.join(parent_directory, "python_based", "test_networks", "network_modified")

    # define the neural network
    path_input = os.path.join(parent_directory, "python_based", "trained_nns", case_name, case_iter)

    # input domain reduction
    delta = 0

    dataset_type = 'all'
    analyzer = WorstCaseAnalyzer(case_name, case_path, path_input, dataset_type)
    analyzer.run_analysis()



#### for debugging...

# if self.milp.model.status not in [2, 15]:
#     print(self.milp.model.status)
#     self.milp.model.computeIIS()
#     self.milp.model.write("C:/Users/bagir/OneDrive - Danmarks Tekniske Universitet/Dokumenter/1) Projects/3) AI Effect/5) Verification/0) DCOPF/python_based/functions/infeasible_model.ilp")
# else:
#     self.milp.model.write("C:/Users/bagir/OneDrive - Danmarks Tekniske Universitet/Dokumenter/1) Projects/3) AI Effect/5) Verification/0) DCOPF/python_based/functions/model.lp")
