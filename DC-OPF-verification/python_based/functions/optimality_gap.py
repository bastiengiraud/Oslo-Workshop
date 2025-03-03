"""
based on E4_worst_case_distance_optimality_milp.m & copmute_worst_case_distance_sub_optimality.m

"""

import numpy as np
import pandapower as pp
import pandapower.converter as pc
import pandas as pd
import scipy.io
import gurobipy as gp
from gurobipy import GRB
import os

from statistical_bound import PrepareDCOPFData, DataLoader, NeuralNetworkPrediction
from exact_bound import MILPModel, PowerSystemModel, NeuralNetwork

os.environ['GRB_LICENSE_FILE'] = 'C:/Users/bagir/gurobi.lic'



class KarushKuhnTucker:
    def __init__(self, mpc, milp, power_system):
        self.milp = milp
        self.mpc = mpc
        self.power_system = power_system
        
        # Add variables
        self.pg_kkt = milp.model.addMVar((mpc.ng + 1, 1), lb = -GRB.INFINITY, ub = GRB.INFINITY, vtype=GRB.CONTINUOUS, name="pg_kkt") # [p.u.]
        self.theta_kkt = milp.model.addMVar((mpc.nb, 1), lb = -GRB.INFINITY, ub = GRB.INFINITY, vtype=GRB.CONTINUOUS, name="theta_kkt")
        self.gencost = (self.mpc.gencost).reshape(-1, 1)
        
        # add KKT conditions
        self.primal()
        self.dual()
        self.complementarity()
        self.stationarity()
        
    def primal(self):
        """
        Primal Constraints DC-OPF
        Constraint 1: M_g * pg - M_d * (pd_NN * pd_delta / mpc.baseMVA + pd_min / mpc.baseMVA) == Bbus * theta
        
        """
        self.milp.model.addConstr(self.mpc.M_g @ (self.pg_kkt) - self.mpc.M_d @  ((self.power_system.pd_NN * self.mpc.pd_delta.reshape(-1, 1) + self.mpc.pd_min.reshape(-1, 1)) / self.mpc.baseMVA)
                                    == self.mpc.B @ self.theta_kkt, "power_balance")

        # Constraint 2: plinemin <= Bline * theta <= plinemax
        self.milp.model.addConstr((self.mpc.Bline @ self.theta_kkt) >= (-1) * np.hstack(((self.mpc.rate_a / self.mpc.baseMVA),(self.mpc.rate_a_tf / self.mpc.baseMVA))).reshape(-1, 1), name="line_flow_min")
        self.milp.model.addConstr((self.mpc.Bline @ self.theta_kkt) <= np.hstack(((self.mpc.rate_a / self.mpc.baseMVA),(self.mpc.rate_a_tf / self.mpc.baseMVA))).reshape(-1, 1), name="line_flow_max")

        # Constraint 3: pgmin <= pg <= pgmax
        self.milp.model.addConstr(self.pg_kkt >= np.hstack((self.mpc.pg_slack_min / self.mpc.baseMVA, self.mpc.pg_min / self.mpc.baseMVA)).reshape(-1, 1), name="pg_min")
        self.milp.model.addConstr(self.pg_kkt <= np.hstack((self.mpc.pg_slack_max / self.mpc.baseMVA, self.mpc.pg_max / self.mpc.baseMVA)).reshape(-1, 1), name="pg_max")

        # Constraint 4: theta at slack bus equals 0
        slack_bus = self.mpc.slack  
        self.milp.model.addConstr(self.theta_kkt[slack_bus, 0] == 0, name="slack_bus_constraint")
    
    def dual(self):
        """
        Dual variables (Lagrange multipliers)
        note that the dual feasibility constraints are already included by setting lb = 0
        
        """
        
        self.mu_g_min = self.milp.model.addMVar((self.mpc.ng + 1, 1), lb=0.0, ub = GRB.INFINITY, vtype=GRB.CONTINUOUS, name="mu_g_min")
        self.mu_g_max = self.milp.model.addMVar((self.mpc.ng + 1, 1), lb=0.0, ub = GRB.INFINITY, vtype=GRB.CONTINUOUS, name="mu_g_max")
        self.mu_line_min = self.milp.model.addMVar((self.mpc.nl + self.mpc.nl_tf, 1), lb=0.0, ub = GRB.INFINITY, vtype=GRB.CONTINUOUS, name="mu_line_min")
        self.mu_line_max = self.milp.model.addMVar((self.mpc.nl + self.mpc.nl_tf, 1), lb=0.0, ub = GRB.INFINITY, vtype=GRB.CONTINUOUS, name="mu_line_max")
        self.lambda_vars = self.milp.model.addMVar((self.mpc.nb, 1), lb = -GRB.INFINITY, ub = GRB.INFINITY, vtype=GRB.CONTINUOUS, name="lambda")
        
    def complementarity(self):
        
        """
        Complementary Slackness Constraints (enforcing non-zero slack when dual variables are active)
        M_pg_min * b_pg_min = pgmin * (1 - b_pg_min) 
        M_pg_max * b_pg_max = pgmax * (1 - b_pg_max)
        M_pline_min * b_pline_min = plinemin * (1 - b_pline_min)
        M_pline_max * b_pline_max = plinemax * (1 - b_pline_max)

        Implement complementary slackness constraints using big-M reformulation
        a) Fortuny-Amat McCarl linearization (big-M reformulation)
        
        """
        
        # Complementary slackness variables (binary variables for activation of limits)
        self.M_pg_min = 1e4
        self.b_pg_min = self.milp.model.addMVar((self.mpc.ng + 1, 1), vtype=GRB.BINARY, name="b_pg_min")
        self.M_pg_max = 1e4
        self.b_pg_max = self.milp.model.addMVar((self.mpc.ng + 1, 1), vtype=GRB.BINARY, name="b_pg_max")
        self.M_pline_min = 1e4
        self.b_pline_min = self.milp.model.addMVar((self.mpc.nl + self.mpc.nl_tf, 1), vtype=GRB.BINARY, name="b_pline_min")
        self.M_pline_max = 1e4
        self.b_pline_max = self.milp.model.addMVar((self.mpc.nl + self.mpc.nl_tf, 1), vtype=GRB.BINARY, name="b_pline_max")
        
        # For generator limits
        self.milp.model.addConstr(np.hstack((self.mpc.pg_slack_min / self.mpc.baseMVA, self.mpc.pg_min / self.mpc.baseMVA)).reshape(-1,1) - self.pg_kkt >= -self.b_pg_min * self.M_pg_min, name="pg_min_slack")
        self.milp.model.addConstr(self.mu_g_min <= (1 - self.b_pg_min) * self.M_pg_min, name="mu_g_min_slack")
        self.milp.model.addConstr(self.pg_kkt - np.hstack((self.mpc.pg_slack_max / self.mpc.baseMVA, self.mpc.pg_max / self.mpc.baseMVA)).reshape(-1,1) >= -self.b_pg_max * self.M_pg_max, name="pg_max_slack")
        self.milp.model.addConstr(self.mu_g_max <= (1 - self.b_pg_max) * self.M_pg_max, name="mu_g_max_slack")

        # For line flow limits
        self.milp.model.addConstr((-1) * np.hstack(((self.mpc.rate_a / self.mpc.baseMVA),(self.mpc.rate_a_tf / self.mpc.baseMVA))).reshape(-1,1) - (self.mpc.Bline @ self.theta_kkt) >= -self.b_pline_min * self.M_pline_min, name="pline_min_slack")
        self.milp.model.addConstr(self.mu_line_min <= (1 - self.b_pline_min) * self.M_pline_min, name="mu_line_min_slack")
        self.milp.model.addConstr((self.mpc.Bline @ self.theta_kkt) - np.hstack(((self.mpc.rate_a / self.mpc.baseMVA),(self.mpc.rate_a_tf / self.mpc.baseMVA))).reshape(-1,1) >= -self.b_pline_max * self.M_pline_max, name="pline_max_slack")
        self.milp.model.addConstr(self.mu_line_max <= (1 - self.b_pline_max) * self.M_pline_max, name="mu_line_max_slack")

    def stationarity(self):

        # Stationarity Constraints
        self.milp.model.addConstr(self.gencost - self.mu_g_min + self.mu_g_max + (self.mpc.M_g.T @ self.lambda_vars) == 0, name="stationarity_1")
        self.milp.model.addConstr(-self.mpc.Bline.T @ self.mu_line_min + self.mpc.Bline.T @ self.mu_line_max - self.mpc.B @ self.lambda_vars == 0, name="stationarity_2")




class SubOptimality:
    def __init__(self, mpc, milp, power_system, nn_prediction, kkt_conditions):
        self.kkt_conditions = kkt_conditions
        self.mpc = mpc
        self.milp = milp
        self.power_system = power_system
        self.nn_prediction = nn_prediction
        
        # placeholders
        self.milp_gap_v_dist_max = np.zeros(self.mpc.ng + 1)
        self.milp_exact_v_dist_max = np.zeros(self.mpc.ng + 1)

        self.v_opt_max = np.zeros(1)
        self.v_opt_time = np.zeros(1)

        # Initialize parameters
        self.max_compl = np.zeros(self.mpc.ng + 2)
        self.max_dual = np.zeros(self.mpc.ng + 2)

        self.v_dist_time = np.zeros(self.mpc.ng + 1)
        self.v_dist_max = np.zeros(self.mpc.ng + 1)
        
        self.time_milp_max = 1e5

        
        self.v_info = {}
        
        self.worst_case_distance()
        self.worst_case_cost()
        
    ####### compute the maximal distance between the worst-case generator set points and the optimal setpoints
    def worst_case_distance(self):

        self.milp.model.Params.TimeLimit = self.time_milp_max  
        #milp.model.setParam("BestBdStop", 1e9)     
        
        # Define constraints and objective
        slack_generation_constraint = None
        
        # add auxiliary variables
        distance = self.milp.model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="distance")
        distance_abs = self.milp.model.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="distance_abs")

        for g in range(self.mpc.ng + 1):
            
            self.milp.model.setParam('Presolve', -1)  # Automatic presolve
            
            # remove any unnecessary constraints
            for constr in ["distance_constr", "distance_abs_constr", "slack_generation"] : 
                try:
                    self.milp.model.remove(self.milp.model.getConstrByName(constr))
                except:
                    pass

            if g == self.mpc.ng:
                # add slack bus constraint
                slack_generation_constraint = self.milp.model.addConstr(
                    gp.quicksum((self.power_system.pd_NN[i, 0] * self.mpc.pd_delta[i] + self.mpc.pd_min[i]) / self.mpc.baseMVA for i in range(self.power_system.pd_NN.shape[0])) ==
                    gp.quicksum(
                        self.power_system.pg_slack +
                        gp.quicksum(self.power_system.pg_pred[j, 0] * self.mpc.pg_delta[j] / self.mpc.baseMVA for j in range(self.power_system.pg_pred.shape[0]))
                    ),
                    name="slack_generation"
                )
                
                
                # take absolute of the objective. here, both pg_slack and pg_kkt are p.u., and not min-max scaled
                self.milp.model.addConstr(distance == (self.power_system.pg_slack - self.kkt_conditions.pg_kkt[0, 0]) / (self.mpc.pg_slack_delta / self.mpc.baseMVA), name="distance_constr")
                self.milp.model.addConstr(distance_abs == gp.abs_(distance), name= "distance_abs_constr")
                
                obj = distance_abs
                
            else:
                # Remove the slack generation constraint if it was added
                if slack_generation_constraint is not None:
                    self.milp.model.remove(slack_generation_constraint)
                    slack_generation_constraint = None
                
                # take absolute of the objective. pg_pred is normalized, pg_kkt is p.u. but not min-max scaled
                self.milp.model.addConstr(distance == self.power_system.pg_pred[g, 0] - (self.kkt_conditions.pg_kkt[g+1, 0] * self.mpc.baseMVA / self.mpc.pg_max[g] ), name="distance_constr")
                self.milp.model.addConstr(distance_abs == gp.abs_(distance), name= "distance_abs_constr")
                
                obj = distance_abs 
                    
            # set objective
            self.milp.model.setObjective(obj, GRB.MAXIMIZE)
            #self.milp.model.setParam("DualReductions", 0)
            
            # Solve the model
            self.milp.model.setParam("OutputFlag", 0)
            self.milp.model.optimize()
                
            # Check if the self.milp solution is not optimal
            if self.milp.model.status != GRB.OPTIMAL:  

                # Attempt to rerun the optimization with a conservative presolve strategy
                self.milp.model.presolve()
                self.milp.model.optimize()

                # If the issue persists after re-optimization, abort with a detailed error message
                if self.milp.model.status not in [GRB.OPTIMAL]:  # , GRB.INFEASIBLE, GRB.USER_OBJ_LIMIT, GRB.UNBOUNDED
                    raise Exception(
                        f"Generator {g}: KKT MILP solve issue "
                        f"Solver Status: {self.milp.model.status}, " # 2 = optimal, 3 = infeasible, 4 = inf or unb, 5 = unbounded, 9 = time limit, 15 = user objective limit
                        f"Objective Value: {getattr(self.milp.model, 'objVal', 'N/A')}, "
                        f"MIP Gap: {getattr(self.milp.model, 'MIPGap', 'N/A')}"
                                                                )

            # Solution validation
            if self.milp.model.Status == GRB.OPTIMAL:
                
                # Extract primal and dual variables
                pg_kkt_values = np.array([v.x for v in self.kkt_conditions.pg_kkt])
                theta_kkt_values = np.array([v.x for v in self.kkt_conditions.theta_kkt])
                
                # Validate primal bounds, we need to double check that the a) primal variables do not violate the derived bounds in the kkt
                tolerance = 1e-3
                if (
                    np.any(self.mpc.pg_min / self.mpc.baseMVA - pg_kkt_values >= (self.kkt_conditions.M_pg_min - tolerance)) or
                    np.any(pg_kkt_values -self.mpc.pg_max / self.mpc.baseMVA >= (self.kkt_conditions.M_pg_min - tolerance)) or
                    np.any(((-1) * self.mpc.rate_a / self.mpc.baseMVA) - (self.mpc.Bline @ theta_kkt_values) >= (self.kkt_conditions.M_pg_min - tolerance)) or
                    np.any((self.mpc.Bline @ theta_kkt_values) - (self.mpc.rate_a / self.mpc.baseMVA) >= (self.kkt_conditions.M_pg_min - tolerance))
                ):
                    raise ValueError("Primal bounds are binding")

                # Validate dual bounds, b) dual variables do not violate the derived bounds in the kkt
                mu_g_min_values = np.array([v.x for v in self.kkt_conditions.mu_g_min])
                mu_g_max_values = np.array([v.x for v in self.kkt_conditions.mu_g_max])
                mu_line_min_values = np.array([v.x for v in self.kkt_conditions.mu_line_min])
                mu_line_max_values = np.array([v.x for v in self.kkt_conditions.mu_line_max])

                self.max_dual[g] = np.max(np.max(np.vstack((mu_g_min_values, mu_g_max_values, mu_line_min_values, mu_line_max_values))))
                if (
                    np.any(mu_g_min_values >= (self.kkt_conditions.M_pg_min - tolerance)) or
                    np.any(mu_g_max_values >= (self.kkt_conditions.M_pg_min - tolerance)) or
                    np.any(mu_line_min_values >= (self.kkt_conditions.M_pg_min - tolerance)) or
                    np.any(mu_line_max_values >= (self.kkt_conditions.M_pg_min - tolerance))
                ):
                    raise ValueError("Dual bounds are binding")

                # Validate complementary slackness, c) the complementary slackness conditions hold        
                self.max_compl[g] = np.max(np.abs(np.vstack(((np.hstack((self.mpc.pg_slack_min / self.mpc.baseMVA, self.mpc.pg_min / self.mpc.baseMVA)).reshape(-1,1) - pg_kkt_values) * mu_g_min_values, 
                                                    (pg_kkt_values - np.hstack((self.mpc.pg_slack_max / self.mpc.baseMVA,self.mpc.pg_max / self.mpc.baseMVA)).reshape(-1,1)) * mu_g_max_values, 
                                                    (np.hstack((((-1) * self.mpc.rate_a / self.mpc.baseMVA),((-1) * self.mpc.rate_a_tf / self.mpc.baseMVA))).reshape(-1,1) - (self.mpc.Bline @ theta_kkt_values).reshape(-1,1)) * mu_line_min_values, 
                                                    ((self.mpc.Bline @ theta_kkt_values).reshape(-1,1) - np.hstack(((self.mpc.rate_a / self.mpc.baseMVA),(self.mpc.rate_a_tf / self.mpc.baseMVA))).reshape(-1,1)) * mu_line_max_values))))

                # Check the complementary slackness condition
                if self.max_compl[g] > 1e-3:
                    raise ValueError('Complementary slackness conditions do not hold -- improve numerical accuracy')

                # Objective value and MILP gap
                self.v_dist_max[g] = self.milp.model.ObjVal
                self.milp_gap_v_dist_max[g] = self.milp.model.MIPGap

                # Compare with neural network prediction
                pd_nn_values = np.array([var.X for var in (self.power_system.pd_NN)])
                pg_pred_nn = self.nn_prediction.nn_prediction_fixed_ReLUs(pd_nn_values[:,0])
                pg_pred_values = np.array([var.X for var in (self.power_system.pg_pred)])

                if np.sum(np.abs(pg_pred_nn - pg_pred_values[:,0])) > 1e-2:
                    print(np.abs(pg_pred_nn - pg_pred_values[:,0]))
                    print("Neural network prediction and MILP do not match -- abort!")
                    # raise ValueError("Neural network prediction and MILP do not match")

            elif self.milp.model.Status != GRB.OPTIMAL:
                self.v_dist_max[g] = self.milp.model.objBound - 1e-4
                self.milp_gap_v_dist_max[g] = -1

                
            print(f"this is the max distance for generator {g}: ", self.v_dist_max[g])
            # print(self.milp.model.Status)


            self.v_dist_time[g] = self.milp.model.Runtime
            if self.milp.model.Status == GRB.OPTIMAL:
                # here we need to build the check which compares that for the identified system loading, the solution produced by the KKTs is
                # actually the optimal solution to the DC-OPF (we use rundcopf here; note that this is computationally much cheaper than solving the MILP
                # so we can do it for every one) Note that this check is for debugging purposes only
                
                # set the load
                mpc_test = self.mpc.mpc
                mpc_test.load['p_mw'] = pd_nn_values[:,0] * self.mpc.pd_delta + self.mpc.pd_min
                
                # solve the dc-opf
                pp.rundcopp(mpc_test)
                pg_dcopf = np.array(mpc_test.res_gen['p_mw'])/self.mpc.baseMVA
                
                # extract the active generator dispatch and compare, throw an error if they do not match
                if sum(np.abs(pg_kkt_values[1:, 0] - pg_dcopf)) > 1e-2:
                    print('KKT solution and rundcopf do not match -- abort!')
                    print(pg_kkt_values[1:, 0])
                    print(pg_dcopf)
                    #raise ValueError('KKT solution and rundcopf do not match')
                else:
                    print('KKT solution and rundcopf do match -- continue')
                
        # remove constraints and clean up model
        for constr in ["distance_constr", "distance_abs_constr", "slack_generation"] : 
            try:
                self.milp.model.remove(self.milp.model.getConstrByName(constr))
            except:
                pass

    def worst_case_cost(self):
 
        # compute the maximum error on the objective function value
        self.milp.model.addConstr(
            gp.quicksum((self.power_system.pd_NN[i, 0] * self.mpc.pd_delta[i] + self.mpc.pd_min[i]) / self.mpc.baseMVA for i in range(self.power_system.pd_NN.shape[0])) ==
            gp.quicksum(
                self.power_system.pg_slack +
                gp.quicksum(self.power_system.pg_pred[j, 0] * self.mpc.pg_delta[j] / self.mpc.baseMVA for j in range(self.power_system.pg_pred.shape[0]))
            ),
            name="slack_generation"
        )

        obj = self.kkt_conditions.gencost[:,0] @ ( gp.vstack((self.power_system.pg_slack * self.mpc.baseMVA, (self.power_system.pg_pred * self.mpc.pg_delta.reshape(-1, 1))))  - self.kkt_conditions.pg_kkt * self.mpc.baseMVA)[:,0]

        self.milp.model.Params.TimeLimit = self.time_milp_max  
        # self.milp.model.Params.BestBdStop = 100000       

        # Solve the model
        self.milp.model.setObjective(obj, GRB.MAXIMIZE)
        self.milp.model.optimize()

        # Error handling and re-optimization with different presolve options
        if self.milp.model.Status not in [GRB.OPTIMAL]:
            self.milp.model.setParam("Presolve", 1)
            self.milp.model.optimize()
            if self.milp.model.Status not in [GRB.OPTIMAL]:
                self.milp.model.setParam("Presolve", 0)
                self.milp.model.optimize()
                if self.milp.model.Status not in [GRB.OPTIMAL]:
                    raise RuntimeError("KKT MILP solve issue")


        if self.milp.model.Status == GRB.OPTIMAL:
            self.v_opt_max[0] = obj.getValue()
            milp_exact_v_opt_max = 1
            
            # Extract primal and dual variables
            pg_kkt_values = np.array([v.x for v in self.kkt_conditions.pg_kkt])
            theta_kkt_values = np.array([v.x for v in self.kkt_conditions.theta_kkt])
            
            # Validate primal bounds, we need to double check that the a) primal variables do not violate the derived bounds in the kkt
            tolerance = 1e-3
            if (
                np.any((self.mpc.pg_min / self.mpc.baseMVA) - pg_kkt_values >= (self.kkt_conditions.M_pg_min - tolerance)) or
                np.any(pg_kkt_values -self.mpc.pg_max / self.mpc.baseMVA >= (self.kkt_conditions.M_pg_min - tolerance)) or
                np.any(((-1) * self.mpc.rate_a / self.mpc.baseMVA) - (self.mpc.Bline @ theta_kkt_values) >= (self.kkt_conditions.M_pg_min - tolerance)) or
                np.any((self.mpc.Bline @ theta_kkt_values) - (self.mpc.rate_a / self.mpc.baseMVA) >= (self.kkt_conditions.M_pg_min - tolerance))
            ):
                raise ValueError("Primal bounds are binding")

            # Validate dual bounds, b) dual variables do not violate the derived bounds in the kkt
            mu_g_min_values = np.array([v.x for v in self.kkt_conditions.mu_g_min])
            mu_g_max_values = np.array([v.x for v in self.kkt_conditions.mu_g_max])
            mu_line_min_values = np.array([v.x for v in self.kkt_conditions.mu_line_min])
            mu_line_max_values = np.array([v.x for v in self.kkt_conditions.mu_line_max])

            self.max_dual[self.mpc.ng + 1] = np.max(np.max(np.vstack((mu_g_min_values, mu_g_max_values, mu_line_min_values, mu_line_max_values))))
            if (
                np.any(mu_g_min_values >= (self.kkt_conditions.M_pg_min - tolerance)) or
                np.any(mu_g_max_values >= (self.kkt_conditions.M_pg_min - tolerance)) or
                np.any(mu_line_min_values >= (self.kkt_conditions.M_pg_min - tolerance)) or
                np.any(mu_line_max_values >= (self.kkt_conditions.M_pg_min - tolerance))
            ):
                raise ValueError("Dual bounds are binding")

            # Validate complementary slackness, c) the complementary slackness conditions hold        
            self.max_compl[self.mpc.ng + 1] = np.max(np.abs(np.vstack(((np.hstack((self.mpc.pg_slack_min / self.mpc.baseMVA, self.mpc.pg_min / self.mpc.baseMVA)).reshape(-1,1) - pg_kkt_values) * mu_g_min_values, 
                                                    (pg_kkt_values - np.hstack((self.mpc.pg_slack_max / self.mpc.baseMVA,self.mpc.pg_max / self.mpc.baseMVA)).reshape(-1,1)) * mu_g_max_values, 
                                                    (np.hstack((((-1) * self.mpc.rate_a / self.mpc.baseMVA),((-1) * self.mpc.rate_a_tf / self.mpc.baseMVA))).reshape(-1,1) - (self.mpc.Bline @ theta_kkt_values).reshape(-1,1)) * mu_line_min_values, 
                                                    ((self.mpc.Bline @ theta_kkt_values).reshape(-1,1) - np.hstack(((self.mpc.rate_a / self.mpc.baseMVA),(self.mpc.rate_a_tf / self.mpc.baseMVA))).reshape(-1,1)) * mu_line_max_values))))

            # Check the complementary slackness condition
            if self.max_compl[self.mpc.ng + 1] > 1e-3:
                raise ValueError('Complementary slackness conditions do not hold -- improve numerical accuracy')

            
            # Compare with neural network prediction
            pd_nn_values = np.array([var.X for var in (self.power_system.pd_NN)])
            pg_pred_nn = self.nn_prediction.nn_prediction_fixed_ReLUs(pd_nn_values[:,0])
            pg_pred_values = np.array([var.X for var in (self.power_system.pg_pred)])

            if np.sum(np.abs(pg_pred_nn - pg_pred_values[:,0])) > 1e-3:
                print(np.abs(pg_pred_nn - pg_pred_values[:,0]))
                print("Neural network prediction and MILP do not match -- abort!")
                # raise ValueError("Neural network prediction and MILP do not match")
        
        
        else:
            self.v_opt_max[0] = self.milp.model.objBound - 1e-3
            milp_exact_v_opt_max = 0


        self.v_opt_time[0] = self.milp.model.Runtime
        if self.milp.model.Status == GRB.OPTIMAL:
                    
            # set the load
            mpc_test = self.mpc.mpc
            mpc_test.load['p_mw'] = pd_nn_values[:,0] * self.mpc.pd_delta + self.mpc.pd_min
            
            # print(mpc_test.load['p_mw'])
            
            # solve the dc-opf
            pp.rundcopp(mpc_test)
            pg_dcopf = np.array(mpc_test.res_gen['p_mw'])/self.mpc.baseMVA
            
            # extract the active generator dispatch and compare, throw an error if they do not match
            if sum(np.abs(pg_kkt_values[1:, 0] - pg_dcopf)) > 1e-3:
                print('KKT solution and rundcopf do not match -- abort!')
                #raise ValueError('KKT solution and rundcopf do not match')
            else:
                print('KKT solution and rundcopf do match -- continue')

        # reference cost
        mpc_optimal = self.mpc.mpc
        pp.rundcopp(mpc_optimal)
        self.cost_ref = np.sum(np.array(mpc_optimal.res_cost))

        self.v_info["v_dist_time"] = max(self.v_dist_time)
        self.v_info["v_dist_wc"] = max(self.v_dist_max) * 100
        self.v_info["v_dist_ID"] = np.argmax(self.v_dist_max)

        # self.v_info["milp_gap_v_dist_max"] = max(self.milp_gap_v_dist_max)
        # self.v_info["milp_exact_v_dist_max"] = max(self.milp_exact_v_dist_max)

        # self.v_info["max_compl"] = max(self.max_compl)
        # self.v_info["max_dual"] = max(self.max_dual)

        # self.v_info["v_opt_max"] = max(self.v_opt_max) 
        self.v_info["v_opt_wc"] = max(self.v_opt_max/self.cost_ref) * 100
        self.v_info["v_opt_time"] = max(self.v_opt_time)


class SubOptimalityAnalyzer:
    """
    Simplifies the process of running suboptimality violation analysis.
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

        # initialize milp model
        self.milp = MILPModel()
        self.power_system = PowerSystemModel(self.mpc, self.milp, self.data_loader.input_nn, self.data_loader.output_nn, 'dcpf')
        self.neural_network = NeuralNetwork(
            self.milp, self.power_system.pd_NN, self.power_system.pg_pred, self.data_loader,
            self.zk_hat_min, self.zk_hat_max, self.ReLU_always_active, self.ReLU_always_inactive
        )

        self.kkt_conditions = KarushKuhnTucker(self.mpc, self.milp, self.power_system)
        self.analyzer = SubOptimality(self.mpc, self.milp, self.power_system, self.nn_prediction, self.kkt_conditions)

    def run_analysis(self):
        """
        Runs worst-case violation analysis and prints formatted results.
        """
        subopt_results = self.analyzer.v_info  # Retrieve worst-case violation info

        print("\nWorst-Case Summary:")
        print("-" * 50)
        for key, value in subopt_results.items():
            if isinstance(value, float):  # Format floats with 3 decimal places
                print(f"{key.replace('_', ' '):<30}: {value:>10.3f}")
            else:  # Print integers as they are
                print(f"{key.replace('_', ' '):<30}: {value:>10}")


if __name__ == "__main__":
    
    # Get the current working directory
    parent_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

    # Define the cases
    case_name = 'case39_DCOPF'
    case_iter = '1'
    case_path = os.path.join(parent_directory, "python_based", "test_networks", "network_modified")

    # define the neural network
    path_input = os.path.join(parent_directory, "python_based", "trained_nns", case_name, case_iter)
    dataset_type = 'all'

    # placeholders
    time_scalability = np.zeros(1)
    
    # # Error checks
    # if np.sum(np.abs(mpc.Pbusinj)) != 0.0:
    #     raise ValueError('Pbusinj not zero -- not supported')

    # if np.sum(np.abs(mpc.Pfinj)) != 0.0:
    #     raise ValueError('Pfinj not zero -- not supported')

    analyzer = SubOptimalityAnalyzer(case_name, case_path, path_input, dataset_type)
    analyzer.run_analysis()







# % create reference cost
# mpc_ref = mpc;
# mpopt = mpoption;
# mpopt.out.all =0;
# results_dcopf=rundcopf(mpc_ref,mpopt);

# v_info.v_opt_wc = v_opt_max./results_dcopf.f;


