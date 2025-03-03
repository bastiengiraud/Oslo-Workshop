"""
based on E4_worst_case_distance_optimality_milp.m & copmute_worst_case_distance_sub_optimality.m

"""

import numpy as np
import pandapower as pp
import pandapower.converter as pc
import pandas as pd
import scipy.io
import gurobipy as gp
from gurobipy import Model, GRB, LinExpr
import gurobi_ml
import copy
import os

from statistical_bound import PrepareDCOPFData, load_data, NeuralNetworkScalability
from exact_bound import MILPModel, PowerSystemModel, NeuralNetwork

os.environ['GRB_LICENSE_FILE'] = 'C:/Users/bagir/gurobi.lic'

# Get the current working directory
parent_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

# Define the cases
case_name = 'case39_DCOPF'
case_iter = '2'
case_path = os.path.join(parent_directory, "python_based", "test_networks", "network_modified")

# define the neural network
path_input = os.path.join(parent_directory, "python_based", "trained_nns", case_name, case_iter)
dataset_type = 'test'

# Number of runs (to rule out individual fluctuations)
nr_iter = 1

# placeholders
time_scalability = np.zeros(nr_iter)
time_milp_max = 10^5

# placeholder
v_dist_time_ = np.zeros((nr_iter))  
v_dist_wc_ = np.zeros((nr_iter))    
v_dist_ID_ = np.zeros((nr_iter))   
v_opt_time_ = np.zeros((nr_iter))    
v_opt_wc_ = np.zeros((nr_iter))    

# get network data
mpc = PrepareDCOPFData(case_name, case_path)
input_nn, output_nn = load_data(path_input, dataset_type)
nn_scalability = NeuralNetworkScalability(path_input, 3, input_nn, output_nn)

# Load tightened ReLU bounds
zk_hat_min = scipy.io.loadmat(f'{path_input}/zk_hat_min.mat')['zk_hat_min']
zk_hat_max = scipy.io.loadmat(f'{path_input}/zk_hat_max.mat')['zk_hat_max']

# Load ReLU stability (active/inactive ReLUs)
ReLU_always_inactive = scipy.io.loadmat(f'{path_input}/ReLU_always_inactive.mat')['ReLU_always_inactive']
ReLU_always_active = scipy.io.loadmat(f'{path_input}/ReLU_always_active.mat')['ReLU_always_active']

# initialize milp model
milp = MILPModel()
power_system = PowerSystemModel(mpc, milp, input_nn, output_nn, 'dcpf')
neural_network = NeuralNetwork(milp, power_system.pd_NN, power_system.pg_pred, nn_scalability, input_nn, output_nn, zk_hat_min, zk_hat_max, ReLU_always_active, ReLU_always_inactive)

# placeholders
v_dist_max = np.zeros(mpc.ng)
v_dist_time = np.zeros(mpc.ng)

milp_gap_v_dist_max = np.zeros(mpc.ng)
milp_exact_v_dist_max = np.zeros(mpc.ng)

v_opt_max = np.zeros(1)
v_opt_time = np.zeros(1)

# Initialize parameters
max_compl = np.zeros(mpc.ng + 1)
max_dual = np.zeros(mpc.ng + 1)

v_dist_time = np.zeros(mpc.ng + 1)
v_dist_max = np.zeros(mpc.ng + 1)

milp_gap_v_dist_max = np.zeros(mpc.ng + 1)

# Gurobi variables
pg_kkt = milp.model.addMVar((mpc.ng + 1, 1), lb = -GRB.INFINITY, ub = GRB.INFINITY, vtype=GRB.CONTINUOUS, name="pg_kkt")
theta_kkt = milp.model.addMVar((mpc.nb, 1), lb = -GRB.INFINITY, ub = GRB.INFINITY, vtype=GRB.CONTINUOUS, name="theta_kkt")

# Cost
gencost = (mpc.gencost * mpc.baseMVA).reshape(-1, 1)

# Generator power bounds
pg_min = mpc.pg_min / mpc.baseMVA
pg_max = mpc.pg_max / mpc.baseMVA
pg_slack_min = mpc.pg_slack_min / mpc.baseMVA
pg_slack_max = mpc.pg_slack_max / mpc.baseMVA

# Line flow bounds
pl_min = (-1) * mpc.rate_a / mpc.baseMVA
pl_max = mpc.rate_a / mpc.baseMVA
pl_min_tf = (-1) * mpc.rate_a_tf / mpc.baseMVA
pl_max_tf = mpc.rate_a_tf / mpc.baseMVA


# Error checks
if np.sum(np.abs(mpc.Pbusinj)) != 0.0:
    raise ValueError('Pbusinj not zero -- not supported')

if np.sum(np.abs(mpc.Pfinj)) != 0.0:
    raise ValueError('Pfinj not zero -- not supported')

########## Primal Constraints DC-OPF
# Constraint 1: M_g * pg - M_d * (pd_NN * pd_delta / mpc.baseMVA + pd_min / mpc.baseMVA) == Bbus * theta
milp.model.addConstr(mpc.M_g @ (pg_kkt) - mpc.M_d @  ((power_system.pd_NN * mpc.pd_delta.reshape(-1, 1) + mpc.pd_min.reshape(-1, 1)) / mpc.baseMVA)
                             == mpc.B @ theta_kkt, "power_balance")

# Constraint 2: plinemin <= Bline * theta <= plinemax
milp.model.addConstr((mpc.Bline @ theta_kkt) >= np.hstack((pl_min,pl_min_tf)).reshape(-1, 1), name="line_flow_min")
milp.model.addConstr((mpc.Bline @ theta_kkt) <= np.hstack((pl_max,pl_max_tf)).reshape(-1, 1), name="line_flow_max")

# Constraint 3: pgmin <= pg <= pgmax
milp.model.addConstr(pg_kkt >= np.hstack((pg_slack_min,pg_min)).reshape(-1, 1), name="pg_min")
milp.model.addConstr(pg_kkt <= np.hstack((pg_slack_max,pg_max)).reshape(-1, 1), name="pg_max")

# Constraint 4: theta at slack bus equals 0
slack_bus = mpc.slack  
milp.model.addConstr(theta_kkt[slack_bus, 0] == 0, name="slack_bus_constraint")

# ############# Dual variables (Lagrange multipliers)
# note that the dual feasibility constraints are already included by setting lb = 0
mu_g_min = milp.model.addMVar((mpc.ng + 1, 1), lb=0.0, ub = GRB.INFINITY, vtype=GRB.CONTINUOUS, name="mu_g_min")
mu_g_max = milp.model.addMVar((mpc.ng + 1, 1), lb=0.0, ub = GRB.INFINITY, vtype=GRB.CONTINUOUS, name="mu_g_max")
mu_line_min = milp.model.addMVar((mpc.nl + mpc.nl_tf, 1), lb=0.0, ub = GRB.INFINITY, vtype=GRB.CONTINUOUS, name="mu_line_min")
mu_line_max = milp.model.addMVar((mpc.nl + mpc.nl_tf, 1), lb=0.0, ub = GRB.INFINITY, vtype=GRB.CONTINUOUS, name="mu_line_max")
lambda_vars = milp.model.addMVar((mpc.nb, 1), lb = -GRB.INFINITY, ub = GRB.INFINITY, vtype=GRB.CONTINUOUS, name="lambda")

# Complementary slackness variables (binary variables for activation of limits)
M_pg_min = 10e5
b_pg_min = milp.model.addMVar((mpc.ng + 1, 1), vtype=GRB.BINARY, name="b_pg_min")
M_pg_max = 10e5
b_pg_max = milp.model.addMVar((mpc.ng + 1, 1), vtype=GRB.BINARY, name="b_pg_max")
M_pline_min = 10e5
b_pline_min = milp.model.addMVar((mpc.nl + mpc.nl_tf, 1), vtype=GRB.BINARY, name="b_pline_min")
M_pline_max = 10e5
b_pline_max = milp.model.addMVar((mpc.nl + mpc.nl_tf, 1), vtype=GRB.BINARY, name="b_pline_max")

# Stationarity Constraints
milp.model.addConstr(gencost - mu_g_min + mu_g_max + (mpc.M_g.T @ lambda_vars) == 0, name="stationarity_1")
milp.model.addConstr(-mpc.Bline.T @ mu_line_min + mpc.Bline.T @ mu_line_max - mpc.B @ lambda_vars == 0, name="stationarity_2")

# Complementary Slackness Constraints (enforcing non-zero slack when dual variables are active)
# M_pg_min * b_pg_min = pgmin * (1 - b_pg_min) 
# M_pg_max * b_pg_max = pgmax * (1 - b_pg_max)
# M_pline_min * b_pline_min = plinemin * (1 - b_pline_min)
# M_pline_max * b_pline_max = plinemax * (1 - b_pline_max)

# Implement complementary slackness constraints using big-M reformulation
# a) Fortuny-Amat McCarl linearization (big-M reformulation)

# For generator limits
milp.model.addConstr(np.hstack((pg_slack_min, pg_min)).reshape(-1,1) - pg_kkt >= -b_pg_min * M_pg_min, name="pg_min_slack")
milp.model.addConstr(mu_g_min <= (1 - b_pg_min) * M_pg_min, name="mu_g_min_slack")
milp.model.addConstr(pg_kkt - np.hstack((pg_slack_max, pg_max)).reshape(-1,1) >= -b_pg_max * M_pg_max, name="pg_max_slack")
milp.model.addConstr(mu_g_max <= (1 - b_pg_max) * M_pg_max, name="mu_g_max_slack")

# For line flow limits
milp.model.addConstr(np.hstack((pl_min,pl_min_tf)).reshape(-1,1) - (mpc.Bline @ theta_kkt) >= -b_pline_min * M_pline_min, name="pline_min_slack")
milp.model.addConstr(mu_line_min <= (1 - b_pline_min) * M_pline_min, name="mu_line_min_slack")
milp.model.addConstr((mpc.Bline @ theta_kkt) - np.hstack((pl_max,pl_max_tf)).reshape(-1,1) >= -b_pline_max * M_pline_max, name="pline_max_slack")
milp.model.addConstr(mu_line_max <= (1 - b_pline_max) * M_pline_max, name="mu_line_max_slack")

milp.model.Params.TimeLimit = time_milp_max  
#milp.model.setParam("BestBdStop", 1e9)     

####### compute the maximal distance between the worst-case generator set points and the optimal setpoints
for g in range(mpc.ng + 1):
    
    milp.model.setParam('Presolve', -1)  # Automatic presolve
    
    # Define constraints and objective
    slack_generation_constraint = None
    
    # remove any unnecessary constraints
    for constr in ["distance_constr", "distance_abs_constr", "slack_generation"] : 
        try:
            milp.model.remove(milp.model.getConstrByName(constr))
        except:
            pass

    if g == mpc.ng:
        # add slack bus constraint
        slack_generation_constraint = milp.model.addConstr(
            gp.quicksum(power_system.pd_NN[i, 0] * mpc.pd_delta[i] + mpc.pd_min[i] for i in range((power_system.pd_NN.shape[0]))) / mpc.baseMVA ==
            gp.quicksum(power_system.pg_slack + power_system.pg_pred[j, 0] * mpc.pg_delta[j] / mpc.baseMVA for j in range((power_system.pg_pred.shape[0]))),
            name="slack_generation"
        )
        
        # add auxiliary variables
        distance = milp.model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="distance")
        distance_abs = milp.model.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="distance_abs")
        
        # take absolute of the objective
        milp.model.addConstr(distance == (power_system.pg_slack - pg_kkt[0, 0]) / ((mpc.pg_slack_delta) * mpc.baseMVA), name="distance_constr")
        milp.model.addConstr(distance_abs == gp.abs_(distance), name= "distance_abs_constr")
         
        obj = distance_abs
          
    else:
        # Remove the slack generation constraint if it was added
        if slack_generation_constraint is not None:
            milp.model.remove(slack_generation_constraint)
            slack_generation_constraint = None
        
        # add auxiliary variables
        distance = milp.model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="distance")
        distance_abs = milp.model.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="distance_abs")
        
        # take absolute of the objective
        milp.model.addConstr(distance == power_system.pg_pred[g, 0] - (pg_kkt[g+1, 0] / mpc.pg_max[g]) * mpc.baseMVA, name="distance_constr")
        milp.model.addConstr(distance_abs == gp.abs_(distance), name= "distance_abs_constr")
         
        obj = distance_abs # (power_system.pg_pred[g, 0] * mpc.pg_max[g] - mpc.pg_max[g]) # 
            
    # set objective
    milp.model.setObjective(obj, GRB.MAXIMIZE)
    #milp.model.setParam("DualReductions", 0)
    
    # Solve the model
    milp.model.setParam("OutputFlag", 0)
    milp.model.optimize()
           
    # Check if the MILP solution is not optimal
    if milp.model.status != GRB.OPTIMAL:  

        # Attempt to rerun the optimization with a conservative presolve strategy
        milp.model.presolve()
        milp.model.optimize()

        # If the issue persists after re-optimization, abort with a detailed error message
        if milp.model.status not in [GRB.OPTIMAL, GRB.INFEASIBLE, GRB.USER_OBJ_LIMIT, GRB.UNBOUNDED]:  
            raise Exception(
                f"Generator {g}: KKT MILP solve issue "
                f"Solver Status: {milp.model.status}, "
                f"Objective Value: {getattr(milp.model, 'objVal', 'N/A')}, "
                f"MIP Gap: {getattr(milp.model, 'MIPGap', 'N/A')}"
                                                        )

    # Solution validation
    if milp.model.Status == GRB.OPTIMAL:
        
        # Extract primal and dual variables
        pg_kkt_values = np.array([v.x for v in pg_kkt])
        theta_kkt_values = np.array([v.x for v in theta_kkt])
        
        # Validate primal bounds, we need to double check that the a) primal variables do not violate the derived bounds in the kkt
        tolerance = 10e-3
        if (
            np.any(pg_min - pg_kkt_values >= (M_pg_min - tolerance)) or
            np.any(pg_kkt_values - pg_max >= (M_pg_min - tolerance)) or
            np.any(pl_min - (mpc.Bline @ theta_kkt_values) >= (M_pg_min - tolerance)) or
            np.any((mpc.Bline @ theta_kkt_values) - pl_max >= (M_pg_min - tolerance))
        ):
            raise ValueError("Primal bounds are binding")

        # Validate dual bounds, b) dual variables do not violate the derived bounds in the kkt
        mu_g_min_values = np.array([v.x for v in mu_g_min])
        mu_g_max_values = np.array([v.x for v in mu_g_max])
        mu_line_min_values = np.array([v.x for v in mu_line_min])
        mu_line_max_values = np.array([v.x for v in mu_line_max])

        max_dual[g] = np.max(np.max(np.vstack((mu_g_min_values, mu_g_max_values, mu_line_min_values, mu_line_max_values))))
        if (
            np.any(mu_g_min_values >= (M_pg_min - tolerance)) or
            np.any(mu_g_max_values >= (M_pg_min - tolerance)) or
            np.any(mu_line_min_values >= (M_pg_min - tolerance)) or
            np.any(mu_line_max_values >= (M_pg_min - tolerance))
        ):
            raise ValueError("Dual bounds are binding")

        # Validate complementary slackness, c) the complementary slackness conditions hold        
        max_compl[g] = np.max(np.abs(np.vstack(((np.hstack((pg_slack_min, pg_min)).reshape(-1,1) - pg_kkt_values) * mu_g_min_values, 
                                               (pg_kkt_values - np.hstack((pg_slack_max, pg_max)).reshape(-1,1)) * mu_g_max_values, 
                                               (np.hstack((pl_min,pl_min_tf)).reshape(-1,1) - (mpc.Bline @ theta_kkt_values).reshape(-1,1)) * mu_line_min_values, 
                                               ((mpc.Bline @ theta_kkt_values).reshape(-1,1) - np.hstack((pl_max,pl_max_tf)).reshape(-1,1)) * mu_line_max_values))))

        # Check the complementary slackness condition
        if max_compl[g] > 1e-4:
            raise ValueError('Complementary slackness conditions do not hold -- improve numerical accuracy')

        # Objective value and MILP gap
        v_dist_max[g] = milp.model.ObjVal
        milp_gap_v_dist_max[g] = milp.model.MIPGap

        # Compare with neural network prediction
        pd_nn_values = np.array([var.X for var in (power_system.pd_NN)])
        pg_pred_nn = nn_scalability.nn_prediction_fixed_ReLUs(pd_nn_values[:,0])
        pg_pred_values = np.array([var.X for var in (power_system.pg_pred)])

        if np.sum(np.abs(pg_pred_nn - pg_pred_values[:,0])) > 10e-4:
            #print(np.abs(pg_pred_nn - pg_pred_values[:,0]))
            print("Neural network prediction and MILP do not match -- abort!")
            # raise ValueError("Neural network prediction and MILP do not match")

    elif milp.model.Status != GRB.OPTIMAL:
        v_dist_max[g] = milp.model.objBound - 1e-4
        milp_gap_v_dist_max[g] = -1

        
    print(f"this is the max distance for generator {g}: ", v_dist_max[g])
    # print(milp.model.Status)


    v_dist_time[g] = milp.model.Runtime
    if milp.model.Status == GRB.OPTIMAL:
        # here we need to build the check which compares that for the identified system loading, the solution produced by the KKTs is
        # actually the optimal solution to the DC-OPF (we use rundcopf here; note that this is computationally much cheaper than solving the MILP
        # so we can do it for every one) Note that this check is for debugging purposes only
        
        # set the load
        mpc_test = mpc.mpc
        mpc_test.load['p_mw'] = pd_nn_values[:,0] * mpc.pd_delta + mpc.pd_min
        
        # solve the dc-opf
        pp.rundcopp(mpc_test)
        pg_dcopf = np.array(mpc_test.res_gen['p_mw'])/mpc.baseMVA
        
        # extract the active generator dispatch and compare, throw an error if they do not match
        if sum(np.abs(pg_kkt_values[1:, 0] - pg_dcopf)) > 10e-3:
            print('KKT solution and rundcopf do not match -- abort!')
            print(pg_kkt_values[1:, 0])
            print(pg_dcopf)
            #raise ValueError('KKT solution and rundcopf do not match')
        else:
            print('KKT solution and rundcopf do match -- continue')

# remove constraints and clean up model
for constr in ["distance_constr", "distance_abs_constr"] : 
    try:
        milp.model.remove(milp.model.getConstrByName(constr))
    except:
        pass


####################################################################################
for constr in ["distance_constr", "distance_abs_constr", "slack_generation"] : 
    try:
        milp.model.remove(milp.model.getConstrByName(constr))
    except:
        pass
        
# compute the maximum error on the objective function value
if slack_generation_constraint is None:  
    slack_generation_constraint = milp.model.addConstr(
        gp.quicksum(power_system.pd_NN[i, 0] * mpc.pd_delta[i] + mpc.pd_min[i] for i in range((power_system.pd_NN.shape[0]))) / mpc.baseMVA ==
        gp.quicksum(power_system.pg_slack + power_system.pg_pred[j, 0] * mpc.pg_delta[j] / mpc.baseMVA for j in range((power_system.pg_pred.shape[0]))),
        name="slack_generation"
    )

obj = gencost[:,0] @ ( gp.vstack((power_system.pg_slack * mpc.baseMVA, (power_system.pg_pred * mpc.pg_delta.reshape(-1, 1))))  - pg_kkt * mpc.baseMVA)[:,0]

milp.model.Params.TimeLimit = time_milp_max  
# milp.model.Params.BestBdStop = 100000       

# Solve the model
milp.model.setObjective(obj, GRB.MAXIMIZE)
milp.model.optimize()

# Error handling and re-optimization with different presolve options
if milp.model.Status not in [GRB.OPTIMAL, GRB.INFEASIBLE, GRB.UNBOUNDED]:
    milp.model.setParam("Presolve", 1)
    milp.model.optimize()
    if milp.model.Status not in [GRB.OPTIMAL, GRB.INFEASIBLE, GRB.UNBOUNDED]:
        milp.model.setParam("Presolve", 0)
        milp.model.optimize()
        if milp.model.Status not in [GRB.OPTIMAL, GRB.INFEASIBLE, GRB.UNBOUNDED]:
            raise RuntimeError("KKT MILP solve issue")


if milp.model.Status == GRB.OPTIMAL:
    print(obj.getValue())
    v_opt_max[0] = obj.getValue()
    milp_exact_v_opt_max = 1
    
    # Extract primal and dual variables
    pg_kkt_values = np.array([v.x for v in pg_kkt])
    theta_kkt_values = np.array([v.x for v in theta_kkt])
    
    # Validate primal bounds, we need to double check that the a) primal variables do not violate the derived bounds in the kkt
    tolerance = 10e-3
    if (
        np.any(pg_min - pg_kkt_values >= (M_pg_min - tolerance)) or
        np.any(pg_kkt_values - pg_max >= (M_pg_min - tolerance)) or
        np.any(pl_min - (mpc.Bline @ theta_kkt_values) >= (M_pg_min - tolerance)) or
        np.any((mpc.Bline @ theta_kkt_values) - pl_max >= (M_pg_min - tolerance))
    ):
        raise ValueError("Primal bounds are binding")

    # Validate dual bounds, b) dual variables do not violate the derived bounds in the kkt
    mu_g_min_values = np.array([v.x for v in mu_g_min])
    mu_g_max_values = np.array([v.x for v in mu_g_max])
    mu_line_min_values = np.array([v.x for v in mu_line_min])
    mu_line_max_values = np.array([v.x for v in mu_line_max])

    max_dual[g] = np.max(np.max(np.vstack((mu_g_min_values, mu_g_max_values, mu_line_min_values, mu_line_max_values))))
    if (
        np.any(mu_g_min_values >= (M_pg_min - tolerance)) or
        np.any(mu_g_max_values >= (M_pg_min - tolerance)) or
        np.any(mu_line_min_values >= (M_pg_min - tolerance)) or
        np.any(mu_line_max_values >= (M_pg_min - tolerance))
    ):
        raise ValueError("Dual bounds are binding")

    # Validate complementary slackness, c) the complementary slackness conditions hold        
    max_compl[g] = np.max(np.abs(np.vstack(((np.hstack((pg_slack_min, pg_min)).reshape(-1,1) - pg_kkt_values) * mu_g_min_values, 
                                            (pg_kkt_values - np.hstack((pg_slack_max, pg_max)).reshape(-1,1)) * mu_g_max_values, 
                                            (np.hstack((pl_min,pl_min_tf)).reshape(-1,1) - (mpc.Bline @ theta_kkt_values).reshape(-1,1)) * mu_line_min_values, 
                                            ((mpc.Bline @ theta_kkt_values).reshape(-1,1) - np.hstack((pl_max,pl_max_tf)).reshape(-1,1)) * mu_line_max_values))))

    # Check the complementary slackness condition
    if max_compl[g] > 1e-3:
        raise ValueError('Complementary slackness conditions do not hold -- improve numerical accuracy')

    
    # Compare with neural network prediction
    pd_nn_values = np.array([var.X for var in (power_system.pd_NN)])
    pg_pred_nn = nn_scalability.nn_prediction_fixed_ReLUs(pd_nn_values[:,0])
    pg_pred_values = np.array([var.X for var in (power_system.pg_pred)])

    if np.sum(np.abs(pg_pred_nn - pg_pred_values[:,0])) > 1e-3:
        raise ValueError("Neural network prediction and MILP do not match")
  
 
else:
    v_opt_max[0] = milp.model.objBound - 1e-4
    milp_exact_v_opt_max = 0


v_opt_time[0] = milp.model.Runtime
if milp.model.Status == GRB.OPTIMAL:
            
    # set the load
    mpc_test = mpc.mpc
    mpc_test.load['p_mw'] = pd_nn_values[:,0] * mpc.pd_delta + mpc.pd_min
    
    # print(mpc_test.load['p_mw'])
    
    # solve the dc-opf
    pp.rundcopp(mpc_test)
    pg_dcopf = np.array(mpc_test.res_gen['p_mw'])/mpc.baseMVA
    
    # extract the active generator dispatch and compare, throw an error if they do not match
    if sum(np.abs(pg_kkt_values[1:, 0] - pg_dcopf)) > 10e-4:
        print('KKT solution and rundcopf do not match -- abort!')
        #raise ValueError('KKT solution and rundcopf do not match')
    else:
        print('KKT solution and rundcopf do match -- continue')




v_info = {}
v_info["v_dist_time"] = max(v_dist_time)
v_info["v_dist_wc"] = max(v_dist_max)
v_info["v_dist_ID"] = np.argmax(v_dist_max)

v_info["milp_gap_v_dist_max"] = max(milp_gap_v_dist_max)
v_info["milp_exact_v_dist_max"] = max(milp_exact_v_dist_max)

v_info["max_compl"] = max(max_compl)
v_info["max_dual"] = max(max_dual)

v_info["v_opt_max"] = max(v_opt_max)
v_info["v_opt_time"] = max(v_opt_time)

print(v_info)


# % create reference cost
# mpc_ref = mpc;
# mpopt = mpoption;
# mpopt.out.all =0;
# results_dcopf=rundcopf(mpc_ref,mpopt);

# v_info.v_opt_wc = v_opt_max./results_dcopf.f;


