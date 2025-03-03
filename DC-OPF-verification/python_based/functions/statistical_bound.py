"""
based on E1_Evaluate_Average_And_Worst_Performance_Data.m

Evaluate the average performance on the testing dataset and the worst-case performance on the entire dataset.
The latter serves as empirical lower bound to the exact worst-case performance over the entire input domain.

"""

import numpy as np
import pandapower as pp
import pandapower.converter as pc
import pandas as pd
from scipy.linalg import inv
from pypower.makeBdc import makeBdc
from pypower.makePTDF import makePTDF
import scipy.io
import logging
import os

    
class PrepareDCOPFData:
    def __init__(self, case_name, case_path):
        self.case_name = case_name
        #self.iteration = iteration
        self.case_path = case_path
        self.net = self.load_case_file()  # Load the case data
        self.load_case_data()
        
    def load_case_file(self):
        """Load MATPOWER case file into a Pandapower network."""
        case_file = os.path.join(self.case_path, f"{self.case_name}.m")
        
        # Check if the file exists
        if not os.path.isfile(case_file):
            raise FileNotFoundError(f"Error: The file {case_file} does not exist.")
        
        # Load the MATPOWER case into a Pandapower network
        net = pc.from_mpc(case_file)
        return net
        
    def load_case_data(self):
        """Load system model and parameters for the given case."""
        # Access the Pandapower network (this is equivalent to self.mpc in the original code)
        self.mpc = self.net
        
        # Access the number of buses, generators, and branches
        self.nb = len(self.mpc.bus)  # Number of buses
        self.ng = len(self.mpc.gen) # Number of generators
        self.nl = len(self.mpc.line)  # Number of branches (lines)
        self.nl_tf = len(self.mpc.trafo)
        self.baseMVA = np.array(self.mpc.gen["sn_mva"])[0]
        
        # Example load-to-bus mapping
        self.id_loads = self.mpc.load.loc[self.mpc.load['p_mw'] != 0, 'bus'].tolist() # Find buses with loads
        self.nloads = len(self.id_loads)
        
        # Find buses with generators
        self.id_gens = self.mpc.gen.loc[self.mpc.gen['p_mw'] != 0, 'bus'].tolist()  # GEN_BUS is column 0 in gen matrix
        self.slack = np.array(self.mpc.ext_grid["bus"])[0]
        self.gen_slack = np.array((self.mpc.gen["bus"][self.mpc.poly_cost["et"] == "ext_grid"]).index)[0]
        self.id_gens.append(self.slack)
        self.id_gens.sort()

        # Mapping from loads to buses
        self.M_d = np.zeros((self.nb, self.nloads))
        for i in range(self.nloads):
            self.M_d[self.id_loads[i], i] = 1

        # Mapping from generators to buses
        self.M_g = np.zeros((self.nb, len(self.id_gens)))
        for i in range(len(self.id_gens)):
            self.M_g[self.id_gens[i], i] = 1

        # Extracting pd_max, pd_min, pd_delta
        self.pd_max = np.array(self.mpc.load.loc[self.mpc.load['bus'].isin(self.id_loads), 'p_mw'])
        self.pd_min = self.pd_max * 0.6
        self.pd_delta = self.pd_max * 0.4

        # Handling generator data
        self.pg_min = np.array(self.mpc.gen['min_p_mw'])
        self.pg_max = np.array(self.mpc.gen['max_p_mw'])
        self.pg_delta = self.pg_max - self.pg_min
        
        # handling external grid data
        self.pg_slack_min = np.array(self.mpc.ext_grid['min_p_mw'])
        self.pg_slack_max = np.array(self.mpc.ext_grid['max_p_mw'])
        self.pg_slack_delta = self.pg_slack_max - self.pg_slack_min
        
        # line limits
        self.pl_max = self.mpc.line["max_loading_percent"]
        rated_voltage = self.mpc.bus.loc[self.mpc.line["from_bus"], "vn_kv"].values
        rated_current = np.array(self.mpc.line["max_i_ka"])
        self.rate_a = np.sqrt(3) * rated_voltage * rated_current
        
        # trafo limits
        self.rate_a_tf = np.array(self.mpc.trafo["sn_mva"])
        
        #filtered_poly_cost = self.mpc.poly_cost[self.mpc.poly_cost["et"] == "gen"]
        self.gencost = np.array(self.mpc.poly_cost["cp1_eur_per_mw"]) #np.array(filtered_poly_cost["cp1_eur_per_mw"])
        
        # Compute admittance matrices
        pypower_net = pc.to_ppc(self.mpc, init='flat') # convert to pypower net to use pypower makeBdc function
        B, Bf, Pbusinj, Pfinj = makeBdc(pypower_net['baseMVA'], pypower_net['bus'], pypower_net['branch']) # exclude transformers
        self.B = B
        self.Bline = Bf
        self.Pbusinj = Pbusinj
        self.Pfinj = Pfinj
        
        # reduced Bbus
        ptdf = makePTDF(pypower_net['baseMVA'], pypower_net['bus'], pypower_net['branch'], slack=self.slack)
        self.ptdf = np.delete(ptdf, self.slack, axis=1)
        
        print("Loaded case data successfully.")


class DataLoader:
    def __init__(self, nn_path, dataset_type):
        self.nn_path = nn_path
        
        self.load_data(dataset_type)
        self.load_neural_network(3) # give the number of NN layers

    def load_data(self, dataset_type):
        """
        Load dataset based on type ('test', 'train', or 'all').
        """
        if dataset_type == "test":
            input_file = f"{self.nn_path}/features_test.csv"
            output_file = f"{self.nn_path}/labels_test.csv"
        elif dataset_type == "train":
            input_file = f"{self.nn_path}/features_train.csv"
            output_file = f"{self.nn_path}/labels_train.csv"
        elif dataset_type == "all":
            input_file = f"{self.nn_path}/NN_input.csv"
            output_file = f"{self.nn_path}/NN_output.csv"
        else:
            raise ValueError("Invalid dataset type. Choose from 'test', 'train', or 'all'.")
        
        # Load the data
        self.input_nn = pd.read_csv(input_file, header=None).to_numpy()
        self.output_nn = pd.read_csv(output_file, header=None).to_numpy()
        
        return self.input_nn, self.output_nn    

    def load_csv(self, file_path):
        """Helper method to load CSV data and return as a NumPy array."""
        return np.genfromtxt(file_path, delimiter=',')
    
    def load_neural_network(self, layers):
        """Load NN weights, biases, and input-output data."""
        path = self.nn_path
        
        # Load weights
        self.weights = {
            "input": self.load_csv(f"{path}/W0.csv").T,  
            "output": self.load_csv(f"{path}/W3.csv").T,  
            "hidden": [self.load_csv(f"{path}/W{i}.csv").T for i in range(1, layers)]  
        }
        
        # Load biases
        self.biases = [self.load_csv(f"{path}/b{i}.csv") for i in range(4)]
        
        print("Neural network data loaded successfully.")
        
        self.w_input = self.weights["input"]  
        self.w_output = self.weights["output"]
        self.w_hidden = self.weights["hidden"]  
        
        # return weights, biases
 


class ReluStability:
    def __init__(self, nn_path, layers, data_loader):
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
        
        self.ReLU_stability = None
        self.relu_stability()
    
    def relu_stability(self):
        """Checks ReLU stability (always on/off) for given neural network weights."""
        
        # Initialize ReLU stability array
        ReLU_stability = np.ones((self.nb_samples, self.ReLU_layers, self.nb_neurons))
        tol_ReLU = 0.0
        
        # Loop over all samples to compute ReLU activations
        for i in range(self.nb_samples):
            pd_nn = self.input_nn[i, :]
            zk_hat = np.dot(self.w_input, pd_nn.T) + self.biases[0] # compute output first layer
            zk = np.maximum(zk_hat, 0) # apply relu to output
            ReLU_stability[i, 0, :] = zk > tol_ReLU  # First layer ReLU
            
            for j in range(self.ReLU_layers - 1):
                zk_hat = np.dot(self.w_hidden[j], zk) + self.biases[j + 1]
                zk = np.maximum(zk_hat, 0)
                ReLU_stability[i, j + 1, :] = zk > tol_ReLU  # Subsequent layers ReLU

        # Calculate active and inactive ReLUs
        self.ReLU_always_active = np.sum(ReLU_stability, axis=0) == self.nb_samples
        self.ReLU_always_inactive = np.sum(ReLU_stability, axis=0) == 0
        
        # Report percentage of always active and inactive ReLUs
        active_percentage = np.sum(self.ReLU_always_active) / (self.ReLU_layers * self.nb_neurons) * 100
        inactive_percentage = np.sum(self.ReLU_always_inactive) / (self.ReLU_layers * self.nb_neurons) * 100
        
        scipy.io.savemat(os.path.join(self.nn_path, 'ReLU_always_inactive.mat'), {'ReLU_always_inactive': self.ReLU_always_inactive})
        scipy.io.savemat(os.path.join(self.nn_path, 'ReLU_always_active.mat'), {'ReLU_always_active': self.ReLU_always_active})
        
        # print(f"The share of always active ReLUs: {active_percentage:.2f} %")
        # print(f"The share of always inactive ReLUs: {inactive_percentage:.2f} %")
    



class NeuralNetworkPrediction:
    def __init__(self, data_loader, ReLU_always_active, ReLU_always_inactive):
        self.w_input = data_loader.w_input
        self.w_output = data_loader.w_output
        self.w_hidden = data_loader.w_hidden
        self.biases = data_loader.biases
        self.ReLU_layers = len(data_loader.w_hidden) + 1
        self.ReLU_always_active = ReLU_always_active.astype(bool)
        self.ReLU_always_inactive = ReLU_always_inactive.astype(bool)
        
    
    def nn_prediction_all_ReLUs(self, pd_nn):
        """
        Compute the neural network prediction.
        """
        # Compute the output of the first layer
        zk_hat = np.dot(self.w_input, pd_nn) + self.biases[0]
        zk = np.maximum(zk_hat, 0)  # Apply ReLU activation

        # Propagate through hidden layers
        for j in range(self.ReLU_layers - 1):
            zk_hat = np.dot(self.w_hidden[j], zk) + self.biases[j + 1]
            zk = np.maximum(zk_hat, 0)  # Apply ReLU activation

        # Compute the final output
        pg_pred = np.dot(self.w_output, zk) + self.biases[-1]
        return pg_pred
    
    def nn_prediction_fixed_ReLUs(self, pd_nn):
        """
        Compute the neural network prediction considering ReLU stability.
        """
        # Compute the output of the first layer
        zk_hat = np.dot(self.w_input, pd_nn) + self.biases[0]
        zk = np.maximum(zk_hat, 0)  # Apply ReLU activation

        # Apply ReLU stability constraints to the first layer
        zk[self.ReLU_always_active[0, :]] = zk_hat[self.ReLU_always_active[0, :]]
        zk[self.ReLU_always_inactive[0, :]] = 0

        # Propagate through hidden layers
        for j in range(self.ReLU_layers - 1):
            zk_hat = np.dot(self.w_hidden[j], zk) + self.biases[j + 1]
            zk = np.maximum(zk_hat, 0)  # Apply ReLU activation

            # Apply ReLU stability constraints
            zk[self.ReLU_always_active[j + 1, :]] = zk_hat[self.ReLU_always_active[j + 1, :]]
            zk[self.ReLU_always_inactive[j + 1, :]] = 0

        # Compute the final output
        pg_pred = np.dot(self.w_output, zk) + self.biases[-1]
        return pg_pred
     


    
class DCOPFEmpircalPerformance:
    def __init__(self, net_data, nn_relu_stability, nn_prediction):
        self.net_data = net_data
        self.nn_relu_stability = nn_relu_stability
        self.nn_prediction = nn_prediction

    def compute_violations(self):
        nb_samples = self.nn_relu_stability.input_nn.shape[0]
        output_pred_ReLU = np.zeros((nb_samples, self.nn_relu_stability.output_nn.shape[1]))

        for j in range(nb_samples):
            pd_nn = self.nn_relu_stability.input_nn[j, :]

            # Predict output with ReLU stability
            output_pred_ReLU[j, :] = self.nn_prediction.nn_prediction_fixed_ReLUs(pd_nn)
            
            # Predict output without ReLU stability (for validation)
            output_pred = self.nn_prediction.nn_prediction_all_ReLUs(pd_nn)

            # here we check that the ReLU stability does not impact the neural network prediction on the entire dataset
            if np.max(np.abs(output_pred_ReLU[j, :] - output_pred)) > 1e-4:
                raise ValueError('ReLU approximation not correct!')

        # Compute errors
        mae = np.mean(np.abs(output_pred_ReLU - self.nn_relu_stability.output_nn))
        v_dist = np.mean(np.max(np.abs(output_pred_ReLU - self.nn_relu_stability.output_nn), axis=1))
        v_dist_wc = np.max(np.max(np.abs(output_pred_ReLU - self.nn_relu_stability.output_nn), axis=1))

        cost_pred = np.zeros(nb_samples)
        cost_true = np.zeros(nb_samples)

        # Run DC OPF
        mpc_optimal = self.net_data.mpc
        pp.rundcopp(mpc_optimal)
        cost_ref = np.array(mpc_optimal.res_cost)

        for j in range(nb_samples):
            pd_sum = self.nn_relu_stability.input_nn[j, :] * self.net_data.pd_delta + self.net_data.pd_min
            pg_pred = output_pred_ReLU[j, :] * self.net_data.pg_delta
            pg_true = self.nn_relu_stability.output_nn[j, :] * self.net_data.pg_delta
            
            pg_slack_pred = sum(pd_sum) - sum(pg_pred)
            pg_slack_true = sum(pd_sum) - sum(pg_true)

            cost_pred[j] = np.sum(np.concatenate((np.array([pg_slack_pred]), pg_pred)) * self.net_data.gencost)
            cost_true[j] = np.sum(np.concatenate((np.array([pg_slack_true]), pg_true)) * self.net_data.gencost)

        v_opt = np.mean((cost_pred - cost_true) / cost_ref)
        v_opt_wc = np.max((cost_pred - cost_true) / cost_ref)

        # Initialize violations
        pg_up_viol = np.zeros((nb_samples, self.net_data.ng + 1))
        pg_down_viol = np.zeros((nb_samples, self.net_data.ng + 1))
        pline_viol = np.zeros((nb_samples, self.net_data.nl + self.net_data.nl_tf))

        # Extract maximum and minimum limits
        pline_l_max = self.net_data.rate_a / self.net_data.baseMVA
        pline_tf_max = self.net_data.rate_a_tf / self.net_data.baseMVA
        pline_max = np.hstack((pline_l_max, pline_tf_max))
        
        pg_min = self.net_data.pg_min / self.net_data.baseMVA
        pg_max = self.net_data.pg_max / self.net_data.baseMVA
        pg_slack_min = self.net_data.pg_slack_min / self.net_data.baseMVA
        pg_slack_max = self.net_data.pg_slack_max / self.net_data.baseMVA

        # Update load values
        self.net_data.mpc.load['p_mw'] = self.net_data.pd_min + (self.nn_relu_stability.input_nn[j, :] * self.net_data.pd_delta)

        # Compute admittance matrices
        pypower_net = pc.to_ppc(mpc_optimal) # convert to pypower net to use pypower makeBdc function
        B, Bf, Pbusinj, Pfinj = makeBdc(pypower_net['baseMVA'], pypower_net['bus'], pypower_net['branch'])
        
        # Ensure no additional injections
        if np.sum(np.abs(Pbusinj)) + np.sum(np.abs(Pfinj)) > 1e-4:
            raise ValueError("Code cannot handle additional injections")

        # Compute reduced bus admittance matrix (removing slack bus)
        slack_bus = self.net_data.slack  # First generator bus as slack
        B = B.todense() # make csc-matrix to dense
        B_red = np.delete(np.delete(B, slack_bus, axis=0), slack_bus, axis=1)
        B_inv = inv(B_red)

        # Store original mpc
        mpc_original = self.net_data.mpc

        for j in range(nb_samples):
            mpc_copy = mpc_original

            # Compute slack bus dispatch (if needed)
            pg_slack = (sum(self.net_data.pd_min) + np.dot(self.nn_relu_stability.input_nn[j, :], self.net_data.pd_delta) - np.dot(self.nn_relu_stability.output_nn[j, :], self.net_data.pg_delta)) / self.net_data.baseMVA
            
            # Compute power injections excluding slack bus
            pg = np.concatenate((np.array([pg_slack]), (output_pred_ReLU[j, :] * self.net_data.pg_delta / self.net_data.baseMVA)))
            pd = (self.net_data.pd_min / self.net_data.baseMVA + self.nn_relu_stability.input_nn[j, :] * self.net_data.pd_delta / self.net_data.baseMVA)
            pinj = self.net_data.M_g @ pg.reshape(-1, 1) - self.net_data.M_d @ pd.reshape(-1, 1)
           
            # remove slack bus injection
            pinj_woslack = np.delete(pinj, slack_bus, axis=0)

            # Compute line flow using DC approximation
            theta_woslack = (B_inv @ pinj_woslack)[:,0]
            
            # insert zero at slack bus position
            theta = np.zeros(self.net_data.nb)
            theta[:slack_bus] = theta_woslack[:slack_bus]
            theta[slack_bus + 1:] = theta_woslack[slack_bus:]
            
            ############### get indices of lines and trafos
            pline = (Bf @ theta) # [0:self.net_data.nl] # take only line flows. Bf also includes trafos 

            # Compute line flow violations
            pline_viol[j, :] = np.maximum(np.maximum(pline - pline_max, 0) , np.maximum(-pline_max - pline, 0))
            
            # compute generator violations. 
            pg_cur = np.concatenate((np.array([pg_slack]), (output_pred_ReLU[j, :] * self.net_data.pg_delta / self.net_data.baseMVA)))
            pg_up_viol[j, :] = np.maximum(pg_cur - np.hstack((pg_slack_max, pg_max)), 0)
            pg_down_viol[j, :] = np.maximum(np.hstack((pg_slack_min, pg_min)) - pg_cur, 0)

            # Debugging comparison with DC power flow results, compare on expectation every 50st sample
            if np.random.rand(1) >= 0.98:
                mpc_copy.gen["p_mw"] = mpc_copy.gen["min_p_mw"] + output_pred_ReLU[j, :] * self.net_data.pg_delta
                mpc_copy.load["p_mw"] = self.net_data.pd_min + self.nn_relu_stability.input_nn[j, :] * self.net_data.pd_delta
                
                # run a dc pf
                pp.rundcpp(mpc_copy)
                pline_l_dcpf = np.array(mpc_copy.res_line["p_from_mw"]) / self.net_data.baseMVA # get the line flows
                pline_tf_dcpf = np.array(mpc_copy.res_trafo["p_hv_mw"]) / self.net_data.baseMVA # get the flows over the trafos
                pline_dcpf = np.hstack((pline_l_dcpf, pline_tf_dcpf)) # combine lines and trafos 
                pg_dcpf = np.hstack((np.array(mpc_copy.res_ext_grid["p_mw"]), np.array(mpc_copy.res_gen["p_mw"]))) / self.net_data.baseMVA
                
                if np.max(np.abs(pline_dcpf - pline)) + np.max(np.abs(pg_dcpf - pg_cur.flatten())) > 10:
                    raise ValueError("Mismatch in constraint violation computation")

        v_line = np.mean(np.max(pline_viol, axis=1))
        v_g = np.mean(np.max(np.hstack((pg_up_viol, pg_down_viol)), axis=1))
        v_line_wc = np.max(np.max(pline_viol, axis=1))
        v_g_wc = np.max(np.max(np.hstack((pg_up_viol, pg_down_viol)), axis=1))

        return mae, v_dist, v_dist_wc, v_opt, v_opt_wc, v_line, v_g, v_line_wc, v_g_wc


def run_dc_opf_evaluation(case_name, case_path, nn_path, data_split='all'):
    """
    Runs the empirical DC OPF evaluation, computes violations, and returns a summary of results.
    """
    logging.info("Initializing dataset and neural network models...")

    # Load network data and neural network
    net_data = PrepareDCOPFData(case_name, case_path)
    data_loader = DataLoader(nn_path, data_split)
    nn_relu_stability = ReluStability(nn_path, 3, data_loader)
    nn_prediction = NeuralNetworkPrediction(data_loader, 
                                            nn_relu_stability.ReLU_always_active, nn_relu_stability.ReLU_always_inactive)

    # Initialize result storage
    results = np.zeros((9, 1))  

    dnn = DCOPFEmpircalPerformance(net_data, nn_relu_stability, nn_prediction)
    
    mae, v_dist, v_dist_wc, v_opt, v_opt_wc, v_line, v_g, v_line_wc, v_g_wc = dnn.compute_violations()
    results[:, 0] = [mae, v_dist, v_dist_wc, v_opt, v_opt_wc, v_line, v_g, v_line_wc, v_g_wc]

    # Compute aggregated statistics
    summary_results = {
        "MAE (%)": np.mean(results[0, :]) * 100,
        "Avg Max Generator Violation (MW)": np.mean(results[6, :]) * 100,
        "Avg Max Line Violation (MW)": np.mean(results[5, :]) * 100,
        "Avg Distance to Optimal Setpoints (%)": np.mean(results[1, :]) * 100,
        "Avg Sub-Optimality (%)": np.mean(results[3, :]) * 100,
        "Worst-Case Generator Violation (MW)": np.mean(results[8, :]) * 100,
        "Worst-Case Line Violation (MW)": np.mean(results[7, :]) * 100,
        "Worst-Case Distance to Optimal Setpoints (%)": np.mean(results[2, :]) * 100,
        "Worst-Case Sub-Optimality (%)": np.mean(results[4, :]) * 100
    }

    # Print the results
    print("\nSummary Results:")
    print("-" * 60)
    for key, value in summary_results.items():
        print(f"{key:<45}: {value:>10.2f}")


    return summary_results




if __name__ == "__main__":
    
    # Get the current working directory
    parent_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

    # Define the cases
    case_name = 'case39_DCOPF'
    case_iter = '1'
    case_path = os.path.join(parent_directory, "python_based", "test_networks", "network_modified")

    # define the neural network
    nn_path = os.path.join(parent_directory, "python_based", "trained_nns", case_name, case_iter)

    results_summary = run_dc_opf_evaluation(case_name, case_path, nn_path)

