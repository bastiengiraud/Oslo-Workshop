def fpr_score(y_actual, y_hat, index=None):
    """
    Function to determine the FP score, and to get the indices of the OPs
    where there is missclassification.

    y_actual = pandas series, comes from train-test split
    y_hat = np.array, comes from prediction

    """
    
    TP = FP = TN = FN = 0
    FP_list = []
    FN_list = []

    for i in range(len(y_actual)):
        # For multi-class predictions, use index to access the correct class label
        if index is not None:
            actual = y_actual.iloc[i, index]  # If y_actual is a DataFrame, use iloc to access the column
            predicted = y_hat[i, index]  # Access the index in y_hat if it's 2D
        else:
            actual = y_actual.iloc[i]  # For 1D series, access directly
            predicted = y_hat[i]  # For 1D arrays, access directly


        if actual == predicted == 1:
            TP += 1
        elif predicted == 1 and actual != predicted:
            FP += 1
            FP_list.append(i)
        elif actual == predicted == 0:
            TN += 1
        elif predicted == 0 and actual != predicted:
            FN += 1
            FN_list.append(i)

    FPR = FP / (FP + TN) if (FP + TN) > 0 else -1
    FNR = FN / (TP + FN) if (TP + FN) > 0 else -1

    return FPR, FNR, FP_list, FN_list


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os

class StabilityDataset:
    def __init__(self, directory, case_size="39", method_type="SD13", total_samples=10000, stability_boundary=0.03, lower_bound=0.0275, upper_bound=0.0325):
        self.case_size = case_size
        self.method_type = method_type
        self.total_samples = total_samples
        self.stability_boundary = stability_boundary
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        self.directory = directory
        self.flow_name_method = f"{case_size}bus_{method_type}_method_flows.csv"
        self.file_name_method = f"{case_size}bus_{method_type}_method_ops.csv"

        self.flow_path_method = os.path.join(self.directory, self.flow_name_method)
        self.file_path_method = os.path.join(self.directory, self.file_name_method)

        self.load_data()
        self.process_data()
        self.split_data()
        self.compute_statistics()
        self.print_statistics()

    def load_data(self):
        self.op_data_method = pd.read_csv(self.file_path_method, sep=';')
        self.op_data_method = self.op_data_method.sample(n=self.total_samples, random_state=42).reset_index(drop=False)
        self.op_data_method_unmod = self.op_data_method
        self.damping_data_method = self.op_data_method['damping']
        
        self.data_method = pd.read_csv(self.flow_path_method, sep=';')
        self.data_method = self.data_method.iloc[self.op_data_method['index']]
    
    def process_data(self):
        self.X_method = self.data_method.drop(columns=['feasible', 'stable'], axis=1)
        feasible_array = self.data_method['feasible'].to_numpy()
        stable_array = self.data_method['stable'].to_numpy()
        secure_array = ((self.data_method['stable'] == 1) & (self.data_method['feasible'] == 1)).astype(int)
        #self.y_method = np.column_stack((feasible_array, stable_array, secure_array))
        self.y_method = secure_array
    
    def split_data(self):
        self.X_train, self.X_test, self.y_train, self.y_test, self.damping_data_method_train, self.damping_data_method_test, self.op_data_method_train, self.op_data_method_test = train_test_split(
            self.X_method, self.y_method, self.damping_data_method, self.op_data_method, test_size=0.25, random_state=42
        )
    
    def compute_statistics(self):
        # self.method_secure = (self.y_method[:, 2] == 1).sum()
        # self.method_insecure = (self.y_method[:, 2] == 0).sum()
        self.method_secure = (self.y_method[:] == 1).sum()
        self.method_insecure = (self.y_method[:] == 0).sum()
        self.method_stable = (self.op_data_method['damping'] >= self.stability_boundary).sum()
        self.method_instable = (self.op_data_method['damping'] <= self.stability_boundary).sum()
        self.method_hic = ((self.op_data_method['damping'] >= self.lower_bound) & (self.op_data_method['damping'] <= self.upper_bound)).sum()
        self.method_feasible = (self.op_data_method['N0'] == 1).sum()
        self.method_infeasible = (self.op_data_method['N0'] == 0).sum()
    
    def print_statistics(self):
        print(f"Method Secure: {self.method_secure}")
        print(f"Method Insecure: {self.method_insecure}")
        print(f"Method Stable: {self.method_stable}")
        print(f"Method Instable: {self.method_instable}")
        print(f"Method HIC: {self.method_hic}")
        print(f"Method Feasible: {self.method_feasible}")
        print(f"Method Infeasible: {self.method_infeasible}")



