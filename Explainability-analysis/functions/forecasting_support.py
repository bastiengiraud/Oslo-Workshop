import numpy as np
import pandas as pd
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim


class TimeSeriesDataPreparer:
    def __init__(self, df, target, exog=None, look_back=48, look_ahead=12):
        self.df = df.copy()
        self.target = target if isinstance(target, list) else [target]
        self.exog = exog if exog else []
        self.look_back = look_back
        self.look_ahead = look_ahead
        self.scalers = {}
        self.data_scaled = None
        self.actual_power = None  # Store actual PV values
    
    def scale_data(self):
        columns = list(dict.fromkeys(["HourUTC"] + self.target + self.exog))
        self.df = self.df[columns]
        
        if self.exog:
            sc_exog = MinMaxScaler(feature_range=(0, 1))
            sc_target = MinMaxScaler(feature_range=(0, 1))
            
            self.scalers['exog'] = sc_exog
            self.scalers['target'] = sc_target
            
            data_exog_scaled = sc_exog.fit_transform(self.df[self.exog])
            data_target_scaled = sc_target.fit_transform(self.df[self.target])
            self.data_scaled = np.hstack((data_target_scaled, data_exog_scaled[:, 1:]))
        else:
            sc = MinMaxScaler(feature_range=(0, 1))
            self.scalers['target'] = sc
            self.data_scaled = sc.fit_transform(self.df[self.target].values.reshape(-1, 1))
    
    def shift_exog(self):
        if self.exog:
            cols_shift = self.df.columns[1:]
            self.df.loc[:, cols_shift] = self.df.loc[:, cols_shift].astype(float).shift(24, axis=0).fillna(0)
    
    def get_indices(self, t_s, t_e, test_ts, test_te):
        index_train_t_s = self.df[self.df['HourUTC'] == t_s].index[0]
        index_train_t_e = self.df[self.df['HourUTC'] == t_e].index[0]
        index_test_t_s = self.df[self.df['HourUTC'] == test_ts].index[0]
        index_test_t_e = self.df[self.df['HourUTC'] == test_te].index[0]
        
        total_test_samples = (index_test_t_e - index_test_t_s) - self.look_back
        remainder = total_test_samples % self.look_ahead
        if remainder != 0:
            index_test_t_e -= remainder
        
        return index_train_t_s, index_train_t_e, index_test_t_s, index_test_t_e
    
    def create_datasets(self, index_train_t_s, index_train_t_e, index_test_t_s, index_test_t_e):
        train_data_scaled = self.data_scaled[(index_train_t_s - self.look_back):(index_train_t_e + self.look_ahead)]
        test_data_scaled = self.data_scaled[(index_test_t_s - self.look_back):(index_test_t_e + self.look_ahead)]
        
        X_train, y_train = [], []
        for i in range(self.look_back, len(train_data_scaled) - self.look_ahead):
            X_train.append(train_data_scaled[i - self.look_back:i, :])
            y_train.append(train_data_scaled[i:i + self.look_ahead, 0])
        
        X_test, y_test = [], []
        for i in range(self.look_back, len(test_data_scaled), self.look_ahead):
            X_test.append(test_data_scaled[i - self.look_back:i, :])
            y_test.append(test_data_scaled[i:i + self.look_ahead, 0])
        
        X_train, y_train = np.array(X_train), np.array(y_train)
        X_test, y_test = np.array(X_test), np.array(y_test)
        
        return X_train, y_train, X_test, y_test
    
    def get_scaler(self):
        """Returns the scaler for inverse transformation"""
        return self.scalers.get('target', None)

    def get_actual_pv(self, index_test_t_s, index_test_t_e):
        """Retrieves and stores actual PV values (unscaled) for comparison"""
        self.actual_power = self.df[self.target].iloc[index_test_t_s:index_test_t_e + self.look_ahead].reset_index(drop=True)

    def plot_predictions(self, y_pred, y_test):
        """Plots actual vs predicted PV power after inverse transformation."""
        if self.actual_power is None:
            raise ValueError("Actual PV data is not set. Use get_actual_pv().")

        scaler = self.get_scaler()
        if scaler:
            y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
            y_test = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        
        plt.figure(figsize=(10, 5))
        plt.plot(self.actual_power, label='Actual PV', linestyle='-', color='blue')
        plt.plot(y_pred, label='Predicted PV', linestyle='--', color='red')

        plt.xlabel('Hour')
        plt.ylabel('PV Power (kW)')
        plt.legend()
        plt.title('Actual vs Predicted PV Power')
        plt.grid(True)
        plt.show()
    
    def visualize(self, train_timestamps, test_timestamps, train_data_scaled, test_data_scaled):
        plt.figure(figsize=(10, 5), dpi=100)
        plt.plot(train_timestamps, train_data_scaled, label="Training set", color="blue")
        plt.plot(test_timestamps, test_data_scaled, label="Testing set", color="red")
        plt.legend()
        plt.xlabel("Time")
        plt.ylabel("Normalized PV Power")
        plt.title("Train-Test Split Visualization")
        plt.xticks(rotation=45)
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.show()
    
    def prepare_data(self, t_s, t_e, test_ts, test_te, visualize=False):
        self.scale_data()
        self.shift_exog()
        
        index_train_t_s, index_train_t_e, index_test_t_s, index_test_t_e = self.get_indices(t_s, t_e, test_ts, test_te)
        X_train, y_train, X_test, y_test = self.create_datasets(index_train_t_s, index_train_t_e, index_test_t_s, index_test_t_e)
        
        self.get_actual_pv(index_test_t_s, index_test_t_e)  # Store actual PV
        
        if visualize:
            train_timestamps = self.df['HourUTC'].iloc[index_train_t_s - self.look_back : index_train_t_e + self.look_ahead]
            test_timestamps = self.df['HourUTC'].iloc[index_test_t_s - self.look_back : index_test_t_e + self.look_ahead]
            self.visualize(train_timestamps, test_timestamps, self.data_scaled[index_train_t_s - self.look_back : index_train_t_e + self.look_ahead], self.data_scaled[index_test_t_s - self.look_back : index_test_t_e + self.look_ahead])
        
        return X_train, y_train, X_test, y_test

class LSTMTimeSeries(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, output_size=1, num_layers=2, dropout=0.2):
        super(LSTMTimeSeries, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out

class LSTMTrainer:
    def __init__(self, model, learning_rate=0.001, device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.train_losses = []
    
    def train(self, X_train, y_train, num_epochs=50, batch_size=32):
        X_train, y_train = torch.tensor(X_train, dtype=torch.float32).to(self.device), \
                           torch.tensor(y_train, dtype=torch.float32).to(self.device)
        dataset = torch.utils.data.TensorDataset(X_train, y_train)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        for epoch in range(num_epochs):
            self.model.train()
            epoch_loss = 0
            
            for batch_X, batch_y in dataloader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                if len(batch_y.shape) == 1:
                    batch_y = batch_y.unsqueeze(-1)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(dataloader)
            self.train_losses.append(avg_loss)
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    def evaluate(self, X_test):
        X_test = torch.tensor(X_test, dtype=torch.float32).to(self.device)
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_test).cpu().numpy()
        return predictions
    
    def plot_losses(self):
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(self.train_losses) + 1), self.train_losses, marker='o', linestyle='-', color='b')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss Over Epochs")
        plt.grid(True)
        plt.show()


