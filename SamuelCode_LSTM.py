import os
import pandas as pd
import numpy as np
import torch.optim
import torch.nn as nn
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import lightning as L
from torch.utils.data import Dataset
import random
import sys
import wandb
import time
import socket
from pytorch_lightning.loggers import WandbLogger

# # Set random seed for reproducibility
# random_seed = 42 # Also remember to change in get_train_val_pairs !!!
# random.seed(random_seed)
# torch.manual_seed(random_seed)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed(random_seed)
#     torch.cuda.manual_seed_all(random_seed)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

feature_names = ["hr", "hrvDifference", "hrvPoincare", "hrvSpectral",
     "scgPEP", "scgLVET", "scgPEPOverLVET", "scgPAT", "scgPTT",
     "ppgIHAT", "femoralPPGPPV", "ppgAmplitudeDistal"]
rel_events = ["baseline_1", "vaso_1", "vaso_2", "vaso_3", "baseline_2"]
abs_events = ["baseline_abs", "abs_0_7", "abs_7_14", "abs_14_21", "abs_21_28"]
res_events = ["ref_28_21", "ref_21_14", "ref_14_7", "ref_7_0"]

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
classes_to_bvds_dict = {
        0: 0,
        1: 25,
        2: 33,
        3: 50,
        4: 67,
        5: 75,
        6: 100
}

class MyProgressBar(L.pytorch.callbacks.TQDMProgressBar):
    # This class is only here to fix a callback problem of lightning.
    # You do not need to understand this.
    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        if not sys.stdout.isatty():
            bar.disable = True
        return bar

    def init_predict_tqdm(self):
        bar = super().init_predict_tqdm()
        if not sys.stdout.isatty():
            bar.disable = True
        return bar

    def init_test_tqdm(self):
        bar = super().init_test_tqdm()
        if not sys.stdout.isatty():
            bar.disable = True
        return bar

class TimeSeriesDataset(Dataset):
    # Creates timeseries dataset from designated x and y data
    def __init__(self, features, labels, hypo_stages):
        self.features = features
        self.labels = labels
        self.hypo_stages = hypo_stages
    def __len__(self):
        return len(self.features)
    def __getitem__(self, idx): #add arg for forwards/backwards?
        return torch.from_numpy(self.features[idx]).type(torch.FloatTensor), \
               torch.from_numpy(self.labels[idx]).type(torch.FloatTensor)
        # Commented out: for cuda-enabled devices
        # return torch.from_numpy(self.features[idx]).type(torch.FloatTensor).cuda(), \
        #        torch.from_numpy(self.labels[idx]).type(torch.FloatTensor).cuda()
    

class RNNAutoregressor(L.LightningModule):
    # Defines LSTM-based Autoregressor/Autoencoder
    def __init__(self, test_pig, in_channels, hidden_size, window_size, forecast_size, output_size, 
                 hidden_layer, num_layers, learning_rate, weight_decay, l1_lambda, dropout, 
                 device_to_use, max_epochs, running_loss, direction, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.test_pig = test_pig
        self.device_to_use = device_to_use
        self.window_size = window_size
        self.forecast_size = forecast_size
        self.learning_rate = learning_rate
        self.l1_lambda = l1_lambda
        self.weight_decay = weight_decay
        self.epochs = max_epochs
        self.hidden_layer = hidden_layer
        self.num_layers = num_layers
        self.running_loss = running_loss
        self.direction = direction
        self.loss_fn = nn.MSELoss()

        # Separate classes for Encoder, Decoder for 
        # easier freezing of LSTM encoding in downstream task
        self.encoder = Encoder(in_channels, hidden_size, num_layers)
        self.decoder = Decoder(hidden_size, output_size, hidden_layer, dropout, num_layers)

    def forward(self, x):
        # Encode the input sequence; used in training and upstream task
        encoded_out, (hn, cn) = self.encoder(x)
        if self.forecast_size == 0:
            repeated_out = encoded_out[:, -1, :].unsqueeze(1).repeat(1, self.window_size, 1) 
        else:
            repeated_out = encoded_out[:, -1, :].unsqueeze(1).repeat(1, self.forecast_size, 1) 
        # Decode repeated output
        reconstructed_out, _ = self.decoder(repeated_out)
        
        return reconstructed_out
    
    def encode(self, x):
        # Function that only encodes; this is used for BVDS downstream task!
        encoder_out, _ = self.encoder(x)
        encoded_out = encoder_out[:, -1, :].unsqueeze(1)
        return encoded_out

    def training_step(self, batch, batch_idx):
        # What happens during the training cycle
        x, y = batch
        x, y = x.to(self.device_to_use), y.to(self.device_to_use)
        x_hat = self(x) #runs through fc layers, results in x_hat=(N,2,M,F)

        # Compare actual feature timeseries with output from LSTM
        loss = self.loss_fn(x_hat, y)

        # IF USING L1 (is a hyperparameter, usually set to 0)
        l1_loss = sum(torch.sum(torch.abs(param)) for param in self.parameters())
        loss += self.l1_lambda * l1_loss

        # Logging in wandb to plot training loss for each pig
        self.log(f'train_loss_pig_{self.test_pig}_{self.direction}', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)

        # Plotting final training epoch to show predicted vs actual feature timeseries
        if batch_idx == 0 and self.current_epoch > self.epochs - 2:
            frame_index = random.randint(0, x.size(0) - 1)
            features = [0, 1, 8, 9]
            plot_features(x, y, x_hat, frame_index, features, self.current_epoch, self.test_pig, 'train',
                           self.direction, self.forecast_size)
        
        return loss

    def validation_step(self, batch, batch_idx):
        # What happens during validation
        x, y = batch # x=(N,T,F) y=(N,2,M,F) if autoreg, (N,M,F) if autoencoder
        x, y = x.to(self.device), y.to(self.device)
        x_hat = self(x) # runs through fc layers
        loss = self.loss_fn(x_hat, y) 
        # Logging in wandb to plot validation loss for each pig
        self.log(f'val_loss_pig_{self.test_pig}_{self.direction}', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)

        # Plotting final validation epoch to show predicted vs actual feature timeseries
        if batch_idx == 0 and self.current_epoch > self.epochs - 2:
            frame_index = random.randint(0, x.size(0) - 1) #is seeding needed here or something?
            features = [0, 1, 8, 10]
            self.running_loss.append(loss)
            plot_features(x, y, x_hat, frame_index, features, self.current_epoch, self.test_pig, 'val',
                           self.direction, self.forecast_size)
            if self.test_pig == 6:
                # Important for selecting best model: Log mean validation loss of all 6 pigs for a model
                self.log(f'mean_val_loss', sum(self.running_loss)/len(self.running_loss), 
                         prog_bar=False, on_step=False, on_epoch=True)

        return loss


    # def predict_step(self, batch, batch_idx):
    # # Prediction step for reconstruction (commented out)
    #     x, y = batch
    #     x = x.to(self.device_to_use)
    #     x_hat = self(x)
    #     return x_hat

    def predict_step(self, batch, batch_idx):
        # Prediction step for BVDS downstream (encoder only)
        # Only runs through encoder and outputs hidden layer,
        # which can be used in downstrean feature regressor
        x, _ = batch
        x = x.to(self.device_to_use)
        encoded_out = self.encode(x)
        return encoded_out

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        return optimizer
    
class Encoder(nn.Module):
    # Encoding portion of Autoreg model:
    # Feed feature timeseries windows into n layers of LSTM 

    def __init__(self, in_channels, hidden_size, num_layers):
        super(Encoder, self).__init__()
        self.lstm1 = nn.LSTM(input_size=in_channels, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def forward(self, x):
        h0_lstm1 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0_lstm1 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out1, (hn1, cn1) = self.lstm1(x, (h0_lstm1, c0_lstm1))

        return out1, (hn1, cn1)
        
class Decoder(nn.Module):
    # Take hidden layer from LSTM and reconstruct features in timeseries

    def __init__(self, hidden_size, output_size, hidden_layer, dropout, num_layers):
        super(Decoder, self).__init__()
        # Hyperparameters can change LSTM dimensions, but will not affect downstream task
        self.lstm1 = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        if hidden_layer == 0:
            self.fc = nn.Sequential(
                nn.Linear(hidden_size, output_size),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(hidden_size, hidden_layer),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_layer, output_size)
            )
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def forward(self, x):
        h0_lstm1 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0_lstm1 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out1, (hn1, cn1) = self.lstm1(x, (h0_lstm1, c0_lstm1))
        out = self.fc(out1)

        return out, (hn1, cn1)
    
class MyDataModule(L.LightningDataModule):
    # DataLoader for each pig's feature timeseries and reconstruction windows,
    # which are either a forecast/backcast or an autoencoding of the same window

    def __init__(self, data_directory, num_pigs, test_pig_num, bvds_mode,
                 all_hypovolemia_stages, train_hypovolemia_stages, test_hypovolemia_stages,
                 batch_size=64, overlap_percentage=0.5, window_size = 30, forecast_size=15, direction=0, random_seed=42): 
        super().__init__()
        # train/test_hypovolemia_stages = ['Absolute', 'Relative', 'Resuscitation']
        # these are here in case you want to train/test your model using certain stages
        self.data_directory = data_directory
        # bvds_mode = 'classification' or 'regression' unchanged from original
        self.bvds_mode = 'classes' if bvds_mode=='classification' else 'labels'   
        self.label_name = 'Class' if bvds_mode=='classification' else 'BVDS' #kept this logic for future implementation for downstream task
        self.num_pigs = num_pigs
        self.test_pig_num = test_pig_num
        self.all_hypovolemia_stages = all_hypovolemia_stages
        self.train_hypovolemia_stages = train_hypovolemia_stages
        self.test_hypovolemia_stages = test_hypovolemia_stages
        self.batch_size = batch_size
        self.overlap_percentage = overlap_percentage   # how many samples should be shared by two consecutive windows
        self.window_size = window_size   # how many time steps you want in a single window
        self.forecast_size = forecast_size # how many samples you want to predict in the future/past
        self.prediction_mode = 'train'
        self.direction = direction
        self.random_seed = random_seed

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        self.data_dict = {}
        # data dict keys will be pig numbers\
        # each data_dict[pig_num] will be also a dictionary with key Absolute, Relative, Resuscitation
        # each of these is also a dictionary with events, and within events are 2 keys 'features' and 'labels'
        for pig_idx in range(1, self.num_pigs + 1):  # traverse pigs
            self.data_dict[pig_idx] = {}
            for hypovolemia_stage in self.all_hypovolemia_stages:
                self.data_dict[pig_idx][hypovolemia_stage] = {}
                # x: "features"
                # y: "labels"
                # Both features and labels are derived from timeseries for reconstruction
                pig_path = os.path.join(self.data_directory, f"Pig{pig_idx}")
                if hypovolemia_stage == "Relative": # cycle through the rel events
                    events = rel_events 
                elif hypovolemia_stage == "Absolute": # cycle through the abs events
                    events = abs_events
                else: # cycle through the res events
                    events = res_events

                stage_data = {}

                for event in events:
                    event_file = os.path.join(pig_path, f"{event}.csv")
                    if os.path.isfile(event_file):
                        # # Removed BVDS labels, since focus is on feature forecasting but kept for later implementation in downstream task
                        # labels =pd.read_csv(event_file)[self.label_name].values 
                        features = pd.read_csv(event_file)[feature_names].values
                        # Inside data dict, will be [pig][stage][event][features/labels]
                        stage_data[event] = {
                            'features': features,
                            'labels': features
                        }
                    
                self.data_dict[pig_idx][hypovolemia_stage] = stage_data
        # now split into training, validation, and testing sets by pig/stage
        self.prepare_train_val_test()

    def set_prediction_mode(self, mode):
        # Can set into different modes that only predict, or train, etc.
        self.prediction_mode = mode

    def prepare_train_val_test(self):
        # train dataset first
        pigs_for_training = [m+1 for m in range(self.num_pigs) if m+1 != self.test_pig_num]
        print("pigs for training: ", pigs_for_training)
        print("pig for testing: ", self.test_pig_num)

        # Preparing training set
        train_X = np.empty((0, self.window_size, len(feature_names)))
        val_X = np.empty((0, self.window_size, len(feature_names)))
        if self.forecast_size == 0: # For autoencoding, predicting the same window
            train_y = np.empty((0, self.window_size, len(feature_names)))
            val_y = np.empty((0, self.window_size, len(feature_names)))
        else: # For forecast/backcast autoregressor
            train_y = np.empty((0, self.forecast_size, len(feature_names)))
            val_y = np.empty((0, self.forecast_size, len(feature_names)))
        #Find which pigs/bvds stages are for training and create dataset with these features
        train_pairs, val_pairs = get_train_val_pairs(pigs_for_training, self.train_hypovolemia_stages, self.random_seed)

        # Create separate dataset for the training data
        for pair in train_pairs:
            (pig_idx, hypovolemia_stage) = pair
            event_dict = self.data_dict[pig_idx][hypovolemia_stage]
            # Loop through each individual event's data, but keep sorting based on hypovolemia stage
            for event, event_data in event_dict.items():
                features = event_data['features']
                labels = event_data['labels']
                ts_features, ts_labels = create_timeseries_dataset_with_labels(features, labels,
                    time_steps=self.window_size,
                    overlap_percentage=self.overlap_percentage,
                    forecast_size=self.forecast_size, 
                    direction=self.direction)
                if ts_features.shape[0] == 0:
                    continue

                train_X = np.vstack((train_X, ts_features))

                if self.forecast_size == 0:
                    train_y = np.vstack((train_y, ts_labels.reshape(-1, self.window_size, len(feature_names))))
                else:
                    train_y = np.vstack((train_y, ts_labels.reshape(-1, self.forecast_size, len(feature_names))))

        # Create separate dataset for the validation data
        for pair in val_pairs:
            (pig_idx, hypovolemia_stage) = pair
            event_dict = self.data_dict[pig_idx][hypovolemia_stage]
            for event, event_data in event_dict.items():
                features = event_data['features']
                labels = event_data['labels']
                ts_features, ts_labels = create_timeseries_dataset_with_labels(features, labels,
                    time_steps=self.window_size,
                    overlap_percentage=self.overlap_percentage,
                    forecast_size=self.forecast_size, 
                    direction=self.direction)
                if ts_features.shape[0] == 0:
                    continue
                
                val_X = np.vstack((val_X, ts_features))

                if self.forecast_size == 0:
                    val_y = np.vstack((val_y, ts_labels.reshape(-1, self.window_size, len(feature_names))))
                else:
                    val_y = np.vstack((val_y, ts_labels.reshape(-1, self.forecast_size, len(feature_names))))


        # For the test pig, create another dataset:
        test_X = np.empty((0, self.window_size, len(feature_names)))
        if self.forecast_size == 0:
            test_y = np.empty((0, self.window_size, len(feature_names)))
        else:
            test_y = np.empty((0, self.forecast_size, len(feature_names)))

        self.test_hypo_stages = []
        for hypovolemia_stage in self.test_hypovolemia_stages:
            # Retrieve the events dictionary for this stage, if it exists
            event_dict = self.data_dict.get(self.test_pig_num, {}).get(hypovolemia_stage, {})

            for event_name, event_data in event_dict.items():
                features = event_data['features']
                labels = event_data['labels']
                ts_features, ts_labels = create_timeseries_dataset_with_labels(features, labels,
                    time_steps=self.window_size,
                    overlap_percentage=self.overlap_percentage,
                    forecast_size=self.forecast_size, 
                    direction=self.direction)
                if ts_features.shape[0] == 0: # Add logic for if no CO labels
                    continue
                self.test_hypo_stages = self.test_hypo_stages + [hypovolemia_stage for _ in range(len(ts_features))]

                test_X = np.vstack((test_X, ts_features))
                if self.forecast_size == 0:
                    test_y = np.vstack((test_y, ts_labels.reshape(-1, self.window_size, len(feature_names))))
                else:
                    test_y = np.vstack((test_y, ts_labels.reshape(-1, self.forecast_size, len(feature_names))))
        self.test_hypo_stages = np.array(self.test_hypo_stages)

        print("Train x shape: ", train_X.shape)
        print("Val x shape: ", val_X.shape)
        print("Test x shape: ", test_X.shape)
        print("Test hypo stages length: ", self.test_hypo_stages.shape)

        # Changed to calculate mean, std from overall window (including forecast/backcast)
        # for consistent normalization
        mean, std = get_timeseries_standardizer(train_X) 
        train_X_std = (train_X - mean) / std
        val_X_std = (val_X - mean) / std
        test_X_std = (test_X - mean) / std
        train_Y_std = (train_y - mean) / std
        val_Y_std = (val_y - mean) / std
        test_Y_std = (test_y - mean) / std   

        self.train = TimeSeriesDataset(train_X_std, train_Y_std, [])
        self.validate = TimeSeriesDataset(val_X_std, val_Y_std, [])
        self.test = TimeSeriesDataset(test_X_std, test_Y_std, self.test_hypo_stages)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True)
    def val_dataloader(self):
        return DataLoader(self.validate, batch_size=self.batch_size, shuffle=False)
    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, shuffle=False)
    def predict_dataloader(self):
        if self.prediction_mode == 'train':
            return DataLoader(self.train, batch_size=self.batch_size, shuffle=False) #Should the shuffle be set to True?
        elif self.prediction_mode == 'val':
            return DataLoader(self.validate, batch_size=self.batch_size, shuffle=False)
        elif self.prediction_mode == 'test':
            return DataLoader(self.test, batch_size=self.batch_size, shuffle=False)


def create_timeseries_dataset_with_labels(data, labels, time_steps, overlap_percentage, forecast_size, direction):
    """
    Creates a timeseries dataset from the given data along with corresponding labels.

    Parameters:
        data (numpy.ndarray): The original dataset with shape N x F.
        labels (numpy.ndarray): The label matrix with shape 2 x M x F.
        time_steps (int): The number of time steps per sample.
        overlap_percentage (float): The percentage of time steps two consecutive windows share (between 0 and 1).

    Returns:
        tuple: A tuple containing timeseries dataset with shape M x T x F, where M is the number of samples,
               T is the number of time steps, and F is the number of features, and corresponding labels.
    """
    num_samples = data.shape[0]

    # Calculate the number of overlapping time steps
    overlap_steps = int(time_steps * overlap_percentage)

    # Calculate the step size between consecutive windows
    step_size = time_steps - overlap_steps

    # Calculate the number of samples: Exclude the backwards/forwards forecasting size and window size
    num_windows = (num_samples - time_steps - 2 * forecast_size) // step_size + 1

    # Initialize lists to store valid windows and their labels
    valid_windows = []
    valid_labels = []

    # Populate the timeseries dataset and labels
    for i in range(num_windows):
        start_idx = i * step_size + forecast_size # leave room for backwards casting
        end_idx = start_idx + time_steps 
        window_data = data[start_idx:end_idx]

        if forecast_size == 0:
            window_labels = window_data
        else:
            if direction == 0: #backwards = 0
                window_labels = labels[start_idx - forecast_size:start_idx]
            else: #forwards direction = 1
                window_labels = labels[end_idx:end_idx + forecast_size]

        valid_windows.append(window_data)
        valid_labels.append(window_labels)

    # Convert lists to numpy arrays
    valid_windows = np.array(valid_windows)
    valid_labels = np.array(valid_labels)

    return valid_windows, valid_labels

def get_timeseries_standardizer(data):
    # Standardize time series data
    # inputs: numpy array with shape NumSamples x SequenceLength x Features
    # outputs: standardized_data: np array with the same shape as data, standardized along the Features axis
    # Compute mean and standard deviation across NumSamples and SequenceLength axes
    mean = np.mean(data, axis=(0, 1), keepdims=True)
    std = np.std(data, axis=(0, 1), keepdims=True)
    return mean, std

def get_train_val_pairs(subject_indices, hypovolemia_stages, random_seed):
    # create pairs of subjects x hypovolemia stages
    # pick 20% and use them as validaiton
    # this one picks one hypovolemia stage directly - since there are 5 training pigs, it corresponds to 20% directly
    pairs = [(x, y) for x in subject_indices for y in hypovolemia_stages]
    # Create separate lists for each element of
    hypo_stage_pairs = {item: [] for item in hypovolemia_stages}
    # each key is a hypovolemia stage, each value is a list of tuples (pig_idx, hypovolemia stage)
    for pair in pairs:
        hypo_stage_pairs[pair[1]].append(pair)
    # Shuffle each list. Reseed random sequence so it is repeatable in downstream.
    random.seed(random_seed)
    for key in hypo_stage_pairs:
        random.shuffle(hypo_stage_pairs[key])

    # Take one example from each list for validation set
    val_pairs = [item.pop() for item in hypo_stage_pairs.values()]

    # Flatten the remaining pairs for training set
    train_pairs = [pair for sublist in hypo_stage_pairs.values() for pair in sublist]
    print("Train pairs:", train_pairs)
    print("Validation pairs:", val_pairs)

    return train_pairs, val_pairs

def get_filename_with_min_val_loss(directory):
    import re
    # Define the pattern to match filenames
    pattern = r'model-\d{4}-\d{4}-epoch=(\d+)-val_loss=([\d.]+)'
    # Initialize variables to store the minimum validation loss and corresponding filename
    min_val_loss = float('inf')
    best_model_filename = None
    # Iterate over files in the directory
    for filename in os.listdir(directory):
        # Check if the filename matches the pattern
        match = re.match(pattern, filename[:-5])
        if match:
            epoch, val_loss = map(float, match.groups())
            # Update the minimum validation loss and corresponding filename if needed
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                best_model_filename = filename
    # If a best model filename is found, print it
    if best_model_filename:
        best_model_path = os.path.join(directory, best_model_filename)
        print("Best model found:", best_model_path)
    else:
        best_model_path = None
        print("No model files found in the directory.")
    return best_model_path

def plot_features(x, y, x_hat, frame_index, features, epoch, test_pig, mode, direction, forecast_size):
    # Plots features of a random sample window. 
    # Includes predicted LSTM feature layed over actual features.
    # Only shows 4 features max; edit which features using parameter.
    if len(features) > 4:
        features = features[0:3]

    fig, axs = plt.subplots(len(features), 1, figsize=(12, 5 * len(features)))

    for i in range(len(features)):
        feature_index = features[i]
        reconstruct = x_hat[frame_index, :, feature_index].cpu().detach().numpy()
        actual = y[frame_index, :, feature_index].cpu().detach().numpy()
        window = x[frame_index, :, feature_index].cpu().detach().numpy()

        if forecast_size == 0: #If autoencoder, simply plot prediction over existing timeseries window
            full_window = actual
            cast = reconstruct
            axs[i].plot(cast, label='Predicted Autoencoder', color='red', linestyle='dashed')
        else:
            if direction == 0: #If backcast
                full_window = np.concatenate((actual, window))
                cast = np.pad(reconstruct, (0, len(full_window) - len(actual)), 'constant', constant_values=np.nan)
                axs[i].plot(cast, label='Predicted Backcast', color='red', linestyle='dashed')
            else: #If forecast
                full_window = np.concatenate((window, actual))
                cast = np.pad(reconstruct, (len(full_window) - len(actual), 0), 'constant', constant_values=np.nan)
                axs[i].plot(cast, label='Predicted Forecast', color='red', linestyle='dashed')

        axs[i].plot(full_window, label='Actual', color='blue')
        axs[i].set_title(f'Sample Predicted vs Actual for Epoch {epoch}, Feature: {feature_index}')
        axs[i].set_xlabel('Time Step')
        axs[i].set_ylabel('Value')
        axs[i].legend()
        axs[i].grid(True)

    plt.tight_layout()

    # Save the plot to a temporary file
    timestr = time.strftime("%m%d-%H%M%S")
    plt_path = f"plots//{mode}_pig{test_pig}_{direction}_{epoch}_{timestr}.png"
    fig.savefig(plt_path)
    plt.close()

    # Log the plot to wandb
    wandb.log({f"plots//{mode}_pig{test_pig}_{direction}_{epoch}_{timestr}": wandb.Image(plt_path)})


#############################################################################################################
#############################################################################################################
#################################### MAIN CODE below ########################################################
#############################################################################################################
#############################################################################################################

def objective():

    if not wandb.run:
        wandb.init(project="RNNAutoregressor", save_code=True)
    wandb.init(settings=wandb.Settings(code_dir="."))
    config = wandb.config

    # Check if running locally or on server
    hostname = socket.gethostname()
    # Find data directory based on host machine (hardcoded to Samuel Liu's environments)
    if hostname == 'samue':
        # Directory with code + model results
        project_dir = r"C:\\Users\\samue\\OneDrive - Georgia Institute of Technology\\GT Files\\ECE 8903 I02\\SamuelCode\\HypovolemiaSamuel"
        # Directory containing processed beatseries data
        data_dir = r"C:\\Users\\samue\\GaTech Dropbox\\ECE\\InanResearchLab\\DocumentedDatasets\\Hypovolemia\\COBVDSAlignedData\\Feature Beatseries (Normalized)"
    # If machine is using the lab server, update directories accordingly
    elif hostname == 'jarvis':
        project_dir = "/home/sliu/Desktop/SamuelCode/bvds/HypovolemiaSamuel"
        data_dir = "/home/sliu/Desktop/Data"
    else:
        raise ValueError("Unknown environment")
    
    # Set random seed for reproducibility
    random_seed = config.rand_seed # Also updates in get_train_val_pairs !!!
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(random_seed)

    mode = 'train'   # train or test
    # train - trains the model and saves the best model in terms of validation loss
    # test - loads the model with the minimum loss inside Model directory

    training_mode = 'regression'  # regression or classification
    # regression maps to [0, 100] interval
    # classification maps to {0, 25, 33, 50, 67, 75, 100}

    all_hypovolemia_stages = ["Absolute", "Relative", "Resuscitation"]
    train_hypovolemia_stages = ["Absolute", "Relative"]
    test_hypovolemia_stages = ["Absolute", "Relative", "Resuscitation"]

    # the following creates Abs-Rel if you use Absolute and Relative to train your model
    folder_name_to_save = '-'.join([m[:3] for m in train_hypovolemia_stages])

    # this is where you save your results
    result_output_dir = os.path.join(project_dir, 'Results', 'TimeSeriesModel', folder_name_to_save)
    if not os.path.isdir(result_output_dir):
        os.makedirs(result_output_dir)

    num_pigs = 6
    window_size = config.window_size
    batch_size = 128
    test_pig_nums = [1, 2, 3, 4, 5, 6] # for each pig we create a different model by excluding that pig
    # some parameters we might play with; Edit in HYPERPARAMETER section at bottom
    hidden_size = config.hidden_size
    hidden_layer = config.hidden_layer
    num_layers = config.num_layers
    forecast_size = config.forecast_size
    learning_rate = config.learning_rate
    weight_decay = config.weight_decay
    l1_lambda = config.l1_lambda
    dropout = config.dropout
    epochs = config.epochs
    overlap = config.overlap
    
    # N * T * F
        # N: batch size
        # T: sequence length
        # F: 12 (num channels)
    #Y: N * 2 * M * F
        # 2 is for future/past prediction
        # M: length of forecast/

##########################################################################################################
################# Section is for TRAINING a new model ####################################################

    if mode == 'train':
        running_loss = []
        #model run name timestamp for easy access
        model_run_str = time.strftime("%m%d-%H%M")
        wandb.log({"autoreg_model_str": model_run_str})
        if forecast_size == 0:
            wandb.run.name = model_run_str + "_autoenc"
        else:
            wandb.run.name = model_run_str
        wandb.run.save()
        for test_pig_num in test_pig_nums:
            print(f"Training for Pig {test_pig_num} Started")
            # where you save your model
            model_output_dir = os.path.join(project_dir, 'Models', 'TimeSeriesModel',
                                            folder_name_to_save, f'Pig{test_pig_num}-{training_mode}')
            if not os.path.exists(model_output_dir):
                os.makedirs(model_output_dir)
                
            if forecast_size > 0: # Creates backwards autoregressor if not autoencoder

                # Creates DataLoader w/ timeseries backcast
                data_module_backward = MyDataModule(
                    data_dir, num_pigs, test_pig_num, training_mode,
                    all_hypovolemia_stages, train_hypovolemia_stages, test_hypovolemia_stages,
                    batch_size=batch_size, overlap_percentage=overlap, window_size=window_size, 
                    forecast_size=forecast_size, 
                    direction=0, # direction 0 = backwards,
                    random_seed=random_seed
                    )
                
                # Creates Backwards-Predicting Autoregressor model 
                model_backward = RNNAutoregressor(
                    test_pig=test_pig_num, in_channels=len(feature_names), 
                    hidden_size=hidden_size, window_size=window_size, 
                    forecast_size=forecast_size, output_size=len(feature_names),
                    hidden_layer=hidden_layer, num_layers=num_layers,
                    learning_rate=learning_rate, weight_decay=weight_decay, 
                    l1_lambda=l1_lambda, dropout=dropout, device_to_use=DEVICE, 
                    max_epochs=epochs, running_loss=running_loss, 
                    direction=0, # direction 0 = backwards
                    )
                
                # Checkpoint callback for Backwards Autoencoder
                checkpoint_callback_backward = ModelCheckpoint(
                    dirpath=model_output_dir,
                    filename='model_b-' + model_run_str + '-{epoch:02d}-{val_loss:.2f}',
                    monitor='val_loss',   # we want to save the model based on validation loss
                    mode='min',   # we want to minimize validation loss
                    save_top_k=1
                )

                #data_module_backward.setup() # Uncomment, run if not using trainer

            # Creates DataLoader w/ timeseries forecast OR autoencoder if forecast=0
            data_module_forward = MyDataModule(
                data_dir, num_pigs, test_pig_num, training_mode,
                all_hypovolemia_stages, train_hypovolemia_stages, 
                test_hypovolemia_stages, batch_size=batch_size, 
                overlap_percentage=overlap, window_size=window_size, 
                forecast_size=forecast_size, 
                direction=1, # direction 1 = forwards
                random_seed=random_seed
                )
            
            # Creates Forwards-Predicting Autoregressor Model OR Autoencoder if forecast=0
            model_forward = RNNAutoregressor(
                test_pig=test_pig_num, in_channels=len(feature_names), 
                hidden_size=hidden_size, window_size=window_size, 
                forecast_size=forecast_size, output_size=len(feature_names),
                hidden_layer=hidden_layer, num_layers=num_layers,
                learning_rate=learning_rate, weight_decay=weight_decay, 
                l1_lambda=l1_lambda, dropout=dropout, device_to_use=DEVICE, 
                max_epochs=epochs, running_loss=running_loss, 
                direction=1, # direction 1 = forwards
                ) 
            
            # Checkpoint callback for Forwards Autoregressor/Autoencoder
            checkpoint_callback_forward = ModelCheckpoint(
                dirpath=model_output_dir,
                filename='model_f-' + model_run_str + '-{epoch:02d}-{val_loss:.2f}',
                monitor='val_loss',   # we want to save the model based on validation loss
                mode='min',   # we want to minimize validation loss
                save_top_k=1
            )

            # Initialize wandb logger to view stats
            wandb_logger = WandbLogger(log_model=True)

            # Two trainers, one for backwards, one for forwards. 
            # Creating backwards trainer if not autoencoder
            if forecast_size > 0:
                trainer_backwards = L.Trainer(
                    logger=wandb_logger, callbacks=[MyProgressBar(), 
                    checkpoint_callback_backward], max_epochs=epochs
                    )
                trainer_backwards.fit(model_backward, data_module_backward)

            # Creating trainer for forwards autoregressor or autoencoder
            trainer_forwards = L.Trainer(
                logger=wandb_logger, callbacks=[MyProgressBar(), 
                checkpoint_callback_forward], max_epochs=epochs
                )
            trainer_forwards.fit(model_forward, data_module_forward)


##########################################################################################################
################# Section is for TESTING, assuming you have trained a model already ######################
    else:
        results = {}
        for test_pig_num in test_pig_nums:
            model_output_dir = os.path.join(project_dir, 'Models', 'TimeSeriesModel',
                                            folder_name_to_save, f'Pig{test_pig_num}-{training_mode}')
            print(f"Testing for Pig {test_pig_num} Started")
            checkpoint_path = get_filename_with_min_val_loss(model_output_dir)
            print("Called model checkpoint path: ", checkpoint_path)

            # load the trained model
            model = RNNAutoregressor.load_from_checkpoint(checkpoint_path)

            data_module = MyDataModule(data_dir, num_pigs, test_pig_num, training_mode,
                                       all_hypovolemia_stages, train_hypovolemia_stages, test_hypovolemia_stages,
                                       batch_size=batch_size, overlap_percentage=overlap,
                                       window_size=window_size)
            data_module.set_prediction_mode('test')
            model.eval()
            trainer = L.Trainer()
            predictions = trainer.predict(model, data_module)
            predictions = torch.cat(predictions, dim=0).numpy()
            gt_data = data_module.test.labels
            if training_mode == 'classification':
                # first convert class labels to correct values
                gt_data = np.vectorize(classes_to_bvds_dict.get)(np.squeeze(gt_data))
                predictions = np.vectorize(classes_to_bvds_dict.get)(np.squeeze(predictions))
            predictions = np.squeeze(predictions)
            gt_data = np.squeeze(gt_data)
            # we do not want to return predictions that are out of bounds
            predictions[predictions > 100] = 100
            predictions[predictions < 0] = 0
            print()
        
        # now all median loss
        print("Global Results")
        for stage in test_hypovolemia_stages + ['overall']:
            print("Stage: ", stage)
            all_pig_results = []
            for pig_num in test_pig_nums:
                all_pig_results.append(results[pig_num][stage])
            print(f"Median RMSE error of {stage}: ", np.median(all_pig_results))


    wandb.finish()

##########################################################################################################
###################### Section is for SETTING HYPERPARAMETERS ############################################
    
if __name__ == "__main__":
    
    ###### Change to True to sweep of hyperparameters! #########
    perform_sweep = False 
    wandbproject = "RNNAutoregressor"
    ###########################################################

    # For Sweeps: Define the search space below
    sweep_configuration = {
        "method": "grid",
        "metric": {"goal": "minimize", "name": "mean_val_loss"},
        "parameters": {
            "learning_rate": {"values": [0.0001]},
            "weight_decay": {"values": [0.0007]},
            "l1_lambda": {"values": [0]},
            "hidden_size": {"values": [256]}, 
            "forecast_size": {"values": [15]}, # Set 0 for Autoencoder, >=1 for predictive Autoregressor
            "overlap": {"values": [0.9]},
            "epochs": {"values": [40]},
            "hidden_layer": {"values": [128]}, # Keep at 0, unless you want multiple FC layers in DECODER
            "num_layers": {"values": [1]}, # Layers of LSTM 
            "dropout": {"values": [0]}, # Dropout rate
            "window_size": {"values": [30]}, # Timeseries window to train on
            "rand_seed": {"values": [45, 46]} # Random seeds
        },
    }

    # Initialize wandb for hyperparameter sweep
    if perform_sweep:
        sweep_id = wandb.sweep(sweep=sweep_configuration, project=wandbproject)
        wandb.agent(sweep_id, function=objective, count=5)
    else:

        # For Non-sweeps, edit hyperparameters for single runs
        wandb.init(project=wandbproject, config={
            "learning_rate": 0.0001,
            "weight_decay": 0.00001,
            "l1_lambda": 0.00000,
            "hidden_size": 128,
            "forecast_size": 10, # Set 0 for Autoencoder, >=1 for predictive Autoregressor
            "overlap": 0.9, # Percentage overlap of timeseries training windows
            "epochs": 30,
            "hidden_layer": 128, # Keep at 0, unless you want multiple FC layers in DECODER
            "num_layers": 1, # Layers of LSTM 
            "dropout": 0, # Dropout rate
            "window_size": 30, # Timeseries window to train on
            "rand_seed": 46
        }, save_code=True)

        objective()
