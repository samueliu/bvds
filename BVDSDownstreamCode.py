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
from sklearn.metrics import mean_squared_error
import random
import sys
import wandb
import time
import socket
from SamuelCode_LSTM import RNNAutoregressor
from pytorch_lightning.loggers import WandbLogger
# from SamuelCode_LSTM import random_seed

########################### Downstream Code for Forecast/Backcast ONLY #############################################

# Set random seed for reproducibility
# NOTE: Seed MUST be IDENTICAL to one used in upstream LSTM!
# OR ELSE train/val/test sets may differ in downstream which will lead to incorrect training

random_seed = 46 # Set seed manually. Or comment out+import from LSTM code (not recommended)
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ['PYTHONHASHSEED'] = str(random_seed)

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
    def __getitem__(self, idx):
        return torch.from_numpy(self.features[idx]).type(torch.FloatTensor), \
               torch.from_numpy(self.labels[idx]).type(torch.FloatTensor) #.cuda() on both return statements?
    
class BVDSRegressor(L.LightningModule):
    # Definition of downstream regressor model for predicting BVDS 
    def __init__(self, test_pig, hidden_size, learning_rate, weight_decay, l1_lambda, dropout, 
                 device_to_use, max_epochs, hidden_layer, num_layers):
        super().__init__()
        self.save_hyperparameters()
        self.test_pig = test_pig
        self.device_to_use = device_to_use
        self.learning_rate = learning_rate
        self.l1_lambda = l1_lambda
        self.weight_decay = weight_decay
        self.epochs = max_epochs
        self.hidden_layer = hidden_layer
        self.num_layers = num_layers
        
        # Different architecture depending on hyperparameter
        # for number of fully connected layers
        if self.num_layers == 2:
            self.bvds_regressor = nn.Sequential(
                # FC layer uses hidden_size * 2 due to forwards and backwards outputs concatenated
                nn.Linear(hidden_size*2, self.hidden_layer), 
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(self.hidden_layer, 1),
            )
        else:
            self.bvds_regressor = nn.Sequential(
                # FC layer uses hidden_size * 2 due to forwards and backwards outputs concatenated
                nn.Linear(hidden_size*2, 1),
                nn.ReLU(),
                nn.Dropout(dropout),
            )

        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        # decode LSTM's output into BVDS value
        bvds = self.bvds_regressor(x)
        return bvds

    def training_step(self, batch, batch_idx):
        x, y = batch
        x, y = x.to(self.device_to_use), y.to(self.device_to_use)
        x_hat = self(x) #runs through fc layers, results in x_hat=(N,1)
        loss = self.loss_fn(x_hat, y)
        l1_loss = sum(torch.sum(torch.abs(param)) for param in self.parameters())
        loss += self.l1_lambda * l1_loss
        self.log(f'train_loss_pig_{self.test_pig}', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        x_hat = self(x) #runs through fc layers
        loss = self.loss_fn(x_hat, y) 
        l1_loss = sum(torch.sum(torch.abs(param)) for param in self.parameters())
        loss += self.l1_lambda * l1_loss
        self.log(f'val_loss_pig_{self.test_pig}', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def predict_step(self, batch, batch_idx):
        # Prediction step which is useful for "predict mode"
        x, y = batch
        x = x.to(self.device_to_use)
        x_hat = self(x)
        return x_hat

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay) #adjust learning rate here
        return optimizer

class MyDataModule(L.LightningDataModule):
    # DataLoader for each pig's feature timeseries and BVDS features.
    # Used for upstream LSTM encoding, but only in PREDICT mode.
    # BVSD features were not loaded in LSTM DataLoader, 
    # so they are recreated here with identical train/val/test partitions

    def __init__(self, data_directory, num_pigs, test_pig_num, bvds_mode,
                 all_hypovolemia_stages, train_hypovolemia_stages, test_hypovolemia_stages,
                 batch_size=64, overlap_percentage=0.5, window_size = 30, forecast_size=15):
        super().__init__()
        # train/test_hypovolemia_stages = ['Absolute', 'Relative', 'Resuscitation']
        # these are here in case you want to train/test your model using certain stages
        self.data_directory = data_directory
        self.bvds_mode = 'classes' if bvds_mode=='classification' else 'labels'   # bvds_mode = 'classification' or 'regression' unchanged from original
        
        # NOTE: Can set this to be CO or BVDS value based on what we downstream task is...
        self.label_name = 'Class' if bvds_mode=='classification' else 'bvds' #kept this logic for future implementation for downstream task
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

    def prepare_data(self):
        pass

    def set_prediction_mode(self, mode):
        # Can set into different modes that only predict, or train, etc.
        self.prediction_mode = mode

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
                        labels = pd.read_csv(event_file)[self.label_name].values 
                        features = pd.read_csv(event_file)[feature_names].values
                        # Inside data dict, will be [pig][stage][event][features/labels]
                        stage_data[event] = {
                            'features': features,
                            'labels': labels
                        }
                    
                self.data_dict[pig_idx][hypovolemia_stage] = stage_data# now split into training, validation, and testing sets by pig/stage
        self.prepare_train_val_test()

    def prepare_train_val_test(self):
        # train dataset first
        pigs_for_training = [m+1 for m in range(self.num_pigs) if m+1 != self.test_pig_num]
        train_X = np.empty((0, self.window_size, len(feature_names)))
        train_y = np.empty((0, 1))
        val_X = np.empty((0, self.window_size, len(feature_names)))
        val_y = np.empty((0, 1))
        #Find which pigs/bvds stages are for training and create dataset with these features
        train_pairs, val_pairs = get_train_val_pairs(pigs_for_training, self.train_hypovolemia_stages)

        # training dataset
        for pair in train_pairs:
            (pig_idx, hypovolemia_stage) = pair
            event_dict = self.data_dict[pig_idx][hypovolemia_stage]
            for event, event_data in event_dict.items():
                features = event_data['features']
                labels = event_data['labels']
                ts_features, ts_labels = create_timeseries_dataset_with_labels(features, labels,
                                                                           time_steps=self.window_size,
                                                                           overlap_percentage=self.overlap_percentage)
                if ts_features.shape[0] == 0:
                    continue
                train_X = np.vstack((train_X, ts_features))
                train_y = np.vstack((train_y, ts_labels.reshape(-1, 1)))
        # validation dataset
        for pair in val_pairs:
            (pig_idx, hypovolemia_stage) = pair
            event_dict = self.data_dict[pig_idx][hypovolemia_stage]
            for event, event_data in event_dict.items():
                features = event_data['features']
                labels = event_data['labels']
                ts_features, ts_labels = create_timeseries_dataset_with_labels(features, labels,
                                                                           time_steps=self.window_size,
                                                                           overlap_percentage=self.overlap_percentage)
                if ts_features.shape[0] == 0:
                    continue
                val_X = np.vstack((val_X, ts_features))
                val_y = np.vstack((val_y, ts_labels.reshape(-1, 1)))

        # for the test pig
        test_X = np.empty((0, self.window_size, len(feature_names)))
        test_y = np.empty((0, 1))
        self.test_hypo_stages = []
        for hypovolemia_stage in self.test_hypovolemia_stages:
            # Retrieve the events dictionary for this stage, if it exists
            event_dict = self.data_dict.get(self.test_pig_num, {}).get(hypovolemia_stage, {})

            for event_name, event_data in event_dict.items():
                features = event_data['features']
                labels = event_data['labels']
                ts_features, ts_labels = create_timeseries_dataset_with_labels(features, labels,
                                                                           time_steps=self.window_size,
                                                                           overlap_percentage=self.overlap_percentage)
                if ts_features.shape[0] == 0:
                    continue
                self.test_hypo_stages = self.test_hypo_stages + [hypovolemia_stage for _ in range(len(ts_features))]
                test_X = np.vstack((test_X, ts_features))
                test_y = np.vstack((test_y, ts_labels.reshape(-1, 1)))
        self.test_hypo_stages = np.array(self.test_hypo_stages)

        mean, std = get_timeseries_standardizer(train_X) #changed to calculate mean, std from overall window (including forecast/backcast) for consistent normalization
        train_X_std = (train_X - mean) / std
        val_X_std = (val_X - mean) / std
        test_X_std = (test_X - mean) / std 

        self.train = TimeSeriesDataset(train_X_std, train_y, [])
        self.validate = TimeSeriesDataset(val_X_std, val_y, [])
        self.test = TimeSeriesDataset(test_X_std, test_y, self.test_hypo_stages)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True) 
    def val_dataloader(self):
        return DataLoader(self.validate, batch_size=self.batch_size, shuffle=False)
    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, shuffle=False)
    #IMPORTANT: Change predict mode between train, val, test, etc.
    def predict_dataloader(self):
        if self.prediction_mode == 'train':
            return DataLoader(self.train, batch_size=self.batch_size, shuffle=False)#Should the shuffle be set to True?
        elif self.prediction_mode == 'val':
            return DataLoader(self.validate, batch_size=self.batch_size, shuffle=False)
        elif self.prediction_mode == 'test':
            return DataLoader(self.test, batch_size=self.batch_size, shuffle=False)

class DownstreamDataModule(L.LightningDataModule):
    # Contains the outputs from the LSTM encoding step as the x "features"
    # And the same BVDS y labels from the previous dataloader
    # Will be used in BVDS downstream regressor
    def __init__(self, encoded_predictions_train, bvds_labels_train, encoded_predictions_val, bvds_labels_val,
                  encoded_predictions_test, bvds_labels_test, bvds_hypo_stages_test, num_pigs, test_pig_num, bvds_mode,
                 all_hypovolemia_stages, train_hypovolemia_stages, test_hypovolemia_stages,
                 batch_size=64):
        super().__init__()
        # train/test_hypovolemia_stages = ['Absolute', 'Relative', 'Resuscitation']
        # these are here in case you want to train/test your model using certain stages
        self.bvds_mode = 'classes' if bvds_mode=='classification' else 'labels'   # bvds_mode = 'classification' or 'regression' unchanged from original
        self.label_name = 'Class' if bvds_mode=='classification' else 'BVDS' #kept this logic for future implementation for downstream task
        self.num_pigs = num_pigs
        self.test_pig_num = test_pig_num
        # "Hyperparameters" that are from the previous DataLoader or
        # are hidden layer outputs from the LSTM encoder
        self.all_hypovolemia_stages = all_hypovolemia_stages
        self.train_hypovolemia_stages = train_hypovolemia_stages
        self.test_hypovolemia_stages = test_hypovolemia_stages
        self.batch_size = batch_size
        self.encoded_predictions_train = encoded_predictions_train
        self.bvds_labels_train = bvds_labels_train
        self.encoded_predictions_val = encoded_predictions_val
        self.bvds_labels_val = bvds_labels_val
        self.encoded_predictions_test = encoded_predictions_test
        self.bvds_labels_test = bvds_labels_test
        self.bvds_hypo_stages_test = bvds_hypo_stages_test

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        # now split
        self.prepare_train_val_test()

    def prepare_train_val_test(self):
        # train dataset first
        pigs_for_training = [m+1 for m in range(self.num_pigs) if m+1 != self.test_pig_num]
        print("pigs for training: ", pigs_for_training)
        print("pig for testing: ", self.test_pig_num)
        train_X = self.encoded_predictions_train.squeeze()
        train_y = self.bvds_labels_train
        val_X = self.encoded_predictions_val.squeeze()
        val_y = self.bvds_labels_val
        test_X = self.encoded_predictions_test.squeeze()
        test_y = self.bvds_labels_test

        print("DOWNSTREAM Train x shape: ", train_X.shape)
        print("DOWNSTREAM Val x shape: ", val_X.shape)
        print("DOWNSTREAM Test x shape: ", test_X.shape)

        self.train = TimeSeriesDataset(train_X, train_y, [])
        self.validate = TimeSeriesDataset(val_X, val_y, [])
        self.test = TimeSeriesDataset(test_X, test_y, self.bvds_hypo_stages_test)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True)
    def val_dataloader(self):
        return DataLoader(self.validate, batch_size=self.batch_size, shuffle=False)
    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, shuffle=False)
    #IMPORTANT: Change predict mode between train, val, test, etc.
    def predict_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, shuffle=False)


def create_timeseries_dataset_with_labels(data, labels, time_steps, overlap_percentage):
    """
    Creates a timeseries dataset from the given data along with corresponding labels.
    This one assumes that if a certain window contain more than one label, that window is dropped

    Parameters:
        data (numpy.ndarray): The original dataset with shape N x F.
        labels (numpy.ndarray): The label matrix with shape N x F.
        time_steps (int): The number of time steps per sample.
        overlap_percentage (float): The percentage of time steps two consecutive windows share (between 0 and 1).

    Returns:
        tuple: A tuple containing timeseries dataset with shape M x T x F, where M is the number of samples,
               T is the number of time steps, and F is the number of features, and corresponding labels.
    """
    num_samples = data.shape[0]
    num_features = data.shape[1]

    # Calculate the number of overlapping time steps
    overlap_steps = int(time_steps * overlap_percentage)

    # Calculate the step size between consecutive windows
    step_size = time_steps - overlap_steps

    # Calculate the number of samples
    num_windows = (num_samples - time_steps) // step_size + 1

    # Initialize lists to store valid windows and their labels
    valid_windows = []
    valid_labels = []

    # Populate the timeseries dataset and labels
    for i in range(num_windows):
        start_idx = i * step_size
        end_idx = start_idx + time_steps
        window_data = data[start_idx:end_idx]
        window_labels = labels[start_idx:end_idx]

        # Check if all labels in the window are the same
        if np.all(window_labels == window_labels[0]):
            valid_windows.append(window_data)
            valid_labels.append(window_labels[0])

    # Convert lists to numpy arrays
    valid_windows = np.array(valid_windows)
    valid_labels = np.array(valid_labels)

    return valid_windows, valid_labels

def get_timeseries_standardizer(data):
    # Standardize time series data
    # inputs: numpy array with shape NumSamples x SequenceLength x Features
    # outputs: standardized_data: numpy array with the same shape as data, but standardized along the Features axis
    # Compute mean and standard deviation across NumSamples and SequenceLength axes
    mean = np.mean(data, axis=(0, 1), keepdims=True)
    std = np.std(data, axis=(0, 1), keepdims=True)
    return mean, std

def get_train_val_pairs(subject_indices, hypovolemia_stages):
    # create pairs of subjects x hypovolemia stages
    # pick 20% and use them as validaiton
    # this one picks one hypovolemia stage directly - since there are 5 training pigs, it corresponds to 20% directly

    pairs = [(x, y) for x in subject_indices for y in hypovolemia_stages]
    # Create separate lists for each element of
    hypo_stage_pairs = {item: [] for item in hypovolemia_stages}
    # each key is a hypovolemia stage, each value is a list of tuples (pig_idx, hypovolemia stage)
    for pair in pairs:
        hypo_stage_pairs[pair[1]].append(pair)

    # Shuffle each list. Reseed random sequence so it has IDENTICAL pairings as upstream task!!!
    # Using same seed as in LSTM model
    # NOTE: Need to look into this if there is a better way to ensure. Maybe have train/val pairs saved 
    # and imported from LSTM model?
    random.seed(random_seed)
    for key in hypo_stage_pairs:
        random.shuffle(hypo_stage_pairs[key])

    # Take one example from each list for validation set
    val_pairs = [item.pop() for item in hypo_stage_pairs.values()]

    # Flatten the remaining pairs for training set
    train_pairs = [pair for sublist in hypo_stage_pairs.values() for pair in sublist]
    # print("Train pairs:", train_pairs)
    # print("Validation pairs:", val_pairs)

    return train_pairs, val_pairs

def get_filename_with_min_val_loss(directory, model_string, direction):
    import re
    # Define the pattern to match filenames
    if direction != '':
        direction = '_' + direction
    pattern = r'model'+ direction +r'-epoch=(\d+)-val_loss=([\d.]+)'
    # pattern = r'model-\d{4}-\d{4}-epoch=(\d+)-val_loss=([\d.]+)' # for future use
    if model_string != '':
        pattern = rf'model{direction}-{model_string}-epoch=(\d+)-val_loss=([\d.]+)'

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

def report_results(gt, predictions, hypovolemia_stages, results, subj_idx):
    # Find overall test results for each stage, pig, and average
    overall_rmse = mean_squared_error(gt, predictions, squared=False)
    unique_stages = list(np.unique(hypovolemia_stages))
    print("unique stages: ", unique_stages)
    results[subj_idx] = {}
    for stage in unique_stages:
        gt_stage = gt[hypovolemia_stages == stage]
        predictions_stage = predictions[hypovolemia_stages == stage]
        stage_rmse = mean_squared_error(gt_stage, predictions_stage, squared=False)
        print(f"{stage} RMSE score: ", stage_rmse)
        results[subj_idx][stage] = stage_rmse
    print("Overall RMSE score: ", overall_rmse)
    results[subj_idx]['overall'] = overall_rmse


#############################################################################################################
#############################################################################################################
############## MAIN CODE below, ONLY for Forecast/Backcast Autoreg Upstream Models ##########################
#############################################################################################################

def objective():

    if not wandb.run:
        wandb.init(project="DownstreamBVDS", save_code=True)
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

    # Edit mode below:
    # train - trains the model and saves the best model in terms of validation loss
    # test - loads the model with the minimum loss inside Model directory
    mode = 'train'   # train or test

    # while loading be careful - current code looks at minimum loss model
    # UNLESS model_str designates which BVDS downstream model to use
    bvds_model_str = '0907-2312' # string representing bvds DOWNSTREAM model ran with specific architecture.

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
    window_size = 60 #placeholder, will load from autoreg model
    batch_size = 128
    test_pig_nums = [1, 2, 3, 4, 5, 6]   # for each pig we will a create a different model by excluding that pig
    # some parameters we might play with; Edit in hyperparameter section
    hidden_size = config.hidden_size
    hidden_layer = config.hidden_layer
    num_layers = config.num_layers
    forecast_size = config.forecast_size #placeholder, will load from autoreg model
    learning_rate = config.learning_rate
    weight_decay = config.weight_decay
    l1_lambda = config.l1_lambda
    dropout = config.dropout
    epochs = config.epochs
    overlap = config.overlap

    # string representing autoregressor model ran with specific architecture. 
    # For example '0725-0124' is the model ran on 07/25 at 1:24. leave blank if want to search.
    autoreg_model_str = config.autoreg_model_str 

    wandb.run.name = "bvds_" + autoreg_model_str
    wandb.run.save()

    # N * T * F
        # N: batch size
        # T: sequence length
        # F: 12 (num channels)
    #Y: N * 2 * M * F
        # 2 is for future/past prediction
        # M: length of forecast/


############################################################################################################
################# Section is for TRAINING a new downstream model with forecast/backcast ####################

    if mode == 'train':
        #model run name timestamp for easy access
        model_run_str = time.strftime("%m%d-%H%M")
        wandb.log({'bvds_model_str': model_run_str}) # save so wandb can name this model too

        for test_pig_num in test_pig_nums:
            print(f"Training for Pig {test_pig_num} Started")
            # where you save your model
            model_output_dir = os.path.join(project_dir, 'Models', 'DownstreamModel',
                                            folder_name_to_save, f'Pig{test_pig_num}-{training_mode}')
            if not os.path.exists(model_output_dir):
                os.makedirs(model_output_dir)
      
            checkpoint_callback = ModelCheckpoint(
                dirpath=model_output_dir,
                filename='model-' + model_run_str + '-{epoch:02d}-{val_loss:.2f}',
                monitor='val_loss',   # we want to save the model based on validation loss
                mode='min',   # we want to minimize validation loss
                save_top_k=1
            )

            wandb_logger = WandbLogger(log_model=True)

            # Finding which (good) LSTM upstream model to use, both forwards and backwards
            autoreg_model_dir = os.path.join(project_dir, 'Models', 'TimeSeriesModel',
                                            folder_name_to_save, f'Pig{test_pig_num}-{training_mode}')
            checkpoint_path_f = get_filename_with_min_val_loss(autoreg_model_dir, autoreg_model_str, 'f') #get best forward autoreg model
            checkpoint_path_b = get_filename_with_min_val_loss(autoreg_model_dir, autoreg_model_str, 'b') #get best backward autoreg model
            autoreg_model_f = RNNAutoregressor.load_from_checkpoint(checkpoint_path_f)
            autoreg_model_b = RNNAutoregressor.load_from_checkpoint(checkpoint_path_b)

            # Use both forward and backward models from LSTM
            # Set to eval mode, which only runs the encode step and returns hidden layer
            autoreg_model_f.eval()
            autoreg_model_b.eval()

            trainer = L.Trainer()

            # Create DataLoader that will be fed into upstream LSTM encoder to get intermediate output
            window_size = autoreg_model_f.window_size
            forecast_size = autoreg_model_f.forecast_size
            data_module = MyDataModule(data_dir, num_pigs, test_pig_num, training_mode,
                            all_hypovolemia_stages, train_hypovolemia_stages, test_hypovolemia_stages,
                            batch_size=batch_size, overlap_percentage=overlap, window_size=window_size, forecast_size=forecast_size)

            #create separate data modules for train/test/val pigs for lstm encoder
            data_module.set_prediction_mode('train')
            # Uses predict to run encoder in LSTM only and get intermediate hidden layers
            # Note: all of train, val, and test will be fed through the prediction, as they 
            # all must be used in the downstream as well for its own train/val/test
            encoded_predictions_train_f = trainer.predict(autoreg_model_f, data_module) #check seeding for train/val pairings!
            encoded_predictions_train_f = torch.cat(encoded_predictions_train_f, dim=0).numpy()
            encoded_predictions_train_b = trainer.predict(autoreg_model_b, data_module)
            encoded_predictions_train_b = torch.cat(encoded_predictions_train_b, dim=0).numpy()
            # Assigns labels to be the BVDS values
            bvds_labels_train = data_module.train.labels
            # Merges both the forwards, backwards outputs into one feature array
            encoded_predictions_train = np.concatenate((encoded_predictions_train_f, encoded_predictions_train_b), axis=2)

            # Repeat for validation set, which should be identical with the upstream set since using same seed
            data_module.set_prediction_mode('val')
            encoded_predictions_val_f = trainer.predict(autoreg_model_f, data_module)
            encoded_predictions_val_f = torch.cat(encoded_predictions_val_f, dim=0).numpy()
            encoded_predictions_val_b = trainer.predict(autoreg_model_b, data_module)
            encoded_predictions_val_b = torch.cat(encoded_predictions_val_b, dim=0).numpy()
            bvds_labels_val = data_module.validate.labels
            encoded_predictions_val = np.concatenate((encoded_predictions_val_f, encoded_predictions_val_b), axis=2)

            # Repeat for test set, which should be identical with the upstream set since using same seed
            data_module.set_prediction_mode('test')
            encoded_predictions_test_f = trainer.predict(autoreg_model_f, data_module)
            encoded_predictions_test_f = torch.cat(encoded_predictions_test_f, dim=0).numpy()
            encoded_predictions_test_b = trainer.predict(autoreg_model_b, data_module)
            encoded_predictions_test_b = torch.cat(encoded_predictions_test_b, dim=0).numpy()
            encoded_predictions_test = np.concatenate((encoded_predictions_test_f, encoded_predictions_test_b), axis=2)
            bvds_labels_test = data_module.test.labels
            bvds_hypo_stages_test = data_module.test.hypo_stages

            # Combine all these hidden layers, BVDS labels into DataLoader for downstream regressor
            encoded_data_module = DownstreamDataModule(
                encoded_predictions_train, bvds_labels_train, 
                encoded_predictions_val, bvds_labels_val,      
                encoded_predictions_test, bvds_labels_test, 
                bvds_hypo_stages_test, num_pigs, 
                test_pig_num, training_mode, 
                all_hypovolemia_stages, train_hypovolemia_stages, 
                test_hypovolemia_stages, batch_size=batch_size
                )

            # Initialize downstream regressor model
            model = BVDSRegressor(test_pig=test_pig_num, hidden_size=autoreg_model_f.encoder.hidden_size, learning_rate=learning_rate, 
                                  weight_decay=weight_decay, l1_lambda=l1_lambda, dropout=dropout, device_to_use=DEVICE,
                                    max_epochs=epochs, hidden_layer=hidden_layer, num_layers=num_layers)
            trainer = L.Trainer(logger=wandb_logger, callbacks=[MyProgressBar(), checkpoint_callback], max_epochs=epochs)
            # Train downstream regressor model
            trainer.fit(model, encoded_data_module)

        # After training, make sure bvds_model_str is the same as the upstream model_run_str for easy searching
        bvds_model_str = model_run_str

##########################################################################################################
########### Section is for TESTING for bvds downstream (will run automatically if training) ##############

    results = {}
    for test_pig_num in test_pig_nums:
        model_output_dir = os.path.join(project_dir, 'Models', 'TimeSeriesModel',
                                        folder_name_to_save, f'Pig{test_pig_num}-{training_mode}')
        
        print("######################################################################")
        print(f"Testing for Pig {test_pig_num} Started")
        print("######################################################################")
        
        checkpoint_callback = ModelCheckpoint(
            dirpath=model_output_dir,
            filename='model-{epoch:02d}-{val_loss:.2f}',
            monitor='val_loss',   # we want to save the model based on validation loss
            mode='min',   # we want to minimize validation loss
            save_top_k=1
        )

        wandb_logger = WandbLogger(log_model=True)

        # Find upstream autoregressor forecast/backcast models to predict and get hidden layer
        autoreg_model_dir = os.path.join(project_dir, 'Models', 'TimeSeriesModel',
                                        folder_name_to_save, f'Pig{test_pig_num}-{training_mode}')
        checkpoint_path_f = get_filename_with_min_val_loss(autoreg_model_dir, autoreg_model_str, 'f') #get best forward autoreg model
        checkpoint_path_b = get_filename_with_min_val_loss(autoreg_model_dir, autoreg_model_str, 'b') #get best backward autoreg model
        autoreg_model_f = RNNAutoregressor.load_from_checkpoint(checkpoint_path_f)
        autoreg_model_b = RNNAutoregressor.load_from_checkpoint(checkpoint_path_b)
        autoreg_model_f.eval()
        autoreg_model_b.eval()

        trainer = L.Trainer()
        # Create DataLoader that will be fed into upstream LSTM encoder to get intermediate output
        window_size = autoreg_model_f.window_size
        forecast_size = autoreg_model_f.forecast_size
        data_module = MyDataModule(data_dir, num_pigs, test_pig_num, training_mode,
                        all_hypovolemia_stages, train_hypovolemia_stages, test_hypovolemia_stages,
                        batch_size=batch_size, overlap_percentage=overlap, window_size=window_size, forecast_size=forecast_size)

        
        #create separate data modules for train/test/val pigs for lstm encoder. This then gets fed into downstream
        #while keeping train/val/test separate.
        data_module.set_prediction_mode('train')
        encoded_predictions_train_f = trainer.predict(autoreg_model_f, data_module) #check seeding for train/val pairings!
        encoded_predictions_train_f = torch.cat(encoded_predictions_train_f, dim=0).numpy()
        encoded_predictions_train_b = trainer.predict(autoreg_model_b, data_module)
        encoded_predictions_train_b = torch.cat(encoded_predictions_train_b, dim=0).numpy()
        bvds_labels_train = data_module.train.labels
        encoded_predictions_train = np.concatenate((encoded_predictions_train_f, encoded_predictions_train_b), axis=2)

        data_module.set_prediction_mode('val')
        encoded_predictions_val_f = trainer.predict(autoreg_model_f, data_module)
        encoded_predictions_val_f = torch.cat(encoded_predictions_val_f, dim=0).numpy()
        encoded_predictions_val_b = trainer.predict(autoreg_model_b, data_module)
        encoded_predictions_val_b = torch.cat(encoded_predictions_val_b, dim=0).numpy()
        bvds_labels_val = data_module.validate.labels
        encoded_predictions_val = np.concatenate((encoded_predictions_val_f, encoded_predictions_val_b), axis=2)

        data_module.set_prediction_mode('test')
        encoded_predictions_test_f = trainer.predict(autoreg_model_f, data_module)
        encoded_predictions_test_f = torch.cat(encoded_predictions_test_f, dim=0).numpy()
        encoded_predictions_test_b = trainer.predict(autoreg_model_b, data_module)
        encoded_predictions_test_b = torch.cat(encoded_predictions_test_b, dim=0).numpy()
        encoded_predictions_test = np.concatenate((encoded_predictions_test_f, encoded_predictions_test_b), axis=2)
        bvds_labels_test = data_module.test.labels
        bvds_hypo_stages_test = data_module.test.hypo_stages

        # Combine these hidden layers, BVDS labels into new DataLoader for downstream regressor model
        # Note: Only the test set will be used in this testing portion.
        encoded_data_module = DownstreamDataModule(
            encoded_predictions_train, bvds_labels_train, 
            encoded_predictions_val, bvds_labels_val,
            encoded_predictions_test, bvds_labels_test, 
            bvds_hypo_stages_test, num_pigs, test_pig_num, 
            training_mode, all_hypovolemia_stages, 
            train_hypovolemia_stages, test_hypovolemia_stages, 
            batch_size=batch_size
            )
        
        # encoded_data_module.setup()

        # Find which downstream BVDS regressor model to load and test
        bvds_model_dir = os.path.join(project_dir, 'Models', 'DownstreamModel',
                                        folder_name_to_save, f'Pig{test_pig_num}-{training_mode}')
        checkpoint_path = get_filename_with_min_val_loss(bvds_model_dir, bvds_model_str, '')
        model = BVDSRegressor.load_from_checkpoint(checkpoint_path)
        model.eval()
        trainer = L.Trainer()

        # Get predictions for BVDS
        predictions = trainer.predict(model, encoded_data_module)
        predictions = torch.cat(predictions, dim=0).numpy()

        # Consolidating overall results + classifying
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

        print(f"Pig {test_pig_num} results")
        report_results(gt_data, predictions, data_module.test_hypo_stages, results, test_pig_num)
        print()

    # now all median loss
    print("Global Results")
    for stage in test_hypovolemia_stages + ['overall']:
        print("Stage: ", stage)
        all_pig_results = []
        for pig_num in test_pig_nums:
            all_pig_results.append(results[pig_num][stage])
        print(f"Median RMSE error of {stage}: ", np.median(all_pig_results))
        result_dict = {stage+'_test': np.median(all_pig_results)}
        wandb.log(result_dict)

    wandb.finish()


##########################################################################################################
###################### Section is for SETTING HYPERPARAMETERS ############################################

if __name__ == "__main__":
    ###### Change to True to sweep of hyperparameters! #########
    perform_sweep = False #change to True if want to run sweep of parameters
    wandbproject = "DownstreamBVDS"
    hostname = socket.gethostname()

    sweep_configuration = {
        "method": "grid",
        "metric": {"goal": "minimize", "name": "val_loss"},
        "parameters": {
            "learning_rate": {"values": [0.001]},
            "weight_decay": {"values": [0.0005]},
            "l1_lambda": {"values": [0]},
            "hidden_size": {"values": [128]}, #SET TO WHAT WAS ON AUTOREGRESSOR MODEL! (Edit: will automatically)
            "forecast_size": {"values": [10]},
            "overlap": {"values": [0.9]},
            "epochs": {"values": [40]},
            "hidden_layer": {"values": [64]}, #This is for the downstream fully connected layers if num_layers=2
            "num_layers": {"values": [2]}, #This is for the downstream fully connected layers
            "dropout": {"values": [0]},
            # IMPORTANT BELOW: names of upstream TRAINED LSTM models to use
            "autoreg_model_str": {"values": [
                '0201-1727', 
                '0201-1631', 
                ]}
        },
    }

    if perform_sweep:
        sweep_id = wandb.sweep(sweep=sweep_configuration, project=wandbproject)
        wandb.agent(sweep_id, function=objective, count=2)
    else:
        # For Non-sweeps, edit hyperparameters for single runs
        wandb.init(project=wandbproject, config={
            "learning_rate": 0.001,
            "weight_decay": 0.0005,
            "l1_lambda": 0.00,
            "hidden_size": 128, #set to the same dimension as autoregressor!
            "forecast_size": 10, #placeholder, will load from autoreg model
            "overlap": 0.9,
            "epochs": 40,
            "hidden_layer": 64, #This is for the downstream fully connected layers if num_layers=2
            "num_layers": 2, #This is for the downstream fully connected layers
            "dropout": 0,
            # IMPORTANT BELOW: name of upstream TRAINED LSTM model to use
            "autoreg_model_str": '0616-0123',
        }, save_code=True)
        objective()




