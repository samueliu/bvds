import os
import pandas as pd
import numpy as np
import torch.optim
import torch.nn as nn
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import lightning as L
from sklearn.metrics import mean_squared_error
from torch.utils.data import Dataset
import random
import sys
import wandb
import time
import socket
from SamuelCode import RNNAutoregressor
from pytorch_lightning.loggers import WandbLogger

# Set random seed for reproducibility
random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

feature_names = ["hr", "hrvDifference", "hrvPoincare", "hrvSpectral",
     "scgPEP", "scgLVET", "scgPEPOverLVET", "scgPAT", "scgPTT",
     "ppgIHAT", "femoralPPGPPV", "ppgAmplitudeDistal"]
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
    def __init__(self, test_pig, hidden_size, learning_rate, weight_decay, l1_lambda, dropout, device_to_use, max_epochs, hidden_layer, num_layers):
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
        
        if self.num_layers == 2:
            self.bvds_regressor = nn.Sequential(
                nn.Linear(hidden_size*2, self.hidden_layer), 
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(self.hidden_layer, 1),
            )
        else:
            self.bvds_regressor = nn.Sequential(
                nn.Linear(hidden_size*2, 1),
                nn.ReLU(),
                nn.Dropout(dropout),
            )

        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        # decode output into bvds
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
        x, y = batch # x=(N,T,F) y=(N,2,M,F)
        x, y = x.to(self.device), y.to(self.device)
        x_hat = self(x) #runs through fc layers
        loss = self.loss_fn(x_hat, y) 
        l1_loss = sum(torch.sum(torch.abs(param)) for param in self.parameters())
        loss += self.l1_lambda * l1_loss
        self.log(f'val_loss_pig_{self.test_pig}', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def predict_step(self, batch, batch_idx):
        x, y = batch
        x = x.to(self.device_to_use)
        x_hat = self(x)
        return x_hat

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay) #adjust learning rate here
        return optimizer

class MyDataModule(L.LightningDataModule):
    def __init__(self, data_directory, num_pigs, test_pig_num, bvds_mode,
                 all_hypovolemia_stages, train_hypovolemia_stages, test_hypovolemia_stages,
                 batch_size=64, overlap_percentage=0.5, window_size = 30, forecast_size=15):
        super().__init__()
        # train/test_hypovolemia_stages = ['Absolute', 'Relative', 'Resuscitation']
        # these are here in case you want to train/test your model using certain stages
        self.data_directory = data_directory
        self.bvds_mode = 'classes' if bvds_mode=='classification' else 'labels'   # bvds_mode = 'classification' or 'regression' unchanged from original
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

    def prepare_data(self):
        pass

    def set_prediction_mode(self, mode):
        self.prediction_mode = mode

    def setup(self, stage=None):
        self.data_dict = {}
        # data dict keys will be pig numbers\
        # each data_dict[pig_num] will be also a dictionary with key Absolute, Relative, Resuscitation
        # each of these is also a dictionary with 2 keys 'features' and 'labels'
        for pig_idx in range(1, self.num_pigs + 1):  # traverse pigs
            self.data_dict[pig_idx] = {}
            for hypovolemia_stage in self.all_hypovolemia_stages:
                self.data_dict[pig_idx][hypovolemia_stage] = {}
                data_path = os.path.join(self.data_directory, hypovolemia_stage)
                features = pd.read_csv(os.path.join(data_path, f"pig_{pig_idx}_features.csv"))[feature_names].values
                labels = pd.read_csv(os.path.join(data_path, f"pig_{pig_idx}_{self.bvds_mode}.csv"))[self.label_name].values 
                self.data_dict[pig_idx][hypovolemia_stage]['features'] = features
                self.data_dict[pig_idx][hypovolemia_stage]['labels'] = labels
        # now split
        self.prepare_train_val_test()
        # now save if enabled
        # save if enabled

    def prepare_train_val_test(self):
        # train dataset first
        pigs_for_training = [m+1 for m in range(self.num_pigs) if m+1 != self.test_pig_num]
        print("pigs for training: ", pigs_for_training)
        print("pig for testing: ", self.test_pig_num)
        train_X = np.empty((0, self.window_size, len(feature_names)))
        train_y = np.empty((0, 1))
        val_X = np.empty((0, self.window_size, len(feature_names)))
        val_y = np.empty((0, 1))
        train_pairs, val_pairs = get_train_val_pairs(pigs_for_training, self.train_hypovolemia_stages)

        # training
        for pair in train_pairs:
            (pig_idx, hypovolemia_stage) = pair
            features = self.data_dict[pig_idx][hypovolemia_stage]['features']
            labels = self.data_dict[pig_idx][hypovolemia_stage]['labels']
            ts_features, ts_labels = create_timeseries_dataset_with_labels(features, labels,
                                                                           time_steps=self.window_size,
                                                                           overlap_percentage=self.overlap_percentage)
            train_X = np.vstack((train_X, ts_features))
            train_y = np.vstack((train_y, ts_labels.reshape(-1, 1)))
        # validation
        for pair in val_pairs:
            (pig_idx, hypovolemia_stage) = pair
            features = self.data_dict[pig_idx][hypovolemia_stage]['features']
            labels = self.data_dict[pig_idx][hypovolemia_stage]['labels']
            ts_features, ts_labels = create_timeseries_dataset_with_labels(features, labels,
                                                                           time_steps=self.window_size,
                                                                           overlap_percentage=self.overlap_percentage)
            val_X = np.vstack((val_X, ts_features))
            val_y = np.vstack((val_y, ts_labels.reshape(-1, 1)))

        # for the test pig
        test_X = np.empty((0, self.window_size, len(feature_names)))
        test_y = np.empty((0, 1))
        self.test_hypo_stages = []
        for hypovolemia_stage in self.test_hypovolemia_stages:
            features = self.data_dict[self.test_pig_num][hypovolemia_stage]['features']
            labels = self.data_dict[self.test_pig_num][hypovolemia_stage]['labels']
            ts_features, ts_labels = create_timeseries_dataset_with_labels(features, labels,
                                                                           time_steps=self.window_size,
                                                                           overlap_percentage=self.overlap_percentage)
            self.test_hypo_stages = self.test_hypo_stages + [hypovolemia_stage for _ in range(len(ts_features))]

            test_X = np.vstack((test_X, ts_features))
            test_y = np.vstack((test_y, ts_labels.reshape(-1, 1)))
        self.test_hypo_stages = np.array(self.test_hypo_stages)
        print("Train x shape: ", train_X.shape)
        print("Train y shape: ", train_y.shape)
        print("Val x shape: ", val_X.shape)
        print("Val y shape: ", val_y.shape)
        print("Test x shape: ", test_X.shape)
        print("Test y shape: ", test_y.shape)
        print("Test hypo stages length: ", self.test_hypo_stages.shape)

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
            return DataLoader(self.train, batch_size=self.batch_size, shuffle=False)
        elif self.prediction_mode == 'val':
            return DataLoader(self.validate, batch_size=self.batch_size, shuffle=False)
        elif self.prediction_mode == 'test':
            return DataLoader(self.test, batch_size=self.batch_size, shuffle=False)

class DownstreamDataModule(L.LightningDataModule):
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
        # now save if enabled
        # save if enabled

    def prepare_train_val_test(self):
        # train dataset first
        pigs_for_training = [m+1 for m in range(self.num_pigs) if m+1 != self.test_pig_num]
        print("pigs for training: ", pigs_for_training)
        print("pig for testing: ", self.test_pig_num)
        train_X = self.encoded_predictions_train.squeeze()
        train_X = train_X.reshape(train_X.shape[0], train_X.shape[-1]*2)
        train_y = self.bvds_labels_train
        val_X = self.encoded_predictions_val.squeeze()
        val_X = val_X.reshape(val_X.shape[0], val_X.shape[-1]*2)
        val_y = self.bvds_labels_val
        test_X = self.encoded_predictions_test.squeeze()
        test_X = test_X.reshape(test_X.shape[0], test_X.shape[-1]*2)
        test_y = self.bvds_labels_test

        print("Train x shape: ", train_X.shape)
        print("Train y shape: ", train_y.shape)
        print("Val x shape: ", val_X.shape)
        print("Val y shape: ", val_y.shape)
        print("Test x shape: ", test_X.shape)
        print("Test y shape: ", test_y.shape)

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
    # Shuffle each list
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
    pattern = r'model-epoch=(\d+)-val_loss=([\d.]+)'
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

def report_results(gt, predictions, hypovolemia_stages, results, subj_idx):
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

def objective():

    if not wandb.run:
        wandb.init(project="DownstreamBVDS", save_code=True)
    wandb.init(settings=wandb.Settings(code_dir="."))
    config = wandb.config

    hostname = socket.gethostname()
    if hostname == 'samue':
        project_dir = r"C:\\Users\\samue\\OneDrive - Georgia Institute of Technology\\GT Files\\ECE 8903 I02\\SamuelCode\\HypovolemiaSamuel"
    elif hostname == 'jarvis':
        project_dir = "/home/sliu/Desktop/SamuelCode/bvds/HypovolemiaSamuel"
    else:
        raise ValueError("Unknown environment")

    # this is the location of the data
    data_dir = os.path.join(project_dir, 'Data')

    mode = 'test'   # train or test
    # train - trains the model and saves the best model in terms of validation loss
    # test - loads the model with the minimum loss inside Model directory
    # while loading be careful - current code looks at minimum loss model which might belong to
    # an earlier version of the model (with different layers, parameters etc.) - this might be updated

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
    window_size = 60
    batch_size = 128
    test_pig_nums = [1, 2, 3, 4, 5, 6]   # for each pig we will a create a different model by excluding that pig
    # some parameters we might play with
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

    if mode == 'train':
        for test_pig_num in test_pig_nums:
            print(f"Training for Pig {test_pig_num} Started")
            # where you save your model
            model_output_dir = os.path.join(project_dir, 'Models', 'DownstreamModel',
                                            folder_name_to_save, f'Pig{test_pig_num}-{training_mode}')
            if not os.path.exists(model_output_dir):
                os.makedirs(model_output_dir)
      
            data_module = MyDataModule(data_dir, num_pigs, test_pig_num, training_mode,
                                all_hypovolemia_stages, train_hypovolemia_stages, test_hypovolemia_stages,
                                batch_size=batch_size, overlap_percentage=overlap, window_size=window_size, forecast_size=forecast_size)

            checkpoint_callback = ModelCheckpoint(
                dirpath=model_output_dir,
                filename='model-{epoch:02d}-{val_loss:.2f}',
                monitor='val_loss',   # we want to save the model based on validation loss
                mode='min',   # we want to minimize validation loss
                save_top_k=1
            )

            wandb_logger = WandbLogger(log_model=True)

            autoreg_model_dir = os.path.join(project_dir, 'Models', 'TimeSeriesModel',
                                            folder_name_to_save, f'Pig{test_pig_num}-{training_mode}')
            checkpoint_path = get_filename_with_min_val_loss(autoreg_model_dir)
            autoreg_model = RNNAutoregressor.load_from_checkpoint(checkpoint_path)

            autoreg_model.eval()
            trainer = L.Trainer()

            #create separate data modules for train/test/val pigs for lstm encoder
            data_module.set_prediction_mode('train')
            encoded_predictions_train = trainer.predict(autoreg_model, data_module)
            encoded_predictions_train = torch.cat(encoded_predictions_train, dim=0).numpy()
            bvds_labels_train = data_module.train.labels

            data_module.set_prediction_mode('val')
            encoded_predictions_val = trainer.predict(autoreg_model, data_module)
            encoded_predictions_val = torch.cat(encoded_predictions_val, dim=0).numpy()
            bvds_labels_val = data_module.validate.labels

            data_module.set_prediction_mode('test')
            encoded_predictions_test = trainer.predict(autoreg_model, data_module)
            encoded_predictions_test = torch.cat(encoded_predictions_test, dim=0).numpy()
            bvds_labels_test = data_module.test.labels
            bvds_hypo_stages_test = data_module.test.hypo_stages

            encoded_data_module = DownstreamDataModule(encoded_predictions_train, bvds_labels_train, encoded_predictions_val, bvds_labels_val,
                                                        encoded_predictions_test, bvds_labels_test, bvds_hypo_stages_test, num_pigs, test_pig_num, training_mode, 
                                                        all_hypovolemia_stages, train_hypovolemia_stages, test_hypovolemia_stages, batch_size=batch_size)
            
            # encoded_data_module.setup()

            model = BVDSRegressor(test_pig=test_pig_num, hidden_size=hidden_size, learning_rate=learning_rate, 
                                  weight_decay=weight_decay, l1_lambda=l1_lambda, dropout=dropout, device_to_use=DEVICE,
                                    max_epochs=epochs, hidden_layer=hidden_layer, num_layers=num_layers)
            trainer = L.Trainer(logger=wandb_logger, callbacks=[MyProgressBar(), checkpoint_callback], max_epochs=epochs)

            trainer.fit(model, encoded_data_module)

    else:  # testing assuming that you have a trained model
        results = {}
        for test_pig_num in test_pig_nums:
            model_output_dir = os.path.join(project_dir, 'Models', 'TimeSeriesModel',
                                            folder_name_to_save, f'Pig{test_pig_num}-{training_mode}')
            print(f"Testing for Pig {test_pig_num} Started")
            checkpoint_path = get_filename_with_min_val_loss(model_output_dir)
            print("Called model checkpoint path: ", checkpoint_path)
                    
            data_module = MyDataModule(data_dir, num_pigs, test_pig_num, training_mode,
                                all_hypovolemia_stages, train_hypovolemia_stages, test_hypovolemia_stages,
                                batch_size=batch_size, overlap_percentage=overlap, window_size=window_size, forecast_size=forecast_size)

            checkpoint_callback = ModelCheckpoint(
                dirpath=model_output_dir,
                filename='model-{epoch:02d}-{val_loss:.2f}',
                monitor='val_loss',   # we want to save the model based on validation loss
                mode='min',   # we want to minimize validation loss
                save_top_k=1
            )

            wandb_logger = WandbLogger(log_model=True)

            autoreg_model_dir = os.path.join(project_dir, 'Models', 'TimeSeriesModel',
                                            folder_name_to_save, f'Pig{test_pig_num}-{training_mode}')
            checkpoint_path = get_filename_with_min_val_loss(autoreg_model_dir)
            autoreg_model = RNNAutoregressor.load_from_checkpoint(checkpoint_path)

            autoreg_model.eval()
            trainer = L.Trainer()

            #create separate data modules for train/test/val pigs for lstm encoder
            data_module.set_prediction_mode('train')
            encoded_predictions_train = trainer.predict(autoreg_model, data_module)
            encoded_predictions_train = torch.cat(encoded_predictions_train, dim=0).numpy()
            bvds_labels_train = data_module.train.labels

            data_module.set_prediction_mode('val')
            encoded_predictions_val = trainer.predict(autoreg_model, data_module)
            encoded_predictions_val = torch.cat(encoded_predictions_val, dim=0).numpy()
            bvds_labels_val = data_module.validate.labels

            data_module.set_prediction_mode('test')
            encoded_predictions_test = trainer.predict(autoreg_model, data_module)
            encoded_predictions_test = torch.cat(encoded_predictions_test, dim=0).numpy()
            bvds_labels_test = data_module.test.labels
            bvds_hypo_stages_test = data_module.test.hypo_stages

            encoded_data_module = DownstreamDataModule(encoded_predictions_train, bvds_labels_train, encoded_predictions_val, bvds_labels_val,
                                                        encoded_predictions_test, bvds_labels_test, bvds_hypo_stages_test, num_pigs, test_pig_num, training_mode, 
                                                        all_hypovolemia_stages, train_hypovolemia_stages, test_hypovolemia_stages, batch_size=batch_size)
            
            # encoded_data_module.setup()

            bvds_model_dir = os.path.join(project_dir, 'Models', 'DownstreamModel',
                                            folder_name_to_save, f'Pig{test_pig_num}-{training_mode}')
            checkpoint_path = get_filename_with_min_val_loss(bvds_model_dir)
            model = BVDSRegressor.load_from_checkpoint(checkpoint_path)
            model.eval()
            trainer = L.Trainer()

            predictions = trainer.predict(model, encoded_data_module)
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

    wandb.finish()

if __name__ == "__main__":
    # Define the search space
    sweep_configuration = {
        "method": "random",
        "metric": {"goal": "minimize", "name": "val_loss"},
        "parameters": {
            "learning_rate": {"values": [0.0007, 0.0001, .0004]},
            "weight_decay": {"values": [0.0001, 0, .0005]},
            "l1_lambda": {"values": [0]},
            "hidden_size": {"values": [256, 128]},
            "forecast_size": {"values": [10]},
            "overlap": {"values": [0.9]},
            "epochs": {"values": [40]},
            "hidden_layer": {"values": [0, 64]},
            "num_layers": {"values": [1, 2]},
            "dropout": {"values": [.1, .25]},
        },
    }
    perform_sweep = True #change to True if want to run sweep of parameters

    hostname = socket.gethostname()
    wandbproject = "DownstreamBVDS"

    if perform_sweep:
        sweep_id = wandb.sweep(sweep=sweep_configuration, project=wandbproject)
        wandb.agent(sweep_id, function=objective, count=10)
    else:
        wandb.init(project=wandbproject, config={
            "learning_rate": 0.0004,
            "weight_decay": 0.0005,
            "l1_lambda": 0.00,
            "hidden_size": 128,
            "forecast_size": 20,
            "overlap": 0.9,
            "epochs": 40,
            "hidden_layer": 64,
            "num_layers": 2,
            "dropout": 0.25,
        }, save_code=True)
        objective()




