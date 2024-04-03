import os
import sys
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from tqdm import trange
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, precision_recall_curve, f1_score
import shap
from captum.attr import IntegratedGradients
from captum.attr import visualization as viz

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torch import optim
import torch.nn.functional as F

from Wang_Chongrui_LSTM import *
from Wang_Chongrui_ALSTM import *
from Wang_Chongrui_TCN import *
from Wang_Chongrui_GATS import *
from Wang_Chongrui_SFM import *
from Wang_Chongrui_utils import *
from Wang_Chongrui_Loss_Function import *


BATCH_SIZE = 256
SEQ_LEN = 30
LEARNING_RATE = 1e-4
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-5
MAX_EPOCH = 100
TEST_SIZE = 0.2
RANDOM_SEED = 42
PATIENCE = 5


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# define early stopping
class EarlyStopping:
    def __init__(self, patience=7, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


def preprocess_tensor(df: pd.DataFrame):
    x = []
    for col in df.columns:
        x_i = df[col]
        x_i = x_i.unstack().values
        x.append(x_i)
    x = np.stack(x, axis=2)
    return x


class Data(Dataset):
    """
    The simple Dataset object from torch that can produce reshuffled batchs of data
    """
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.X[index], self.y[index]


class FeatureImportance:
    def __init__(self, model, data_loader):
        self.model = model
        self.data_loader = data_loader

    def plot_shap_values(self):
        # Convert DataLoader to a list of batches for SHAP
        test_x = [batch[0] for batch in self.data_loader]
        test_x = torch.cat(test_x, 0)
        
        # Creating a background dataset for SHAP
        background = test_x[:100]
        e = shap.DeepExplainer(self.model, background)
        shap_values = e.shap_values(test_x[:100])

        # Plotting SHAP values
        shap.summary_plot(shap_values, feature_names=self.data_loader.dataset.columns)

    def plot_integrated_gradients(self, target_class_idx):
        integrated_gradients = IntegratedGradients(self.model)
        data_iter = iter(self.data_loader)
        inputs, _ = next(data_iter)
        inputs.requires_grad = True

        attributions_ig = integrated_gradients.attribute(inputs, target=target_class_idx)
        
        # Visualizing the attributions
        for i in range(inputs.shape[0]):
            viz.visualize_image_attr_multiple(attributions_ig[i].cpu().detach().numpy(), 
                                              original_image=inputs[i].cpu().detach().numpy(), 
                                              methods=["original_image", "heat_map"],
                                              signs=["all", "positive"],
                                              titles=["Original Image", "Integrated Gradients"])
            plt.show()
    

def get_DL_data(train_df: pd.DataFrame, test_df: pd.DataFrame, ignore_cols: list, method: str):

    factor = pd.concat([train_df, test_df], axis=0)
    train_dates = np.sort(pd.unique(train_df['date']))
    test_dates = np.sort(pd.unique(test_df['date']))
    timeLst = np.concatenate([train_dates, test_dates])
    train_start_date, train_end_date = train_dates[0], train_dates[-1]
    test_start_date, test_end_date = test_dates[0], test_dates[-1]

    # get the dates for corresponding data sets
    train_start_idx = np.where(train_start_date <= timeLst)[0][0]
    train_end_idx = np.where(timeLst <= train_end_date)[0][-1]
    test_start_idx = np.where(test_start_date <= timeLst)[0][0]
    test_end_idx = np.where(timeLst <= test_end_date)[0][-1]
    train_dates = timeLst[train_start_idx:train_end_idx+1]
    test_dates = timeLst[test_start_idx-SEQ_LEN+1:test_end_idx+1]

    train_df = factor[factor['date'].isin(train_dates)].copy()
    test_df = factor[factor['date'].isin(test_dates)].copy()

    feature_cols = factor.columns.difference(ignore_cols)
    train_X = train_df[feature_cols]
    test_X = test_df[feature_cols]

    train_X, test_X = train_X.set_index(['ticker', 'date']), test_X.set_index(['ticker', 'date'])
    train_X, test_X = preprocess_tensor(train_X), preprocess_tensor(test_X) # [B, T, F]
    train_X, test_X = np.nan_to_num(train_X, nan=0), np.nan_to_num(test_X, nan=0)
    train_X, test_X = torch.from_numpy(train_X), torch.from_numpy(test_X)
    train_X, test_X = train_X.permute(1, 0, 2), test_X.permute(1, 0, 2) # [T, B, F]
    train_X, test_X = train_X.unfold(0, SEQ_LEN, 1).permute(1, 0, 3, 2), test_X.unfold(0, SEQ_LEN, 1).permute(1, 0, 3, 2) # [B, T-n, n, F]
    train_X, test_X = train_X.reshape((-1, train_X.size(2), train_X.size(3))), test_X.reshape((-1, test_X.size(2), test_X.size(3)))

    if method == 'classification':
        train_Y, test_Y = train_df[['ticker', 'date', 'label']], test_df[['ticker', 'date', 'label']]
    else:
        train_Y, test_Y = train_df[['ticker', 'date', 'return']], test_df[['ticker', 'date', 'return']]

    train_Y, test_Y = train_Y.set_index(['ticker', 'date']), test_Y.set_index(['ticker', 'date'])
    train_Y, test_Y = preprocess_tensor(train_Y), preprocess_tensor(test_Y)
    train_Y, test_Y = np.nan_to_num(train_Y, nan=0), np.nan_to_num(test_Y, nan=0)
    train_Y, test_Y = torch.from_numpy(train_Y), torch.from_numpy(test_Y)
    train_Y, test_Y = train_Y.permute(1, 0, 2), test_Y.permute(1, 0, 2) # [T, B, 1]
    train_Y, test_Y = train_Y.unfold(0, SEQ_LEN, 1).permute(1, 0, 2, 3), test_Y.unfold(0, SEQ_LEN, 1).permute(1, 0, 2, 3) # [B, T-n, 1, n]
    train_Y, test_Y = train_Y[:,:,:,-1].reshape(-1), test_Y[:,:,:,-1].reshape(-1)

    train_X, valid_X, train_Y, valid_Y = train_test_split(train_X, train_Y, test_size=TEST_SIZE, random_state=RANDOM_SEED)
    train_dataset, valid_dataset, test_dataset = Data(train_X, train_Y), Data(valid_X, valid_Y), Data(test_X, test_Y)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_X.size(-1), train_dataloader, valid_dataloader, test_dataloader


def initialize_model(model_name: str, loss_name: str, input_size: int, method: str):

    output_size = 1 if method == 'regression' else 2

    if model_name == 'LSTM':
        model = LSTMModel(input_size=input_size, output_size=output_size)
    elif model_name == 'ALSTM':
        model = ALSTMModel(input_size=input_size, output_size=output_size)
    elif model_name == 'TCN':
        model = TCNModel(num_input=SEQ_LEN, output_size=output_size, num_feature=input_size)
    elif model_name == 'GATS':
        model = GATModel(d_feat=input_size, output_size=output_size)
    elif model_name == 'SFM':
        model = SFMModel(d_feat=input_size, output_dim=output_size)
    else:
        raise ValueError(f'Parameter model_name should be LSTM/ALSTM/TCN/Transformer, get {model_name} instead')
    
    if loss_name == 'BCE':
        loss = BinaryCrossEntropyLoss()
    elif loss_name == 'Focal':
        loss = FocalLoss()
    elif loss_name == 'IC':
        loss = IC_loss()
    elif loss_name == 'WIC':
        loss = WeightedICLoss()
    elif loss_name == 'Sharpe':
        loss = SharpeLoss()
    else:
        raise ValueError(f'Parameter loss_name should be BCE/Focal, get {loss_name} instead')
    
    return model, loss


def get_DL_score(train_df, test_df, save_dir, model_name, loss_name, ignore_cols, method='classification'):
    if method not in ['classification', 'regression']:
        raise ValueError("method should be 'classification' or 'regression'")

    # create folder to store the unit data
    if not os.path.exists(f'{save_dir}/Model'):
        os.makedirs(f'{save_dir}/Model')

    if not os.path.exists(f'{save_dir}/Factor'):
        os.makedirs(f'{save_dir}/Factor')

    num_feature, train_dataloader, valid_dataloader, test_dataloader = get_DL_data(train_df, test_df, ignore_cols, method)

    model_dir = f'{save_dir}/Model/{model_name}_{loss_name}_{method}.m'
    if os.path.exists(model_dir):
        if method == 'classification':
            best_model_state = joblib.load(model_dir)
        else:
            best_model_state, best_threshold = joblib.load(model_dir)
        model, loss = initialize_model(model_name, loss_name, num_feature, method=method)
        model.load_state_dict(best_model_state)
    else:
        model, loss = initialize_model(model_name, loss_name, num_feature, method=method)
        model.to(device)
        if method == 'regression':
            best_model_state, best_threshold = train_regression(model, train_dataloader, loss, valid_dataloader)
            joblib.dump([best_model_state, best_threshold], model_dir)
        else:
            best_model_state = train(model, train_dataloader, loss, valid_dataloader)
            joblib.dump(best_model_state, model_dir)
        model.load_state_dict(best_model_state)

    # Assuming test_df is your initial DataFrame
    factor = test_df[['date', 'ticker', 'Open', 'High', 'Low', 'Close', 'Volume', 'return', 'label']].copy()
    # Set the index of factor to be a MultiIndex of ticker and date
    factor.set_index(['ticker', 'date'], inplace=True)
    # Create a complete MultiIndex of all combinations of unique dates and tickers
    all_dates = pd.unique(factor.index.get_level_values('date'))
    all_tickers = pd.unique(factor.index.get_level_values('ticker'))
    complete_index = pd.MultiIndex.from_product([all_tickers, all_dates], names=['ticker', 'date'])
    # Reindex the factor DataFrame with the complete index, introducing NaNs where data is missing
    factor = factor.reindex(complete_index)
    # Reset index to turn it back into columns
    factor.reset_index(inplace=True)
    # Sort the DataFrame based on 'ticker' and 'date'
    factor.sort_values(['ticker', 'date'], inplace=True)

    # Evaluate on test data
    model.to(device)
    model.eval()
    all_preds = []
    with torch.no_grad():
        for x_test, y_test in test_dataloader:
            x_test, y_test = x_test.float(), y_test.float()
            x_test, y_test = x_test.to(device), y_test.to(device)
            logits = model(x_test)

            if method == 'classification':
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)
            elif method == 'regression':
                preds = logits.squeeze()
                # Apply the threshold found during training
                preds = (preds >= best_threshold).int()

            all_preds.extend(preds.cpu().numpy())

    factor['pred'] = all_preds
    factor.dropna(subset=['return'], inplace=True)
    accuracy = accuracy_score(factor['label'].values, factor['pred'].values)
    precision = precision_score(factor['label'].values, factor['pred'].values, pos_label=1)
    print(f"Out of Sample Accuracy by {model_name} with {loss_name}: {accuracy}")
    print(f"Out of Sample Precision by {model_name} with {loss_name}: {precision}")

    factor.to_csv(f"{save_dir}/Factor/{model_name}_{loss_name}_{method}.csv", index=False)


def train(model, train_dataloader, criterion, valid_dataloader=None, MAX_EPOCH=MAX_EPOCH, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, patience=10):

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    
    print(f'{MAX_EPOCH} epochs to train: ')

    best_val_loss = math.inf
    best_model_state = None

    for epoch in range(1, MAX_EPOCH+1):

        Total_loss, Total_accuracy, Total_precision = 0, 0, 0
        print(f'epoch {epoch}/{MAX_EPOCH}:')
        train_data_size = len(train_dataloader)
        model.train()

        with trange(train_data_size) as train_bar:
            for i in train_bar:
                x_train, y_train = next(iter(train_dataloader))
                x_train, y_train = x_train.float(), y_train.long()
                x_train, y_train = x_train.to(device), y_train.to(device)

                optimizer.zero_grad()

                optimizer.zero_grad()
                y_pred = model(x_train)

                # Convert logits to probabilities and get the class with the higher probability
                probs = torch.softmax(y_pred, dim=1)
                preds = torch.argmax(probs, dim=1)

                # Convert y_train to one-hot encoding
                y_train_one_hot = F.one_hot(y_train, num_classes=2).float()

                train_loss = criterion(y_pred, y_train_one_hot)

                Total_loss += train_loss.item()

                train_loss.backward()
                torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=3)
                optimizer.step()

                # Calculate accuracy and precision
                y_train_np = y_train.detach().cpu().numpy()
                Total_accuracy += accuracy_score(y_train_np, preds.cpu().numpy())
                Total_precision += precision_score(y_train_np, preds.cpu().numpy(), zero_division=0)

                # Update progress bar
                avg_loss = Total_loss / (i + 1)
                avg_accuracy = Total_accuracy / (i + 1)
                avg_precision = Total_precision / (i + 1)
                train_bar.set_postfix(loss=avg_loss, accuracy=avg_accuracy, precision=avg_precision)

        if valid_dataloader is not None:
            model.eval()
            val_loss, val_acc, val_pre = evaluate(model, valid_dataloader, criterion)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict()
                print(f"New best model saved with validation loss: {best_val_loss}")
                print(f"New best model saved with validation accuracy: {val_acc}")
                print(f"New best model saved with validation precision: {val_pre}")

            early_stopping(val_loss)
            if early_stopping.early_stop:
                print("Early stopping")
                break

    return best_model_state


def evaluate(model, dataloader, criterion):

    print('Evaluating ...')
    test_data_size = len(dataloader)
    model.eval()

    Total_loss, Total_accuracy, Total_precision = 0, 0, 0

    with trange(test_data_size) as test_bar:
        for i in test_bar:
            x_test, y_test = next(iter(dataloader))
            x_test, y_test = x_test.float(), y_test.long()
            x_test, y_test = x_test.to(device), y_test.to(device)

            y_pred = model(x_test)

            # Convert y_train to one-hot encoding
            y_test_one_hot = F.one_hot(y_test, num_classes=2).float()

            Loss = criterion(y_pred, y_test_one_hot).item()

            # Convert logits to probabilities and get the class with the higher probability
            probs = torch.softmax(y_pred, dim=1)
            preds = torch.argmax(probs, dim=1)

            # Calculate accuracy and precision
            y_test_np = y_test.detach().cpu().numpy()
            Total_accuracy += accuracy_score(y_test_np, preds.cpu().numpy())
            Total_precision += precision_score(y_test_np, preds.cpu().numpy(), zero_division=0)

            Total_loss += Loss
            avg_loss = Total_loss / (i + 1)
            avg_accuracy = Total_accuracy / (i + 1)
            avg_precision = Total_precision / (i + 1)
            test_bar.set_postfix(loss=avg_loss, accuracy=avg_accuracy, precision=avg_precision)

        return Total_loss / test_data_size, Total_accuracy / test_data_size, Total_precision / test_data_size
    

# Train function specifically for regression
def train_regression(model, train_dataloader, criterion, valid_dataloader=None, MAX_EPOCH=MAX_EPOCH, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, patience=10):

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    
    print(f'{MAX_EPOCH} epochs to train: ')

    best_val_loss = math.inf
    best_model_state = None

    for epoch in range(1, MAX_EPOCH+1):

        Total_loss, Total_accuracy, Total_precision = 0, 0, 0
        print(f'epoch {epoch}/{MAX_EPOCH}:')
        train_data_size = len(train_dataloader)
        model.train()

        with trange(train_data_size) as train_bar:
            for i in train_bar:
                x_train, y_train = next(iter(train_dataloader))
                x_train, y_train = x_train.float(), y_train.float()
                x_train, y_train = x_train.to(device), y_train.to(device)

                optimizer.zero_grad()

                optimizer.zero_grad()
                y_pred = model(x_train)

                train_loss = criterion(y_pred, y_train)

                Total_loss += train_loss.item()

                train_loss.backward()
                torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=3)
                optimizer.step()

                # Update progress bar
                avg_loss = Total_loss / (i + 1)
                train_bar.set_postfix(loss=avg_loss)

        if valid_dataloader is not None:
            model.eval()
            val_loss = evaluate_regression(model, valid_dataloader, criterion)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict()
                print(f"New best model saved with validation loss: {best_val_loss}")

            early_stopping(val_loss)
            if early_stopping.early_stop:
                print("Early stopping")
                break

    # Determine best threshold on validation set
    if valid_dataloader is not None:
        best_threshold = find_best_threshold(model, valid_dataloader)
    else:
        best_threshold = find_best_threshold(model, train_dataloader)

    return best_model_state, best_threshold


def evaluate_regression(model, dataloader, criterion):

    print('Evaluating ...')
    test_data_size = len(dataloader)
    model.eval()

    Total_loss = 0

    with trange(test_data_size) as test_bar:
        for i in test_bar:
            x_test, y_test = next(iter(dataloader))
            x_test, y_test = x_test.float(), y_test.float()
            x_test, y_test = x_test.to(device), y_test.to(device)
            y_pred = model(x_test)
            Loss = criterion(y_pred, y_test).item()
            Total_loss += Loss
            avg_loss = Total_loss / (i + 1)
            test_bar.set_postfix(loss=avg_loss)

        return Total_loss / test_data_size


def find_best_threshold(model, dataloader):
    model.eval()
    all_targets = []
    all_predictions = []
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            x, y = x.float(), y.float()
            predictions = model(x).squeeze()

            # Check if predictions and targets are 2D, and flatten if they are
            if predictions.ndim == 2:
                predictions = predictions.flatten()
            if y.ndim == 2:
                y = y.flatten()

            all_predictions.extend(predictions.cpu().numpy())
            # Binarize the continuous targets
            binarized_targets = (y > 0).int().cpu().numpy()
            all_targets.extend(binarized_targets)

    _, _, thresholds = precision_recall_curve(all_targets, all_predictions)
    f1_scores = [f1_score(all_targets, all_predictions >= thresh) for thresh in thresholds]

    # Find the index of the highest F1 score
    best_thresh_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_thresh_idx]
    best_f1_score = f1_scores[best_thresh_idx]

    print(f"Best threshold: {best_threshold}, Best F1 score: {best_f1_score}")
    return best_threshold
    

def get_DL_score_ensemble(train_df: pd.DataFrame, test_df: pd.DataFrame, save_dir: str, model_name: str, loss_name: str, method: str, ignore_cols: list, n_models=5):

    # create folder to store the unit data
    if not os.path.exists(f'{save_dir}/Model'):
        os.makedirs(f'{save_dir}/Model')

    if not os.path.exists(f'{save_dir}/Factor'):
        os.makedirs(f'{save_dir}/Factor')

    num_feature, train_dataloader, valid_dataloader, test_dataloader = get_DL_data(train_df, test_df, ignore_cols)

    # Train multiple models
    models = []
    for i in range(n_models):
        model_dir = f'{save_dir}/Model/{model_name}_{loss_name}_{i}.m'
        if os.path.exists(model_dir):
            if method == 'classification':
                best_model_state = joblib.load(model_dir)
            else:
                best_model_state, best_threshold = joblib.load(model_dir)
            model, loss = initialize_model(model_name, loss_name, num_feature, method=method)
            model.load_state_dict(best_model_state)
        else:
            # Initialize and train model
            model, loss = initialize_model(model_name, loss_name, num_feature, method=method)
            model.to(device)
            if method == 'regression':
                best_model_state, best_threshold = train_regression(model, train_dataloader, loss, valid_dataloader)
            else:
                best_model_state = train(model, train_dataloader, loss, valid_dataloader)
            joblib.dump([best_model_state, best_threshold], model_dir)
            model.load_state_dict(best_model_state)
        if method == 'classification':
            models.append(model)
        else:
            models.append([model, best_threshold])

    # Assuming test_df is your initial DataFrame
    factor = test_df[['date', 'ticker', 'Open', 'High', 'Low', 'Close', 'Volume', 'return', 'label']].copy()
    # Set the index of factor to be a MultiIndex of ticker and date
    factor.set_index(['ticker', 'date'], inplace=True)
    # Create a complete MultiIndex of all combinations of unique dates and tickers
    all_dates = pd.unique(factor.index.get_level_values('date'))
    all_tickers = pd.unique(factor.index.get_level_values('ticker'))
    complete_index = pd.MultiIndex.from_product([all_tickers, all_dates], names=['ticker', 'date'])
    # Reindex the factor DataFrame with the complete index, introducing NaNs where data is missing
    factor = factor.reindex(complete_index)
    # Reset index to turn it back into columns
    factor.reset_index(inplace=True)
    # Sort the DataFrame based on 'ticker' and 'date'
    factor.sort_values(['ticker', 'date'], inplace=True)

    # Aggregate predictions from all models
    aggregated_preds = np.zeros(factor.shape[0])
    with torch.no_grad():
        if method == 'classification':
            for model in models:
                model.to(device)
                model.eval()
                batch_preds = []
                for x_test, y_test in test_dataloader:
                    x_test, y_test = x_test.float().to(device), y_test.float().to(device)
                    logits = model(x_test)
                    probs = torch.softmax(logits, dim=1)
                    preds = torch.argmax(probs, dim=1)
                    batch_preds.extend(preds.cpu().numpy())
                aggregated_preds += np.array(batch_preds)
        else:
            for model, threshold in models:
                model.to(device)
                model.eval()
                batch_preds = []
                for x_test, y_test in test_dataloader:
                    x_test, y_test = x_test.float().to(device), y_test.float().to(device)
                    logits = model(x_test)
                    preds = logits.squeeze()
                    # Apply the threshold found during training
                    preds = (preds >= threshold).int()
                    batch_preds.extend(preds.cpu().numpy())
                aggregated_preds += np.array(batch_preds)

    # Final predictions
    final_preds = (aggregated_preds / n_models) > 0.5
    factor['pred'] = final_preds
    factor.dropna(subset=['return'], inplace=True)

    accuracy = accuracy_score(factor['label'].values, factor['pred'].values)
    precision = precision_score(factor['label'].values, factor['pred'].valuess, pos_label=1)
    print(f"Out of Sample Accuracy by {model_name} with {loss_name}: {accuracy}")
    print(f"Out of Sample Precision by {model_name} with {loss_name}: {precision}")

    factor.to_csv(f"{save_dir}/Factor/{model_name}_{loss_name}_ensemble.csv", index=False)


def train_bagging(models, train_dataloader, criterion, valid_dataloader=None, MAX_EPOCH=100, lr=0.001, weight_decay=0, patience=10, num_models=5):

    trained_models = []

    for i in range(num_models):
        print(f"Training model {i+1}/{num_models}")
        model = models[i].to(device)
        # Create a subset of the data for this model
        subset_dataloader = create_subset_dataloader(train_dataloader, subset_size=0.8)
        trained_model = train(model, subset_dataloader, criterion, valid_dataloader, MAX_EPOCH, lr, weight_decay, patience)
        trained_models.append(trained_model)

    return trained_models


def create_subset_dataloader(dataloader, subset_size=0.8):
    # Extract the original dataset from the dataloader
    original_dataset = dataloader.dataset
    # Determine the size of the subset
    subset_length = int(len(original_dataset) * subset_size)
    # Create random indices for the subset
    subset_indices = np.random.choice(len(original_dataset), subset_length, replace=False)
    # Create a subset dataset
    subset_dataset = Subset(original_dataset, subset_indices)
    # Create a new dataloader from the subset dataset
    subset_dataloader = DataLoader(subset_dataset, batch_size=dataloader.batch_size, shuffle=True)

    return subset_dataloader


def get_DL_data_LTR(train_df: pd.DataFrame, test_df: pd.DataFrame, ignore_cols: list):

    factor = pd.concat([train_df, test_df], axis=0)
    train_dates = np.sort(pd.unique(train_df['date']))
    test_dates = np.sort(pd.unique(test_df['date']))
    timeLst = np.concatenate([train_dates, test_dates])
    train_start_date, train_end_date = train_dates[0], train_dates[-1]
    test_start_date, test_end_date = test_dates[0], test_dates[-1]

    # get the dates for corresponding data sets
    train_start_idx = np.where(train_start_date <= timeLst)[0][0]
    train_end_idx = np.where(timeLst <= train_end_date)[0][-1]
    test_start_idx = np.where(test_start_date <= timeLst)[0][0]
    test_end_idx = np.where(timeLst <= test_end_date)[0][-1]
    train_dates = timeLst[train_start_idx:train_end_idx+1]
    test_dates = timeLst[test_start_idx-SEQ_LEN+1:test_end_idx+1]

    train_df = factor[factor['date'].isin(train_dates)].copy()
    test_df = factor[factor['date'].isin(test_dates)].copy()

    feature_cols = factor.columns.difference(ignore_cols)
    train_X = train_df[feature_cols]
    test_X = test_df[feature_cols]

    train_X, test_X = train_X.set_index(['ticker', 'date']), test_X.set_index(['ticker', 'date'])
    train_X, test_X = preprocess_tensor(train_X), preprocess_tensor(test_X) # [B, T, F]
    train_X, test_X = np.nan_to_num(train_X, nan=0), np.nan_to_num(test_X, nan=0)
    train_X, test_X = torch.from_numpy(train_X), torch.from_numpy(test_X)
    train_X, test_X = train_X.permute(1, 0, 2), test_X.permute(1, 0, 2) # [T, B, F]
    train_X, test_X = train_X.unfold(0, SEQ_LEN, 1).permute(1, 0, 3, 2), test_X.unfold(0, SEQ_LEN, 1).permute(1, 0, 3, 2) # [B, T-n, n, F]

    train_Y, test_Y = train_df[['ticker', 'date', 'return']], test_df[['ticker', 'date', 'return']]
    train_Y, test_Y = train_Y.set_index(['ticker', 'date']), test_Y.set_index(['ticker', 'date'])
    train_Y, test_Y = preprocess_tensor(train_Y), preprocess_tensor(test_Y)
    train_Y, test_Y = np.nan_to_num(train_Y, nan=0), np.nan_to_num(test_Y, nan=0)
    train_Y, test_Y = torch.from_numpy(train_Y), torch.from_numpy(test_Y)
    train_Y, test_Y = train_Y.permute(1, 0, 2), test_Y.permute(1, 0, 2) # [T, B, 1]
    train_Y, test_Y = train_Y.unfold(0, SEQ_LEN, 1).permute(1, 0, 2, 3), test_Y.unfold(0, SEQ_LEN, 1).permute(1, 0, 2, 3) # [B, T-n, 1, n]
    train_Y, test_Y = train_Y[:,:,:,-1].reshape(train_Y.size(0), train_Y.size(1)), test_Y[:,:,:,-1].reshape(test_Y.size(0), test_Y.size(1))
    train_Y = np.log(train_Y + 1)

    train_X, test_X = train_X.permute(1, 0, 2, 3), test_X.permute(1, 0, 2, 3)
    train_Y, test_Y = train_Y.permute(1, 0), test_Y.permute(1, 0)

    train_X, valid_X, train_Y, valid_Y = train_test_split(train_X, train_Y, test_size=TEST_SIZE, random_state=RANDOM_SEED)
    train_dataset, valid_dataset, test_dataset = Data(train_X, train_Y), Data(valid_X, valid_Y), Data(test_X, test_Y)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_X.size(-1), train_dataloader, valid_dataloader, test_dataloader


def initialize_model_LTR(model_name: str, loss_name: str, input_size: int, method: str):

    if model_name == 'LSTM':
        model = LSTMRank(input_size=input_size)
    elif model_name == 'ALSTM':
        model = ALSTMRank(input_size=input_size)
    elif model_name == 'TCN':
        model = TCNRank(num_input=SEQ_LEN, num_feature=input_size)
    elif model_name == 'GATS':
        model = GATRank(d_feat=input_size)
    elif model_name == 'SFM':
        model = SFMRank(d_feat=input_size)
    else:
        raise ValueError(f'Parameter model_name should be LSTM/ALSTM/TCN/Transformer, get {model_name} instead')
    
    if loss_name == 'ListMLE':
        loss = ListMLE()
    elif loss_name == 'ListNet':
        loss = ListNet()
    else:
        raise ValueError(f'Parameter loss_name should be BCE/Focal, get {loss_name} instead')
    
    return model, loss


def get_DL_score_LTR(train_df, test_df, save_dir, model_name, loss_name, ignore_cols, method='regression'):
    
    if method != 'regression':
        raise ValueError("method should be 'regression' for Learning to Rank loss function")

    # create folder to store the unit data
    if not os.path.exists(f'{save_dir}/Model'):
        os.makedirs(f'{save_dir}/Model')

    if not os.path.exists(f'{save_dir}/Factor'):
        os.makedirs(f'{save_dir}/Factor')

    num_feature, train_dataloader, valid_dataloader, test_dataloader = get_DL_data_LTR(train_df, test_df, ignore_cols)

    model_dir = f'{save_dir}/Model/{model_name}_{loss_name}_{method}.m'
    if os.path.exists(model_dir):
        best_model_state, best_threshold = joblib.load(model_dir)
        model, loss = initialize_model_LTR(model_name, loss_name, num_feature, method=method)
        model.load_state_dict(best_model_state)
    else:
        model, loss = initialize_model_LTR(model_name, loss_name, num_feature, method=method)
        model.to(device)
        best_model_state, best_threshold = train_regression(model, train_dataloader, loss, valid_dataloader)
        joblib.dump([best_model_state, best_threshold], model_dir)
        model.load_state_dict(best_model_state)

    # Assuming test_df is your initial DataFrame
    factor = test_df[['date', 'ticker', 'Open', 'High', 'Low', 'Close', 'Volume', 'return', 'label']].copy()
    # Set the index of factor to be a MultiIndex of ticker and date
    factor.set_index(['ticker', 'date'], inplace=True)
    # Create a complete MultiIndex of all combinations of unique dates and tickers
    all_dates = pd.unique(factor.index.get_level_values('date'))
    all_tickers = pd.unique(factor.index.get_level_values('ticker'))
    complete_index = pd.MultiIndex.from_product([all_tickers, all_dates], names=['ticker', 'date'])
    # Reindex the factor DataFrame with the complete index, introducing NaNs where data is missing
    factor = factor.reindex(complete_index)
    # Reset index to turn it back into columns
    factor.reset_index(inplace=True)
    # Sort the DataFrame based on 'ticker' and 'date'
    factor.sort_values(['ticker', 'date'], inplace=True)

    # Evaluate on test data
    model.to(device)
    model.eval()
    all_preds = []
    with torch.no_grad():
        for x_test, y_test in test_dataloader:
            x_test, y_test = x_test.float(), y_test.float()
            x_test, y_test = x_test.to(device), y_test.to(device)
            logits = model(x_test)
            preds = logits.squeeze()
            # Apply the threshold found during training
            preds = (preds >= best_threshold).int()
            preds = preds.permute(1,0).reshape(-1)
            all_preds.extend(preds.cpu().numpy())

    factor['pred'] = all_preds
    factor.dropna(subset=['return'], inplace=True)
    accuracy = accuracy_score(factor['label'].values, factor['pred'].values)
    precision = precision_score(factor['label'].values, factor['pred'].values, pos_label=1)
    print(f"Out of Sample Accuracy by {model_name} with {loss_name}: {accuracy}")
    print(f"Out of Sample Precision by {model_name} with {loss_name}: {precision}")

    factor.to_csv(f"{save_dir}/Factor/{model_name}_{loss_name}_{method}.csv", index=False)
    

# if __name__ == "__main__":

#     save_dir = '.'
#     model_name = 'LSTM'
#     # model_name = 'ALSTM'
#     # model_name = 'TCN'
#     # model_name = 'SFM'
#     # model_name = 'GATS'
#     loss_name = 'BCE'
#     # loss_name = 'Focal'
#     # loss_name = 'IC'
#     # loss_name = 'WIC'
#     # loss_name = 'Sharpe'
#     # loss_name = 'ListMLE'
#     # loss_name = 'ListNet'
#     method = 'classification'
#     # method = 'regression'
#     ignore_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits', 'return', 'label']
#     n_models = 5

#     # get_DL_score(train_df, test_df, save_dir, model_name, loss_name, ignore_cols, method)
#     # get_DL_score_ensemble(train_df, test_df, save_dir, model_name, loss_name, ignore_cols, n_models)
#     # get_DL_score_LTR(train_df, test_df, save_dir, model_name, loss_name, ignore_cols, method)

#     stock = pd.read_csv("alphas_processed.csv")
#     train_df, test_df = split_dataset(stock, train_ratio=0.6)

#     # modelLst = ['LSTM', 'ALSTM', 'TCN', 'SFM', 'GATS']

#     modelLst = ['LSTM']
#     lossLst = ['BCE', 'Focal']
#     method = 'classification'

#     for model_name in modelLst:
#         for loss_name in lossLst:
#             get_DL_score(train_df, test_df, save_dir, model_name, loss_name, ignore_cols, method)

#     modelLst = ['LSTM']
#     lossLst = ['IC', 'WIC', 'Sharpe']
#     method = 'regression'

#     for model_name in modelLst:
#         for loss_name in lossLst:
#             get_DL_score(train_df, test_df, save_dir, model_name, loss_name, ignore_cols, method)

#     modelLst = ['LSTM']
#     lossLst = ['ListMLE', 'ListNet']
#     method = 'regression'

#     for model_name in modelLst:
#         for loss_name in lossLst:
#             get_DL_score_LTR(train_df, test_df, save_dir, model_name, loss_name, ignore_cols, method)

#     # modelLst = ['LSTM']
#     # lossLst = ['BCE']
#     # method = 'classification'

#     # for model_name in modelLst:
#     #     for loss_name in lossLst:
#     #         get_DL_score(train_df, test_df, save_dir, model_name, loss_name, ignore_cols, method)