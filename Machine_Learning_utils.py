import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, f1_score
import shap
from hyperopt import fmin, tpe, hp
from hyperopt.pyll.base import scope
from sklearn.linear_model import RidgeClassifier, Ridge
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor, RandomForestClassifier, RandomForestRegressor, VotingClassifier, AdaBoostClassifier, AdaBoostRegressor, BaggingClassifier, VotingRegressor, BaggingRegressor
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor


class Model(object):
    """
    Model used for classification task
    Currently updated models:
    1. Random Forest
    2. Adaboost
    3. XGboost
    4. Catboost
    5. Light Gradient Boosting Machine
    6. Ridge
    """
    def __init__(self, model):
        self.model = model
        self.name = None

    # set parameters for the model
    def set_params(self, **params):
        self.model.set_params(**params)

    # fit the model on the data
    def fit(self, train_X, train_Y):
        return self.model.fit(train_X, train_Y)

    # get the underlying scikit-learn model
    def get_sklearn_model(self):
        return self.model
    
    # do cross validation to see the performance
    def cross_validation(self, train_X, train_Y, cv=5, scoring="neg_mean_squared_error", verbose=False):
        """
        Cross validation

        Args:
            cv: the number of splits for the cross validation
            scoring: the scroing method of the cross validation
                     for regression: "neg_mean_absolute_error", "neg_mean_squared_error", "r2", ...
        """
        score = cross_val_score(self.model, train_X, train_Y, cv=cv, scoring=scoring).mean()
        if verbose:
            print("The {} of cross validation is {}".format(scoring, score))
        return score
    
    # do time series cross validation to see the performance
    def cross_validation_ts(self, train_X, train_Y, n_splits=5, scoring="neg_mean_squared_error", verbose=False):
        """
        Time series cross validation

        Args:
            n_splits: the number of splits for the cross validation
            scoring: the scoring method of the cross validation
                     for regression: "neg_mean_absolute_error", "neg_mean_squared_error", "r2", ...
                     for classification: "accuracy", "f1", ...
        """
        tscv = TimeSeriesSplit(n_splits=n_splits)
        score = cross_val_score(self.model, train_X, train_Y, cv=tscv, scoring=scoring).mean()
        if verbose:
            print("The average {} score across all time splits is {}".format(scoring, score))
        return score

    def hyperopt(self, train_X, train_Y, uniform_dict, int_dict, choice_dict, maximum=True, max_evals=10, cv=5,
                 scoring="neg_mean_squared_error"):
        """
        hyperparameter optimization

        Args:
            uniform_dict: the dictionary contains the hyperparameters in float form
            int_dict: the dictionary contains the hyperparameters in int form
            choice_dict: the dictionary contains the hyperparameters in other discrete form
        """
        space, int_key, choice_key = {}, [], []
        # define the type of the hyperparameters
        for key, value in uniform_dict.items():
            space.update({key:hp.uniform(key,value[0],value[1])})
        for key, value in int_dict.items():
            space.update({key:scope.int(hp.uniform(key,value[0],value[1]))})
            int_key.append(key)
        for key, value in choice_dict.items():
            space.update({key:hp.choice(key,value)})
            choice_key.append((key,value))

        # define the loss function
        def loss(params):
            self.model.set_params(**params)
            if maximum:
                return -self.cross_validation(train_X, train_Y, cv=cv, scoring=scoring)
            else:
                return self.cross_validation(train_X, train_Y, cv=cv, scoring=scoring)

        # process for hyperparameter pruning
        optparams = fmin(fn=loss, space=space, algo=tpe.suggest, max_evals=max_evals)
        for key in int_key:
            optparams[key] = int(optparams[key])
        for item in choice_key:
            optparams.update({item[0]:item[1][optparams[item[0]]]})
        # set the best hyperparameters to the model
        self.model.set_params(**optparams)
        print("The optimal parameters of model {} in terms of {} is {}".format(self.name, scoring, optparams))


    def plot_feature_importance(self, feature_names, top_n=-1):
        """
        Plot feature importance for tree-based models.

        Args:
        feature_names (list): List of feature names.
        top_n (int): Number of top features to display. If -1, display all features.
        """
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            indices = np.argsort(importances)[::-1]

            # Select top_n features
            if top_n != -1:
                indices = indices[:top_n]

            plt.figure(figsize=(10, 6))
            plt.title("Feature importances")
            plt.bar(range(len(indices)), importances[indices],
                    color="r", align="center")
            plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=45)
            plt.xlim([-1, len(indices)])
            plt.show()
            plt.close()
        else:
            print("Feature importance not available for this model type.")

    def get_shap_values(self, X):
        """
        Get SHAP values for any model.

        Args:
        X (pandas.DataFrame): The input features.
        """
        explainer = shap.Explainer(self.model, X)
        shap_values = explainer(X)
        shap.summary_plot(shap_values, X, feature_names=X.columns)


class RFC(Model):
    """
    Random Forest Classifier
    """
    def __init__(self):
        Model.__init__(self, RandomForestClassifier())
        self.name = "RFC"


class AdaBoostC(Model):
    """
    Adaboost Classifier
    """
    def __init__(self):
        Model.__init__(self, AdaBoostClassifier())
        self.name = "AdaboostC"


class XGBoostC(Model):
    """
    XGboost Classifier
    """
    def __init__(self):
        Model.__init__(self, XGBClassifier())
        self.name = "XGboostC"


class CatBoostC(Model):
    """
    Catboost Classifier
    """
    def __init__(self):
        Model.__init__(self, CatBoostClassifier())
        self.name = "CatboostC"


class LightGBMC(Model):
    """
    Light Gradient Boosting Machine Classifier
    """
    def __init__(self):
        Model.__init__(self, LGBMClassifier())
        self.name = "LightGBMC"


class RidgeC(Model):
    """
    Ridge Regression Classifier
    """
    def __init__(self):
        Model.__init__(self, RidgeClassifier())
        self.name = "RidgeC"


class RFR(Model):
    """
    Random Forest Regressor
    """
    def __init__(self):
        Model.__init__(self, RandomForestRegressor())
        self.name = "RFR"


class RFC(Model):
    """
    Random Forest Classifier
    """
    def __init__(self):
        Model.__init__(self, RandomForestClassifier())
        self.name = "RFC"


class AdaBoostR(Model):
    """
    Adaboost Regressor
    """
    def __init__(self):
        Model.__init__(self, AdaBoostRegressor())
        self.name = "AdaboostR"


class AdaBoostC(Model):
    """
    Adaboost Classifier
    """
    def __init__(self):
        Model.__init__(self, AdaBoostClassifier())
        self.name = "AdaboostC"


class XGBoostR(Model):
    """
    XGboost Regressor
    """
    def __init__(self):
        Model.__init__(self, XGBRegressor())
        self.name = "XGboostR"


class XGBoostC(Model):
    """
    XGboost Classifier
    """
    def __init__(self):
        Model.__init__(self, XGBClassifier())
        self.name = "XGboostC"


class CatBoostR(Model):
    """
    Catboost Regressor
    """
    def __init__(self):
        Model.__init__(self, CatBoostRegressor())
        self.name = "CatboostR"


class CatBoostC(Model):
    """
    Catboost Classifier
    """
    def __init__(self):
        Model.__init__(self, CatBoostClassifier())
        self.name = "CatboostC"


class LightGBMR(Model):
    """
    Light Gradient Boosting Machine Regressor
    """
    def __init__(self):
        Model.__init__(self, LGBMRegressor())
        self.name = "LightGBMR"


class LightGBMC(Model):
    """
    Light Gradient Boosting Machine Classifier
    """
    def __init__(self):
        Model.__init__(self, LGBMClassifier())
        self.name = "LightGBMC"


class RidgeR(Model):
    """
    Ridge Regression Regressor
    """
    def __init__(self):
        Model.__init__(self, Ridge())
        self.name = "RidgeR"

class RidgeC(Model):
    """
    Ridge Regression Classifier
    """
    def __init__(self):
        Model.__init__(self, RidgeClassifier())
        self.name = "RidgeC"


def get_score_ML(train_df, test_df, save_dir, model_name, ignore_cols, method='classification', params=None, hyperparams=None):
    
    if method not in ['classification', 'regression']:
        raise ValueError("method should be 'classification' or 'regression'")
    
    if not os.path.exists(f'{save_dir}/Model'):
        os.makedirs(f'{save_dir}/Model')

    if not os.path.exists(f'{save_dir}/Factor'):
        os.makedirs(f'{save_dir}/Factor')

    # Model initialization
    model_class = {
        'AdaBoost': AdaBoostC if method == 'classification' else AdaBoostR,
        'XGBoost': XGBoostC if method == 'classification' else XGBoostR,
        'CatBoost': CatBoostC if method == 'classification' else CatBoostR,
        'LightGBM': LightGBMC if method == 'classification' else LightGBMR,
        'RF': RFC if method == 'classification' else RFR,
        'Ridge': RidgeC if method == 'classification' else RidgeR
    }.get(model_name)

    if model_class is None:
        raise ValueError(f'The parameter model_name should be AdaBoost/XGBoost/CatBoost/LightGBM/RF/Ridge, get {model_name} instead.')

    model = model_class()

    feature_cols = train_df.columns.difference(ignore_cols)
    train_features = train_df[feature_cols].values
    train_label = train_df['label'].values
    test_features = test_df[feature_cols].values
    test_label = test_df['label'].values

    model_dir = f'{save_dir}/Model/{model_name}_{method}.m'
    if os.path.exists(model_dir):
        modelFitted = joblib.load(model_dir)
    else:
        if params is not None:
            model.set_params(**params)
        
        if hyperparams is not None:
            model.hyperopt(train_features, train_label, 
                        uniform_dict=hyperparams['uniform'],
                        int_dict=hyperparams['int'], 
                        choice_dict=hyperparams['choice'],
                        maximum=hyperparams['maximum'],
                        scoring=hyperparams['scoring'])

        modelFitted = model.fit(train_features, train_label)
        joblib.dump(modelFitted, model_dir)

    if method == 'classification':
        pred_y = modelFitted.predict(test_features)
        accuracy = accuracy_score(test_label, pred_y)
        precision = precision_score(test_label, pred_y, pos_label=1)
        # report = classification_report(test_label, pred_y)
        print(f"Out of Sample Accuracy by {model_name}: {accuracy}")
        print(f"Out of Sample Precision by {model_name}: {precision}")
        # print(f"Out of Sample Classification Report by {model_name}: \n{report}")

    elif method == 'regression':
        # Predict on training set for threshold determination
        train_preds = modelFitted.predict(train_features)
        thresholds = np.linspace(min(train_preds), max(train_preds), 100)
        best_f1 = 0
        best_threshold = 0
        for threshold in thresholds:
            binary_preds = (train_preds >= threshold).astype(int)
            f1 = f1_score(train_label, binary_preds)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        # Predict on test set using the best threshold
        test_preds = modelFitted.predict(test_features)
        pred_y = (test_preds >= best_threshold).astype(int)
        accuracy = accuracy_score(test_label, pred_y)
        precision = precision_score(test_label, pred_y)
        print(f"Optimal threshold: {best_threshold}")
        print(f"Out of Sample Accuracy by {model_name}: {accuracy}")
        print(f"Out of Sample Precision by {model_name}: {precision}")

    factor = test_df[['date', 'ticker', 'Open', 'High', 'Low', 'Close', 'Volume', 'return']].copy()
    factor['pred'] = pred_y
    factor.to_csv(f"{save_dir}/Factor/{model_name}_{method}.csv", index=False)


# Function to initialize and return the appropriate model based on the method
def initialize_model(model_name, method, params=None):
    # Define classifiers and regressors
    model_classifiers = {
        'AdaBoost': AdaBoostClassifier(),
        'XGBoost': XGBClassifier(),
        'CatBoost': CatBoostClassifier(),
        'LightGBM': LGBMClassifier(),
        'RF': RandomForestClassifier(),
        'Ridge': RidgeClassifier()
    }

    model_regressors = {
        'AdaBoost': AdaBoostRegressor(),
        'XGBoost': XGBRegressor(),
        'CatBoost': CatBoostRegressor(),
        'LightGBM': LGBMRegressor(),
        'RF': RandomForestRegressor(),
        'Ridge': Ridge()
    }

    # Select the appropriate model dictionary
    model_dict = model_classifiers if method == 'classification' else model_regressors

    # Initialize model
    model = model_dict.get(model_name)
    if model is None:
        raise ValueError(f'The parameter model_name should be AdaBoost/XGBoost/CatBoost/LightGBM/RF/Ridge, got {model_name} instead.')

    # Set parameters if provided
    if params and model_name in params:
        model.set_params(**params[model_name])

    return model

# Modified function to handle both classification and regression
def get_score_ML_ensemble(train_df, test_df, save_dir, model_names, ignore_cols, method='classification', ensemble_method='vote', params=None):
    
    if not os.path.exists(f'{save_dir}/Model'):
        os.makedirs(f'{save_dir}/Model')
    if not os.path.exists(f'{save_dir}/Factor'):
        os.makedirs(f'{save_dir}/Factor')

    feature_cols = train_df.columns.difference(ignore_cols)
    train_features = train_df[feature_cols].values
    train_label = train_df['label'].values
    test_features = test_df[feature_cols].values
    test_label = test_df['label'].values

    # Select the ensemble method
    if ensemble_method == 'vote':
        models = [(f"{model_name}_{i}", initialize_model(model_name, method, params)) for i, model_name in enumerate(model_names)]
        if method == 'classification':
            ensemble_model = VotingClassifier(estimators=models, voting='soft')
        else:
            ensemble_model = VotingRegressor(estimators=models)
    elif ensemble_method == 'bag':
        # Assuming a single model for bagging
        models = [(model_name, initialize_model(model_name, method, params)) for model_name in model_names]
        base_model = models[0][1]
        if method == 'classification':
            ensemble_model = BaggingClassifier(base_model, n_estimators=len(models), max_samples=1.0/len(models), bootstrap=True)
        else:
            ensemble_model = BaggingRegressor(base_model, n_estimators=len(models), max_samples=1.0/len(models), bootstrap=True)
    else:
        raise ValueError(f'Invalid ensemble method: {ensemble_method}')

    # Fit and save the ensemble model
    model_dir = f'{save_dir}/Model/{model_names[0]}_{ensemble_method}_{method}_model.m'
    if os.path.exists(model_dir):
        modelFitted = joblib.load(model_dir)
    else:
        modelFitted = ensemble_model.fit(train_features, train_label)
        joblib.dump(modelFitted, model_dir)

    if method == 'classification':
        pred_y = modelFitted.predict(test_features)
        accuracy = accuracy_score(test_label, pred_y)
        precision = precision_score(test_label, pred_y, pos_label=1)
        # report = classification_report(test_label, pred_y)
        print(f"Out of Sample Accuracy by {model_names[0]}: {accuracy}")
        print(f"Out of Sample Precision by {model_names[0]}: {precision}")
        # print(f"Out of Sample Classification Report by {model_name}: \n{report}")

    elif method == 'regression':
        # Predict on training set for threshold determination
        train_preds = modelFitted.predict(train_features)
        thresholds = np.linspace(min(train_preds), max(train_preds), 100)
        best_f1 = 0
        best_threshold = 0
        for threshold in thresholds:
            binary_preds = (train_preds >= threshold).astype(int)
            f1 = f1_score(train_label, binary_preds)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        # Predict on test set using the best threshold
        test_preds = modelFitted.predict(test_features)
        pred_y = (test_preds >= best_threshold).astype(int)
        accuracy = accuracy_score(test_label, pred_y)
        precision = precision_score(test_label, pred_y)
        print(f"Optimal threshold: {best_threshold}")
        print(f"Out of Sample Accuracy by {model_names[0]}: {accuracy}")
        print(f"Out of Sample Precision by {model_names[0]}: {precision}")

    factor = test_df[['date', 'ticker', 'Open', 'High', 'Low', 'Close', 'Volume', 'return']].copy()
    factor['pred'] = pred_y
    factor.to_csv(f"{save_dir}/Factor/{ensemble_method}_{method}_model.csv", index=False)
