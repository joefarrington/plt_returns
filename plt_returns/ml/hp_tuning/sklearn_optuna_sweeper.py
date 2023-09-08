import os
import sklearn
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedGroupKFold, cross_validate, cross_val_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import make_scorer, roc_curve, auc
import mlflow
import hydra
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig

class DatetimeProcessor(BaseEstimator, TransformerMixin):
    """
    Extract components from pandas.Timestamp
    """
    def fit(self, X, y=None):
        return self
    
    def transform(self, X: pd.DataFrame):
        X = X.copy()
        cols = X.columns
        for c in cols:
            X[f'{c}_hour'] = X[c].dt.hour
            X[f'{c}_dayofweek'] = X[c].dt.dayofweek
        X = X.drop(cols, axis=1)
        return X
    
    def get_feature_names_out(self, input_features=None):
        """Return a list of output feature names"""
        feature_names = []
        for c in input_features:
            feature_names.append(f'{c}_hour')
            feature_names.append(f'{c}_weekday')
        return feature_names

def modified_auc(y_true: np.array, y_pred_proba: np.array, max_fpr:float=0.6):
    """Calculate partial_auc"""
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)  # We consider the positive class

    # Only consider the part of the curve where the false positive rate is less than max_fpr
    cutoff_index = np.where(fpr <= max_fpr)[0].max()
    
    partial_fpr = fpr[:cutoff_index+1]
    partial_tpr = tpr[:cutoff_index+1]

    # Catch case where we end up with only one point
    if len(partial_fpr) < 2:
        return 0

    # Calculate the AUC of the partial ROC curve
    partial_auc = auc(partial_fpr, partial_tpr)
    
    return partial_auc

partial_auc_scorer = make_scorer(modified_auc, needs_proba=True)

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):

    # Load the data
    Xy = pd.read_csv(Path(cfg.data_filepath))
    y = Xy['not_all_transfused'] # Now trying to pred at least one unit returned
    pt = Xy['patient_mrn'] # Use for assigned to CV groups
    X = Xy.drop(columns=['request_number', 'patient_mrn', 'not_all_transfused'])

    # Convert specified column to datetime
    datetime_cols = OmegaConf.to_container(cfg.datetime_cols)
    for c in datetime_cols:
        X[c] = pd.to_datetime(X[c])

    other_cat_cols = X.select_dtypes(include='object').columns.drop(['ward_name', 'discipline', 'plt_count_request_location'])
    binary_cols = OmegaConf.to_container(cfg.binary_cols)
    num_cols = X.select_dtypes(exclude=['object', 'datetime64']).columns.drop(binary_cols)

    # Set up the model pipeline components
    clf = hydra.utils.instantiate(cfg.clf)
    preproc_pipeline = ColumnTransformer([
    ("num", hydra.utils.instantiate(cfg.preproc_num), num_cols),
    ("binary", hydra.utils.instantiate(cfg.preproc_num), binary_cols),
    ("ward_name", hydra.utils.instantiate(cfg.preproc_ward_name), ['ward_name']),
    ("discipline", hydra.utils.instantiate(cfg.preproc_discipline), ['discipline']),
    ("plt_count_request_location", hydra.utils.instantiate(cfg.preproc_plt_count_request_location), ['plt_count_request_location']),
    ("other_cat", hydra.utils.instantiate(cfg.preproc_other_cat), other_cat_cols),
    ("dt", DatetimeProcessor(), datetime_cols)
])  

    # Turn any "None" strings in model config to  None
    model_params = OmegaConf.to_container(cfg.model_params, resolve=True)
    for k,v in model_params.items():
        if v == "None":
            model_params[k] = None

    # Create the pipeline and set the params
    pipe = Pipeline([('preproc', preproc_pipeline), ("clf", clf)])
    pipe.set_params(**model_params)

    # Set up mlflow tracking
    mlflow.set_tracking_uri(Path(cfg.mlflow.tracking_uri))
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    with mlflow.start_run() as run:
        # Log key elements of config
        mlflow.log_param("classifier", cfg.clf._target_)
        mlflow.log_param("preproc_num", cfg.preproc_num._target_)
        mlflow.log_param("preproc_ward_name", cfg.preproc_ward_name._target_)
        mlflow.log_param("preproc_discipline", cfg.preproc_discipline._target_)
        mlflow.log_param("preproc_other_cat", cfg.preproc_other_cat._target_)
        mlflow.log_param("cv_splits", cfg.cross_validation.n_splits)
        mlflow.log_param("tuned_metric", cfg.reporting.score)

        # Log params
        mlflow.log_params(model_params)

        # Set up the CV object
        cv = StratifiedGroupKFold(n_splits=cfg.cross_validation.n_splits)
   
        # Run cross validation
        scoring = {"partial_auc": partial_auc_scorer} | OmegaConf.to_container(cfg.cross_validation.scoring)
        cv_runner = cross_validate(pipe, X, y, groups=pt, cv=cv, scoring=scoring, n_jobs=-1, return_train_score=True, verbose=5)
        metric_dict = {}
        for k, v in cv_runner.items():
            metric_dict[f'mean_{k}'] = v.mean()
            metric_dict[f'std_{k}'] = v.std()
            mlflow.log_metrics(metric_dict)

        
        # Store cv results as csv in addition to mlflow logging
        cv_results = pd.DataFrame(cv_runner)
        cv_results.to_csv('cv_results.csv')

        # Store outputs (including hydra config) as mlflow artifacts
        mlflow.log_artifacts(".")

    return metric_dict[f'mean_test_{cfg.reporting.score}']


if __name__ == "__main__":
    main()