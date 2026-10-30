"""Prepare numerical C-MAPSS predictors and leakage-safe preprocessing steps."""
import os
import sys

import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.exception import CustomException  # noqa: E402
from src.logger import logging  # noqa: E402


class DataTransformation:
    def get_data_transformer_object(self, scale=True):
        steps = [('remove_constant', VarianceThreshold(threshold=0.0))]
        if scale:
            steps.append(('scaler', StandardScaler()))
        return Pipeline(steps)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            feature_columns = train_df.drop(columns=['engine_id', 'RUL']).columns.tolist()
            missing = set(feature_columns) - set(test_df.columns)
            if missing:
                raise ValueError(f'Test data missing training features: {sorted(missing)}')
            if not feature_columns:
                raise ValueError('No input features were found')
            if not all(pd.api.types.is_numeric_dtype(train_df[col]) for col in feature_columns):
                raise ValueError('C-MAPSS features must be numeric')

            X_train = train_df[feature_columns]
            y_train = train_df['RUL']
            train_groups = train_df['engine_id']
            X_test = test_df[feature_columns]
            y_test = test_df['RUL']
            logging.info('Prepared %s numerical features', len(feature_columns))
            return X_train, y_train, train_groups, X_test, y_test, feature_columns
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == '__main__':
    from src.components.data_ingestion import DataIngestion

    train_path, test_path = DataIngestion().initiate_data_ingestion('FD001')
    X, y, groups, X_test, y_test, features = (
        DataTransformation().initiate_data_transformation(train_path, test_path)
    )
    print('Train:', X.shape, 'Test:', X_test.shape, 'Features:', len(features))
