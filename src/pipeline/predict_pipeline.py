"""Predict engine remaining useful life using a saved C-MAPSS pipeline."""
import os
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.exception import CustomException
from src.utils import load_object


class CustomData:
    def __init__(self, dataset='FD001', **features):
        self.dataset = dataset
        self.features = features

    def get_data_as_data_frame(self):
        if not self.features:
            raise ValueError('No feature values provided')
        return pd.DataFrame([self.features])


class PredictPipeline:
    def __init__(self, dataset='FD001'):
        if dataset not in ['FD001', 'FD002', 'FD003', 'FD004']:
            raise ValueError('Choose FD001, FD002, FD003, or FD004')
        self.dataset = dataset
        self.model_path = os.path.join(PROJECT_ROOT, 'artifacts', dataset + '_model.pkl')

    def load_model(self):
        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(
                f'{self.model_path} not found. Train the {self.dataset} model first.'
            )
        return load_object(self.model_path)

    def get_required_features(self):
        saved = self.load_model()
        return saved['features']

    def predict(self, features):
        try:
            saved = self.load_model()
            required = saved['features']

            if isinstance(features, dict):
                features = pd.DataFrame([features])
            if not isinstance(features, pd.DataFrame):
                raise TypeError('Pass a pandas DataFrame or one dictionary of sensor values')
            if features.empty:
                raise ValueError('No engine observations provided')

            missing = [col for col in required if col not in features.columns]
            if missing:
                raise ValueError('Missing required features: ' + ', '.join(missing))

            X = features[required].copy()
            for col in required:
                X[col] = pd.to_numeric(X[col], errors='raise')
            if X.isna().any().any() or not np.isfinite(X.to_numpy(dtype=float)).all():
                raise ValueError('Sensor values must be finite numbers without missing values')
            if 'cycle' in X.columns and (X['cycle'] < 1).any():
                raise ValueError('cycle must be at least 1')

            predictions = saved['pipeline'].predict(X)
            return np.maximum(0, predictions)

        except Exception as e:
            raise CustomException(e, sys)

    def predict_with_details(self, features):
        try:
            if isinstance(features, dict):
                features = pd.DataFrame([features])

            saved = self.load_model()
            predicted_rul = self.predict(features)
            margin = saved['margin']

            output = pd.DataFrame({
                'Dataset': self.dataset,
                'Predicted RUL': np.round(predicted_rul, 2),
                'Lower RUL': np.round(np.maximum(0, predicted_rul - margin), 2),
                'Upper RUL': np.round(predicted_rul + margin, 2)
            })
            if 'engine_id' in features.columns:
                output.insert(1, 'engine_id', features['engine_id'].to_numpy())
            if 'cycle' in features.columns:
                output.insert(2 if 'engine_id' in features.columns else 1,
                              'cycle', features['cycle'].to_numpy())
            return output

        except Exception as e:
            raise CustomException(e, sys)

    def predict_csv(self, csv_path, save_path=None):
        try:
            data = pd.read_csv(csv_path)
            results = self.predict_with_details(data)
            if save_path is not None:
                folder = os.path.dirname(save_path)
                if folder:
                    os.makedirs(folder, exist_ok=True)
                results.to_csv(save_path, index=False)
            return results

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Predict C-MAPSS RUL from a CSV')
    parser.add_argument('--dataset', default='FD001',
                        choices=['FD001', 'FD002', 'FD003', 'FD004'])
    parser.add_argument('--csv', required=True, help='Input CSV containing sensor columns')
    parser.add_argument('--output', default=None, help='Optional predictions CSV path')
    args = parser.parse_args()

    predictor = PredictPipeline(args.dataset)
    result = predictor.predict_csv(args.csv, args.output)
    print(result.head(10).to_string(index=False))
