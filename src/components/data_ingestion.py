"""Read one C-MAPSS subset's existing train and test CSV files."""
import os
import sys
from dataclasses import dataclass

import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.exception import CustomException
from src.logger import logging


@dataclass
class DataIngestionConfig:
    data_dir: str = os.path.join(PROJECT_ROOT, 'notebooks', 'data')
    artifact_dir: str = os.path.join(PROJECT_ROOT, 'artifacts')


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self, dataset='FD001'):
        try:
            if dataset not in ('FD001', 'FD002', 'FD003', 'FD004'):
                raise ValueError('Dataset must be FD001, FD002, FD003, or FD004')

            folder = self.ingestion_config.data_dir
            train_path = os.path.join(folder, 'train_' + dataset + '.csv')
            test_path = os.path.join(folder, 'test_last_cycle_' + dataset + '_labeled.csv')
            if not os.path.isfile(train_path) or not os.path.isfile(test_path):
                raise FileNotFoundError(
                    f'Expected {train_path} and {test_path}. '
                    'If your files have different names, update these two paths.'
                )

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            required = {'engine_id', 'cycle', 'RUL'}
            for label, frame in [('train', train_df), ('test', test_df)]:
                missing_columns = required - set(frame.columns)
                if missing_columns:
                    raise ValueError(f'{dataset} {label} missing columns: {sorted(missing_columns)}')
                if frame.isna().any().any():
                    raise ValueError(f'{dataset} {label} contains missing values; inspect the CSV')
                if frame.duplicated(['engine_id', 'cycle']).any():
                    raise ValueError(f'{dataset} {label} has duplicate engine-cycle pairs')
                if (frame['RUL'] < 0).any():
                    raise ValueError(f'{dataset} {label} has negative RUL labels')

            os.makedirs(self.ingestion_config.artifact_dir, exist_ok=True)
            saved_train = os.path.join(self.ingestion_config.artifact_dir, dataset + '_train.csv')
            saved_test = os.path.join(self.ingestion_config.artifact_dir, dataset + '_test.csv')
            train_df.to_csv(saved_train, index=False)
            test_df.to_csv(saved_test, index=False)
            logging.info('%s loaded: train=%s rows, test=%s rows', dataset, len(train_df), len(test_df))
            return saved_train, saved_test
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == '__main__':
    ingestion = DataIngestion()
    for name in ['FD001', 'FD002', 'FD003', 'FD004']:
        saved_train, saved_test = ingestion.initiate_data_ingestion(name)
        print(name, '->', saved_train, '|', saved_test)
