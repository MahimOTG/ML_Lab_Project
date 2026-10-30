"""Train C-MAPSS models for FD001, FD002, FD003 and FD004."""
import os
import sys

import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.exception import CustomException
from src.logger import logging


class TrainPipeline:
    def run(self, dataset='all'):
        try:
            if dataset == 'all':
                datasets = ['FD001', 'FD002', 'FD003', 'FD004']
            elif dataset in ['FD001', 'FD002', 'FD003', 'FD004']:
                datasets = [dataset]
            else:
                raise ValueError('Choose FD001, FD002, FD003, FD004, or all')

            reports = []
            for name in datasets:
                print('\nTraining dataset:', name)
                logging.info('Starting pipeline for %s', name)

                ingestion = DataIngestion()
                train_path, test_path = ingestion.initiate_data_ingestion(name)

                transformation = DataTransformation()
                X, y, groups, X_test, y_test, features = (
                    transformation.initiate_data_transformation(train_path, test_path)
                )

                trainer = ModelTrainer()
                report, comparison, significance = trainer.initiate_model_trainer(
                    X, y, groups, X_test, y_test, name
                )

                reports.append(report)
                print('Selected model:', report['best_model'])
                print('Validation RMSE:', round(report['validation_rmse'], 3))
                print('Test RMSE:', round(report['test_rmse'], 3))
                print('Test MAE:', round(report['test_mae'], 3))
                print('Test R2:', round(report['test_r2'], 3))
                logging.info('Finished %s, %s features', name, len(features))

            summary = pd.DataFrame(reports)
            artifact_dir = os.path.join(PROJECT_ROOT, 'artifacts')
            os.makedirs(artifact_dir, exist_ok=True)
            summary_path = os.path.join(artifact_dir, 'training_summary.csv')
            if os.path.isfile(summary_path):
                existing = pd.read_csv(summary_path)
                existing = existing[~existing['dataset'].isin(summary['dataset'])]
                summary = pd.concat([existing, summary], ignore_index=True)
            summary = summary.sort_values('dataset').reset_index(drop=True)
            summary.to_csv(summary_path, index=False)
            return summary

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train C-MAPSS RUL prediction models')
    parser.add_argument('--dataset', default='all',
                        choices=['all', 'FD001', 'FD002', 'FD003', 'FD004'])
    args = parser.parse_args()

    summary = TrainPipeline().run(args.dataset)
    print('\nTraining summary:')
    print(summary.to_string(index=False))
