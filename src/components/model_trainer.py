"""Compare regressors with engine-wise validation and save a fitted pipeline per subset."""
import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from scipy.stats import ttest_rel
from statsmodels.stats.multitest import multipletests

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.components.data_transformation import DataTransformation  # noqa: E402
from src.exception import CustomException  # noqa: E402
from src.logger import logging  # noqa: E402
from src.utils import save_object  # noqa: E402


@dataclass
class ModelTrainerConfig:
    artifact_dir: str = os.path.join(PROJECT_ROOT, 'artifacts')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def get_models(self):
        transform = DataTransformation()

        def pipeline(regressor, scale=False):
            return Pipeline([
                ('preprocess', transform.get_data_transformer_object(scale=scale)),
                ('model', regressor)
            ])

        models = {
            'Linear Regression': pipeline(LinearRegression(), scale=True),
            'Ridge': pipeline(Ridge(alpha=10), scale=True),
            'Decision Tree': pipeline(DecisionTreeRegressor(max_depth=8, min_samples_leaf=10, random_state=42)),
            'Random Forest': pipeline(RandomForestRegressor(n_estimators=50, max_depth=12, min_samples_leaf=5, random_state=42, n_jobs=-1)),
            'Gradient Boosting': pipeline(GradientBoostingRegressor(n_estimators=80, max_depth=3, random_state=42)),
            'SVR': pipeline(SVR(C=100, epsilon=10), scale=True)
        }
        try:
            from xgboost import XGBRegressor # pyright: ignore[reportMissingImports]
            models['XGBoost'] = pipeline(XGBRegressor(n_estimators=80, max_depth=4, learning_rate=0.1, random_state=42, n_jobs=2))
        except ImportError:
            logging.info('XGBoost is optional and is not installed')
        return models

    def initiate_model_trainer(self, X, y, groups, X_test, y_test, dataset='FD001'):
        try:
            splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
            train_idx, val_idx = next(splitter.split(X, y, groups=groups))
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            val_engines = groups.iloc[val_idx]

            models = self.get_models()
            results = []
            engine_errors = pd.DataFrame({'engine_id': val_engines.to_numpy()})
            fitted_predictions = {}

            for model_name, model in models.items():
                print('  Training', model_name, flush=True)
                fitted_model = clone(model)
                fitted_model.fit(X_train, y_train)
                predictions = np.maximum(0, fitted_model.predict(X_val))
                fitted_predictions[model_name] = predictions
                engine_errors[model_name] = np.abs(y_val.to_numpy() - predictions)
                results.append({
                    'Dataset': dataset, 'Model': model_name,
                    'MAE': mean_absolute_error(y_val, predictions),
                    'RMSE': np.sqrt(mean_squared_error(y_val, predictions)),
                    'R2': r2_score(y_val, predictions)
                })

            comparison = pd.DataFrame(results).sort_values('RMSE').reset_index(drop=True)
            best_name = comparison.loc[0, 'Model']
            best_model = models[best_name]
            best_pred = fitted_predictions[best_name]
            engine_errors = engine_errors.groupby('engine_id').mean()

            significance_rows = []
            for model_name in models:
                if model_name != best_name:
                    t_stat, p_value = ttest_rel(engine_errors[best_name], engine_errors[model_name])
                    significance_rows.append({'Best Model': best_name, 'Compared With': model_name,
                                              't-statistic': t_stat, 'p-value': p_value})
            significance = pd.DataFrame(significance_rows)
            if not significance.empty:
                significance['Adjusted p'] = multipletests(significance['p-value'], method='holm')[1]
                significance['Significant (0.05)'] = significance['Adjusted p'] < 0.05

            folds = GroupKFold(n_splits=3)
            print('  Cross-validating', best_name, flush=True)
            cv_scores = cross_val_score(clone(best_model), X_train, y_train,
                                        groups=groups.iloc[train_idx], cv=folds,
                                        scoring='neg_mean_squared_error', n_jobs=1)
            cv_rmse = float(np.sqrt(-cv_scores).mean())
            near_failure = y_val <= 30
            near_failure_mae = (float(mean_absolute_error(y_val[near_failure], best_pred[near_failure]))
                                if near_failure.any() else None)

            margin = float(np.quantile(np.abs(y_val.to_numpy() - best_pred), 0.90))
            final_model = clone(best_model)
            final_model.fit(X, y)
            test_predictions = np.maximum(0, final_model.predict(X_test))

            os.makedirs(self.model_trainer_config.artifact_dir, exist_ok=True)
            model_path = os.path.join(self.model_trainer_config.artifact_dir, dataset + '_model.pkl')
            save_object(model_path, {'pipeline': final_model, 'features': X.columns.tolist(),
                                     'dataset': dataset, 'model_name': best_name, 'margin': margin})

            comparison.to_csv(os.path.join(self.model_trainer_config.artifact_dir, dataset + '_comparison.csv'), index=False)
            significance.to_csv(os.path.join(self.model_trainer_config.artifact_dir, dataset + '_t_tests.csv'), index=False)
            test_results = pd.DataFrame({'Actual RUL': y_test.to_numpy(), 'Predicted RUL': test_predictions})
            test_results.to_csv(os.path.join(self.model_trainer_config.artifact_dir, dataset + '_predictions.csv'), index=False)

            report = {
                'dataset': dataset, 'best_model': best_name, 'model_path': model_path,
                'validation_rmse': float(comparison.loc[0, 'RMSE']),
                'cv_rmse': cv_rmse, 'near_failure_mae': near_failure_mae,
                'test_mae': float(mean_absolute_error(y_test, test_predictions)),
                'test_rmse': float(np.sqrt(mean_squared_error(y_test, test_predictions))),
                'test_r2': float(r2_score(y_test, test_predictions)),
                'validation_margin_90': margin
            }
            logging.info('%s: Best=%s, test RMSE=%.3f', dataset, best_name, report['test_rmse'])
            return report, comparison, significance
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == '__main__':
    from src.components.data_ingestion import DataIngestion

    train_path, test_path = DataIngestion().initiate_data_ingestion('FD001')
    X, y, groups, X_test, y_test, features = (
        DataTransformation().initiate_data_transformation(train_path, test_path)
    )
    trainer = ModelTrainer()
    report, comparison, significance = trainer.initiate_model_trainer(
        X, y, groups, X_test, y_test, 'FD001'
    )
    print(comparison.to_string(index=False))
    print('\nBest model:', report['best_model'], '| Test RMSE:', round(report['test_rmse'], 3))
