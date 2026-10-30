import os
import sys
from collections import deque
from functools import lru_cache

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
from flask import Flask, redirect, render_template, request, url_for

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)

app = application

DATASETS = ['FD001', 'FD002', 'FD003', 'FD004']
ARTIFACT_DIR = os.path.join(PROJECT_ROOT, 'artifacts')
DATA_DIR = os.path.join(PROJECT_ROOT, 'notebooks', 'data')
RECENT_PREDICTIONS = deque(maxlen=10)


def model_path(dataset):
    return os.path.join(ARTIFACT_DIR, dataset + '_model.pkl')


def dataset_options():
    return [{'name': name, 'trained': os.path.isfile(model_path(name))} for name in DATASETS]


def trained_datasets():
    return [item['name'] for item in dataset_options() if item['trained']]


def choose_dataset(requested):
    trained = trained_datasets()
    if not trained:
        return None
    if requested in trained:
        return requested
    return trained[0]


@lru_cache(maxsize=4)
def train_frame(dataset):
    train_path = os.path.join(DATA_DIR, 'train_' + dataset + '.csv')
    return pd.read_csv(train_path)


@lru_cache(maxsize=8)
def feature_defaults(dataset, features):
    try:
        frame = train_frame(dataset)
    except FileNotFoundError:
        return {}
    return {feature: round(float(frame[feature].median()), 2)
            for feature in features if feature in frame.columns}


def build_trend(dataset, engine_id, predictor):
    if engine_id is None:
        return None
    try:
        frame = train_frame(dataset)
    except FileNotFoundError:
        return None
    engine_frame = frame[frame['engine_id'] == engine_id]
    if engine_frame.empty:
        return None
    features = predictor.get_required_features()
    predictions = predictor.predict(engine_frame[features])
    actual = engine_frame['RUL'] if 'RUL' in engine_frame.columns else []
    return {
        'engine_id': int(engine_id),
        'cycles': [int(value) for value in engine_frame['cycle']],
        'predicted': [round(float(value), 2) for value in predictions],
        'actual': [round(float(value), 2) for value in actual],
    }


def remember_prediction(dataset, engine_id, cycle, predicted, lower, upper):
    RECENT_PREDICTIONS.appendleft({
        'dataset': dataset,
        'engine_id': '-' if engine_id in (None, '') else engine_id,
        'cycle': '-' if cycle in (None, '') else round(float(cycle), 2),
        'predicted': round(float(predicted), 2),
        'lower': round(float(lower), 2),
        'upper': round(float(upper), 2),
    })


def dashboard_context(dataset=None, **extra):
    options = dataset_options()
    selected = choose_dataset(dataset)
    features = []
    defaults = {}
    if selected is not None:
        features = PredictPipeline(selected).get_required_features()
        defaults = feature_defaults(selected, tuple(features))

    values = dict(defaults)
    engine_id = 1
    values['engine_id'] = engine_id
    trend = None
    if selected is not None:
        try:
            trend = build_trend(selected, engine_id, PredictPipeline(selected))
        except Exception:
            trend = None

    streams = [feature for feature in features if feature.startswith('operating_setting')]
    sensors = [feature for feature in features if feature.startswith('sensor_')]

    context = {
        'active_page': 'home',
        'datasets': options,
        'dataset': selected,
        'features': features,
        'settings': streams,
        'sensors': sensors,
        'defaults': defaults,
        'values': values,
        'engine_id': engine_id,
        'trend': trend,
        'recent_predictions': list(RECENT_PREDICTIONS),
        'result': None,
        'upload_results': None,
        'upload_count': 0,
        'error': None,
        'tab': 'manual',
    }
    context.update(extra)
    return context


@app.route('/')
def index():
    requested_tab = request.args.get('tab', 'manual')
    tab = requested_tab if requested_tab in ('manual', 'upload') else 'manual'
    dataset = request.args.get('dataset')
    context = dashboard_context(dataset, tab=tab, active_page='home')
    return render_template('home.html', **context)


@app.route('/predict', methods=['GET', 'POST'])
@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return redirect(url_for('index'))

    mode = request.form.get('mode', 'manual')
    requested_dataset = request.form.get('dataset')
    available = trained_datasets()
    if requested_dataset and requested_dataset in DATASETS and requested_dataset not in available:
        context = dashboard_context(available[0] if available else None,
                                    error='No trained model for ' + requested_dataset + '. Train it first.')
        return render_template('home.html', **context)

    dataset = choose_dataset(requested_dataset)
    if dataset is None:
        context = dashboard_context(None, error='No trained model found. Run: python -m src.pipeline.train_pipeline --dataset FD001')
        return render_template('home.html', **context)

    if mode == 'upload':
        return handle_upload(dataset)

    predictor = PredictPipeline(dataset)
    features = predictor.get_required_features()
    context = dashboard_context(dataset, tab='manual', active_page='home')

    values = {}
    feature_values = {}
    errors = []

    engine_id_raw = request.form.get('engine_id', '').strip()
    try:
        engine_id = int(float(engine_id_raw))
        if engine_id < 1:
            raise ValueError
    except (TypeError, ValueError):
        engine_id = 1
        errors.append('Engine ID must be a positive whole number.')

    for feature in features:
        raw = request.form.get(feature, '').strip()
        values[feature] = raw
        if not raw:
            errors.append('Missing value for ' + feature)
            continue
        try:
            feature_values[feature] = float(raw)
        except ValueError:
            errors.append(feature + ' must be a number.')

    if 'cycle' in feature_values and feature_values['cycle'] < 1:
        errors.append('Cycle must be at least 1.')

    if errors:
        context.update(values=values, engine_id=engine_id, error=' '.join(errors))
        return render_template('home.html', **context)

    data = CustomData(dataset=dataset, **feature_values)
    pred_df = data.get_data_as_data_frame()
    results = predictor.predict_with_details(pred_df)
    row = results.iloc[0].to_dict()
    row['engine_id'] = engine_id
    remember_prediction(dataset, engine_id, feature_values['cycle'],
                        row['Predicted RUL'], row['Lower RUL'], row['Upper RUL'])

    context.update(
        values=values,
        engine_id=engine_id,
        result=row,
        trend=build_trend(dataset, engine_id, predictor),
        recent_predictions=list(RECENT_PREDICTIONS),
    )
    return render_template('home.html', **context)


def handle_upload(dataset):
    predictor = PredictPipeline(dataset)
    features = predictor.get_required_features()
    context = dashboard_context(dataset, tab='upload', active_page='home')
    uploaded_file = request.files.get('csv_file')

    if uploaded_file is None or uploaded_file.filename == '':
        context['error'] = 'Choose a CSV file to upload.'
        return render_template('home.html', **context)
    if not uploaded_file.filename.lower().endswith('.csv'):
        context['error'] = 'Only .csv files are supported.'
        return render_template('home.html', **context)

    try:
        frame = pd.read_csv(uploaded_file)
    except Exception as error:
        context['error'] = 'Could not read the CSV file: ' + str(error)
        return render_template('home.html', **context)

    missing = [feature for feature in features if feature not in frame.columns]
    if missing:
        context['error'] = 'CSV is missing required columns: ' + ', '.join(missing)
        return render_template('home.html', **context)

    try:
        results = predictor.predict_with_details(frame)
    except Exception as error:
        context['error'] = 'Prediction failed: ' + str(error)
        return render_template('home.html', **context)

    for row in results.head(8).to_dict('records'):
        remember_prediction(dataset, row.get('engine_id'), row.get('cycle'),
                            row['Predicted RUL'], row['Lower RUL'], row['Upper RUL'])

    context.update(
        upload_results=results.head(20).round(2).to_dict('records'),
        upload_count=len(results),
        recent_predictions=list(RECENT_PREDICTIONS),
        error=None,
    )
    return render_template('home.html', **context)


def load_performance():
    entries = []
    summary_path = os.path.join(ARTIFACT_DIR, 'training_summary.csv')
    summary = None
    if os.path.isfile(summary_path):
        try:
            summary = pd.read_csv(summary_path)
        except Exception:
            summary = None

    for dataset in DATASETS:
        if not os.path.isfile(model_path(dataset)):
            continue
        entry = {'dataset': dataset, 'comparison': [], 'significance': [], 'summary': None}
        comparison_path = os.path.join(ARTIFACT_DIR, dataset + '_comparison.csv')
        significance_path = os.path.join(ARTIFACT_DIR, dataset + '_t_tests.csv')

        if os.path.isfile(comparison_path):
            comparison = pd.read_csv(comparison_path)
            if 'RMSE' in comparison.columns:
                comparison = comparison.sort_values('RMSE')
            entry['comparison'] = comparison.round(3).to_dict('records')
        if os.path.isfile(significance_path):
            significance = pd.read_csv(significance_path)
            entry['significance'] = significance.round(4).to_dict('records')
        if summary is not None:
            match = summary[summary['dataset'] == dataset]
            if not match.empty:
                entry['summary'] = match.round(3).to_dict('records')[0]
        entries.append(entry)
    return entries


@app.route('/performance')
def performance():
    return render_template('performance.html', active_page='performance',
                           entries=load_performance(), datasets=DATASETS)


@app.route('/about')
def about():
    info = []
    for dataset in DATASETS:
        try:
            frame = train_frame(dataset)
        except FileNotFoundError:
            info.append({'name': dataset, 'engines': '-', 'rows': '-', 'max_cycle': '-'})
            continue
        info.append({
            'name': dataset,
            'engines': int(frame['engine_id'].nunique()),
            'rows': int(len(frame)),
            'max_cycle': int(frame['cycle'].max()),
        })
    return render_template('about.html', active_page='about', info=info)


if __name__ == '__main__':
    app.run(host='0.0.0.0')
