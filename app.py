import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from flask import Flask, render_template, request
from ml import ART, PLOTS, MODEL_NAMES, FEATURES, NeuralRegressor, predict_torch

app = Flask(__name__)


def loaded():
    return (ART / 'summary.json').is_file() and (ART / 'preprocessor.joblib').is_file()


def overview():
    if not loaded():
        return None, None
    summary = json.loads((ART / 'summary.json').read_text(encoding='utf-8'))
    metrics = pd.read_csv(ART / 'metrics.csv').to_dict(orient='records')
    return summary, metrics


@app.get('/')
def index():
    summary, metrics = overview()
    return render_template('index.html', summary=summary, metrics=metrics,
                           available=loaded(), names=MODEL_NAMES)


@app.get('/analytics')
def analytics():
    summary, metrics = overview()
    if not loaded():
        return render_template('analytics.html', available=False)
    error_ranges = pd.read_csv(ART / 'errors_by_range.csv').to_dict(orient='records')
    plots = [f'{key}_predictions.png' for key in MODEL_NAMES]
    return render_template('analytics.html', available=True, metrics=metrics,
                           ranges=error_ranges, plots=plots, names=MODEL_NAMES)


def model_predict(key, frame):
    processor = joblib.load(ART / 'preprocessor.joblib')
    X = np.asarray(processor.transform(frame), dtype=np.float32)
    if key in ('mlp', 'mlp_regularized'):
        net = NeuralRegressor(X.shape[1], regularized=(key == 'mlp_regularized'))
        try:
            state = torch.load(ART / f'{key}.pt', map_location='cpu', weights_only=True)
        except FileNotFoundError as exc:
            raise ValueError('Модель ще не навчена') from exc
        net.load_state_dict(state)
        y_scaler = joblib.load(ART / 'target_scaler.joblib')
        return float(predict_torch(net, X, y_scaler)[0])
    estimator = joblib.load(ART / f'{key}.joblib')
    return float(estimator.predict(X)[0])


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    summary, metrics = overview()
    result = None
    error = None
    values = {'age': '35', 'bmi': '27.5', 'children': '1', 'sex': 'female',
              'smoker': 'no', 'region': 'southeast', 'model': 'boost'}
    if request.method == 'POST':
        values.update({field: request.form.get(field, '').strip() for field in [*FEATURES, 'model']})
        try:
            if not loaded():
                raise ValueError('Спочатку виконайте навчання через python train.py')
            age = int(values['age'])
            bmi = float(values['bmi'].replace(',', '.'))
            children = int(values['children'])
            if not (18 <= age <= 100 and 10 <= bmi <= 70 and 0 <= children <= 15):
                raise ValueError('Некоректні значення віку, індексу маси тіла або кількості дітей.')
            if values['sex'] not in ('male', 'female') or values['smoker'] not in ('yes', 'no'):
                raise ValueError('Некоректна категоріальна ознака.')
            if values['region'] not in ('southwest', 'southeast', 'northwest', 'northeast'):
                raise ValueError('Оберіть коректний регіон.')
            if values['model'] not in MODEL_NAMES:
                raise ValueError('Оберіть модель зі списку.')
            frame = pd.DataFrame([{'age': age, 'sex': values['sex'], 'bmi': bmi,
                                   'children': children, 'smoker': values['smoker'],
                                   'region': values['region']}])
            result = model_predict(values['model'], frame)
        except (ValueError, TypeError, FileNotFoundError) as exc:
            error = str(exc)
    return render_template('predict.html', names=MODEL_NAMES, available=loaded(),
                           values=values, result=result, error=error)


if __name__ == '__main__':
    app.run(debug=False)
