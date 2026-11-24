import json
import random
from copy import deepcopy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from ml import DATA, ART, PLOTS, FEATURES, MODEL_NAMES, TARGET, preprocessing, NeuralRegressor, predict_torch

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(2)
ART.mkdir(exist_ok=True)
PLOTS.mkdir(parents=True, exist_ok=True)

if not DATA.is_file():
    raise FileNotFoundError(f"Файл {DATA} не знайдено. Скачайте insurance.csv з Kaggle і покладіть у data/.")
df = pd.read_csv(DATA)
needed = FEATURES + [TARGET]
missing_columns = sorted(set(needed) - set(df.columns))
if missing_columns:
    raise ValueError(f"Відсутні стовпці: {missing_columns}. Очікуються: {needed}")
original_rows = len(df)
missing_values = {k: int(v) for k, v in df[needed].isnull().sum().items()}
duplicates = int(df.duplicated().sum())
types = {k: str(v) for k, v in df[needed].dtypes.items()}
df = df[needed].drop_duplicates().copy()
for col in ['age', 'bmi', 'children', 'charges']:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df = df.dropna(subset=[TARGET])
if len(df) < 100:
    raise ValueError('Для навчання потрібно щонайменше 100 коректних рядків.')
X = df[FEATURES]
y = df[TARGET].astype(float)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=SEED)
processor = preprocessing()
Xtr = np.asarray(processor.fit_transform(X_train), dtype=np.float32)
Xte = np.asarray(processor.transform(X_test), dtype=np.float32)
joblib.dump(processor, ART / 'preprocessor.joblib')

models = {
    'linear': LinearRegression(),
    'ridge': Ridge(alpha=10.0),
    'forest': RandomForestRegressor(n_estimators=200, min_samples_leaf=2, random_state=SEED, n_jobs=-1),
    'boost': GradientBoostingRegressor(n_estimators=170, learning_rate=0.05, max_depth=3, random_state=SEED),
}
records = []
predictions = {}

def score(key, yp):
    yp = np.asarray(yp).reshape(-1)
    mse = mean_squared_error(y_test, yp)
    row = {'model_id': key, 'model': MODEL_NAMES[key], 'MAE': float(mean_absolute_error(y_test, yp)),
           'MSE': float(mse), 'RMSE': float(np.sqrt(mse)), 'R2': float(r2_score(y_test, yp))}
    records.append(row)
    predictions[key] = yp
    print(f"{row['model']:<41} MAE={row['MAE']:8.2f} RMSE={row['RMSE']:8.2f} R²={row['R2']:.4f}")

for key, model in models.items():
    model.fit(Xtr, y_train)
    joblib.dump(model, ART / f'{key}.joblib')
    score(key, model.predict(Xte))

xt, xv, yt, yv = train_test_split(Xtr, y_train.to_numpy(), test_size=0.20, random_state=SEED)
y_scaler = StandardScaler().fit(yt.reshape(-1, 1))
joblib.dump(y_scaler, ART / 'target_scaler.joblib')
xt_t = torch.tensor(xt, dtype=torch.float32)
xv_t = torch.tensor(xv, dtype=torch.float32)
yt_t = torch.tensor(y_scaler.transform(yt.reshape(-1, 1)), dtype=torch.float32)
yv_t = torch.tensor(y_scaler.transform(yv.reshape(-1, 1)), dtype=torch.float32)

for key in ('mlp', 'mlp_regularized'):
    torch.manual_seed(SEED)
    regularized = key == 'mlp_regularized'
    net = NeuralRegressor(Xtr.shape[1], regularized=regularized)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.004, weight_decay=0.005 if regularized else 0.0)
    loss_fn = nn.MSELoss()
    best_loss, patience, remaining, best_state = float('inf'), 30, 30, None
    history = []
    for epoch in range(350):
        net.train()
        for inds in torch.randperm(len(xt_t)).split(64):
            optimizer.zero_grad()
            loss = loss_fn(net(xt_t[inds]), yt_t[inds])
            loss.backward()
            optimizer.step()
        net.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(net(xv_t), yv_t).item())
        history.append(val_loss)
        if val_loss < best_loss - 1e-5:
            best_loss = val_loss
            best_state = deepcopy(net.state_dict())
            remaining = patience
        else:
            remaining -= 1
            if remaining == 0:
                break
    net.load_state_dict(best_state)
    torch.save(net.state_dict(), ART / f'{key}.pt')
    score(key, predict_torch(net, Xte, y_scaler))
    plt.figure(figsize=(7, 4))
    plt.plot(np.arange(1, len(history) + 1), history)
    plt.xlabel('Епоха')
    plt.ylabel('Помилка на валідації')
    plt.title('Навчання - ' + MODEL_NAMES[key])
    plt.tight_layout()
    plt.savefig(PLOTS / f'{key}_loss.png', dpi=150)
    plt.close()

results = pd.DataFrame(records).sort_values('RMSE').reset_index(drop=True)
results.to_csv(ART / 'metrics.csv', index=False, encoding='utf-8-sig')
compare = pd.DataFrame({'actual': y_test.to_numpy()})
for key, vals in predictions.items():
    compare[key] = vals
compare.to_csv(ART / 'predictions.csv', index=False, encoding='utf-8-sig')

bands = pd.qcut(y_test, q=3, labels=['Низькі', 'Середні', 'Високі'], duplicates='drop')
band_rows = []
for key, yp in predictions.items():
    for band in bands.cat.categories:
        mask = np.asarray(bands == band)
        band_rows.append({'model_id': key, 'model': MODEL_NAMES[key], 'range': str(band),
                          'count': int(mask.sum()), 'MAE': float(mean_absolute_error(y_test.to_numpy()[mask], yp[mask]))})
pd.DataFrame(band_rows).to_csv(ART / 'errors_by_range.csv', index=False, encoding='utf-8-sig')

plt.figure(figsize=(11, 5))
plt.bar(results['model'], results['RMSE'])
plt.xticks(rotation=20, ha='right')
plt.ylabel('RMSE, USD')
plt.title('Порівняння моделей за RMSE')
plt.tight_layout()
plt.savefig(PLOTS / 'metrics_rmse.png', dpi=150)
plt.close()

for key, pred in predictions.items():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].scatter(y_test, pred, alpha=.55, s=15)
    xymin = float(min(y_test.min(), pred.min()))
    xymax = float(max(y_test.max(), pred.max()))
    axes[0].plot([xymin, xymax], [xymin, xymax], '--', color='black', linewidth=1)
    axes[0].set(xlabel='Фактичні витрати, USD', ylabel='Прогнозовані витрати, USD', title='Фактичні та прогнозовані значення')
    axes[1].scatter(pred, y_test.to_numpy() - pred, alpha=.55, s=15)
    axes[1].axhline(0, linestyle='--', color='black', linewidth=1)
    axes[1].set(xlabel='Прогнозовані витрати, USD', ylabel='Залишок, USD', title='Графік залишків')
    fig.suptitle(MODEL_NAMES[key]); fig.tight_layout()
    fig.savefig(PLOTS / f'{key}_predictions.png', dpi=150)
    plt.close(fig)

summary = {
    'source': 'Medical Cost Personal Dataset - Kaggle (mirichoi0218)',
    'original_rows': original_rows, 'duplicates': duplicates, 'missing_values': missing_values,
    'types': types, 'after_cleaning': len(df), 'train_size': len(X_train), 'test_size': len(X_test),
    'features_after_encoding': int(Xtr.shape[1]),
    'target_min': float(y.min()), 'target_max': float(y.max()),
    'target_mean': float(y.mean()), 'target_median': float(y.median()),
    'best_model_id': str(results.loc[0, 'model_id']), 'random_state': SEED,
}
(ART / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
print('\nГотово! Всі 6 моделей, метрики та графіки збережено в artifacts/ і static/plots/.')
