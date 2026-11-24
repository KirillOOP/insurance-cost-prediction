from pathlib import Path
import numpy as np
import torch
from torch import nn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE = Path(__file__).resolve().parent
DATA = BASE / "data" / "insurance.csv"
ART = BASE / "artifacts"
PLOTS = BASE / "static" / "plots"
FEATURES = ["age", "sex", "bmi", "children", "smoker", "region"]
NUM = ["age", "bmi", "children"]
CAT = ["sex", "smoker", "region"]
TARGET = "charges"
MODEL_NAMES = {
    "linear": "Linear Regression",
    "ridge": "Ridge Regression",
    "forest": "Random Forest Regressor",
    "boost": "Gradient Boosting Regressor",
    "mlp": "Багатошаровий перцептрон",
    "mlp_regularized": "MLP із Dropout та L2-регуляризацією",
}

def preprocessing():
    return ColumnTransformer([
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), NUM),
        ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                   ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), CAT),
    ], remainder="drop", sparse_threshold=0)

class NeuralRegressor(nn.Module):
    def __init__(self, in_features, regularized=False):
        super().__init__()
        if regularized:
            self.net = nn.Sequential(nn.Linear(in_features, 64), nn.ReLU(), nn.Dropout(0.25),
                                     nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.2),
                                     nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1))
        else:
            self.net = nn.Sequential(nn.Linear(in_features, 32), nn.ReLU(),
                                     nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, x):
        return self.net(x)

def predict_torch(model, X, y_scaler):
    model.eval()
    with torch.no_grad():
        vals = model(torch.tensor(np.asarray(X, dtype=np.float32))).numpy().reshape(-1, 1)
    return y_scaler.inverse_transform(vals).reshape(-1)
