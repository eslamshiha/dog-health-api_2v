from fastapi import FastAPI
import joblib
import numpy as np
import os
import gdown



# تحميل الموديل لو مش موجود
if not os.path.exists("dog_health_model.pkl"):
    gdown.download(
        "https://drive.google.com/uc?id=1FjJQ5mCJp0Aj-qnZBnZoDBw2jcakoxWD",
        "dog_health_model.pkl",
        quiet=False
    )

if not os.path.exists("scaler.pkl"):
    gdown.download(
        "https://drive.google.com/uc?id=15wnB38NIX3Lo-z2ZvkEV0DDJe8YIBEcV",
        "scaler.pkl",
        quiet=False
    )

if not os.path.exists("label_encoder.pkl"):
    gdown.download(
        "https://drive.google.com/uc?id=10ZQz9i9eMtsW0cntpmjjIC9H6Gw5f1Xp",
        "label_encoder.pkl",
        quiet=False
    )

if not os.path.exists("imputer.pkl"):
    gdown.download(
        "https://drive.google.com/uc?id=16McYA3KpTknsQHz_GwO0MLkEVZ83GRfT",
        "imputer.pkl",
        quiet=False
    )

app = FastAPI()

model        = joblib.load("dog_health_model.pkl")
scaler       = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")
imputer      = joblib.load("imputer.pkl")         


def extract_features_window(window):
    arr = np.array(window, dtype=np.float32)

    if len(arr) < 100:
        repeats = int(np.ceil(100 / len(arr)))
        arr = np.tile(arr, (repeats, 1))[:100]

    means = np.mean(arr, axis=0)
    stds  = np.std(arr, axis=0)

    energy      = np.sum(stds ** 2)
    means_safe  = np.abs(means) + 1e-8
    entropy     = -np.sum(means_safe * np.log(means_safe))

    ratio = means[0] / (means[1] + 1e-8) if abs(means[1]) > 1e-8 else 0.0

    features = np.concatenate([means, stds, [energy, entropy, ratio]])
    return features.reshape(1, -1)


def vital_status(window_hr, window_temp):
    mean_hr   = np.mean(window_hr)
    mean_temp = np.mean(window_temp)

    if mean_hr > 140 or mean_temp > 39.5:
        return "Fatigued"
    elif mean_hr > 120 or mean_temp > 39.2:
        return "Warning"
    else:
        return "Healthy"


def final_decision(motion, vital):
    if motion == "Fatigued" and vital == "Fatigued":
        return "Fatigued"
    elif motion == "Fatigued" or vital == "Fatigued":
        return "Warning"
    else:
        return "Healthy"


@app.get("/")
def home():
    return {"message": "Model API is running"}


@app.post("/predict")
def predict(data: dict):
    motion_window = data["motion_window"]

    features = extract_features_window(motion_window)

    features = imputer.transform(features)
    features = scaler.transform(features)

    pred          = model.predict(features)
    motion_status = label_encoder.inverse_transform(pred)[0]

    proba = model.predict_proba(features)[0]
    proba_dict = {label_encoder.classes_[i]: round(float(p), 3) for i, p in enumerate(proba)}
    print("Probabilities:", proba_dict)



    hr_window   = data["hr_window"]
    temp_window = data["temp_window"]

    vital = vital_status(hr_window, temp_window)
    final = final_decision(motion_status, vital)

    if vital == "Fatigued":
        reason = "High heart rate or temperature detected"
    elif motion_status == "Fatigued":
        reason = "Abnormal movement detected"
    else:
        reason = "Dog is in normal condition"

    return {
        "motion_status": motion_status,
        "vital_status":  vital,
        "final_status":  final,
        "reason":        reason
    }