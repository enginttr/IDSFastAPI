import pandas as pd
import numpy as np
import json
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.inspection import permutation_importance


DATASET_PATH = "../dataset/CICIDS2017.csv"

MODEL_RF_PATH = "rf_model.pkl"
MODEL_KNN_PATH = "knn_model.pkl"
MODEL_SVM_PATH = "svm_model.pkl"

SCALER_PATH = "scaler.pkl"
FEATURES_PATH = "selected_features.pkl"
METRICS_PATH = "model_metrics.json"
IMPORTANCE_PATH = "feature_importance.json"
TEST_DATASET_PATH = "test_dataset.csv"

SAMPLE_SIZE = 100000
FEATURE_COUNT = 100


def clean_dataset(df):
    df.columns = df.columns.str.strip()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    df["Label"] = df["Label"].apply(
        lambda x: 0 if str(x).strip().upper() == "BENIGN" else 1
    )

    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

    if "Label" in numeric_columns:
        numeric_columns.remove("Label")

    return df, numeric_columns


def calculate_metrics(y_test, y_pred):
    return {
        "accuracy": round(accuracy_score(y_test, y_pred) * 100, 2),
        "precision": round(precision_score(y_test, y_pred, zero_division=0) * 100, 2),
        "recall": round(recall_score(y_test, y_pred, zero_division=0) * 100, 2),
        "f1_score": round(f1_score(y_test, y_pred, zero_division=0) * 100, 2)
    }


def normalize_importance(importance_dict):
    total = sum(abs(v) for v in importance_dict.values())

    if total == 0:
        return {k: 0 for k in importance_dict.keys()}

    return {
        k: round((abs(v) / total) * 100, 4)
        for k, v in importance_dict.items()
    }


def sort_importance(importance_dict):
    return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))


def main():
    print("Dataset okunuyor...")
    df = pd.read_csv(DATASET_PATH, low_memory=False)

    print("Dataset temizleniyor...")
    df, numeric_features = clean_dataset(df)

    if len(df) > SAMPLE_SIZE:
        print(f"Dataset örnekleniyor: {SAMPLE_SIZE} kayıt")
        df = df.sample(n=SAMPLE_SIZE, random_state=42)

    selected_features = numeric_features[:FEATURE_COUNT]

    print(f"Kullanılan feature sayısı: {len(selected_features)}")

    X = df[selected_features]
    y = df["Label"]

    print("Train/Test ayrımı yapılıyor...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("Test dataset kaydediliyor...")
    test_df = X_test.copy()
    test_df["Label"] = y_test.values
    test_df.to_csv(TEST_DATASET_PATH, index=False)

    print("Veri ölçekleniyor...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Random Forest eğitiliyor...")

    base_rf_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )
    base_rf_model.fit(X_train, y_train)

    rf_model = CalibratedClassifierCV(
        estimator=RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced"
        ),
        cv=3,
        method="sigmoid"
    )
    rf_model.fit(X_train, y_train)

    print("KNN eğitiliyor...")

    knn_model = CalibratedClassifierCV(
        estimator=KNeighborsClassifier(
            n_neighbors=11,
            weights="distance",
            n_jobs=-1
        ),
        cv=3,
        method="sigmoid"
    )
    knn_model.fit(X_train_scaled, y_train)

    print("SVM eğitiliyor...")

    svm_model = CalibratedClassifierCV(
        estimator=LinearSVC(
            random_state=42,
            max_iter=5000,
            class_weight="balanced"
        ),
        cv=3,
        method="sigmoid"
    )
    svm_model.fit(X_train_scaled, y_train)

    print("Tahminler alınıyor...")

    rf_pred = rf_model.predict(X_test)
    knn_pred = knn_model.predict(X_test_scaled)
    svm_pred = svm_model.predict(X_test_scaled)

    metrics = {
        "random_forest": calculate_metrics(y_test, rf_pred),
        "knn": calculate_metrics(y_test, knn_pred),
        "svm": calculate_metrics(y_test, svm_pred)
    }

    print("Random Forest feature importance hesaplanıyor...")

    rf_raw_importance = {
        feature: float(value)
        for feature, value in zip(selected_features, base_rf_model.feature_importances_)
    }
    rf_importance = normalize_importance(rf_raw_importance)

    print("KNN permutation importance hesaplanıyor...")

    knn_result = permutation_importance(
        knn_model,
        X_test_scaled,
        y_test,
        n_repeats=3,
        random_state=42,
        n_jobs=-1
    )

    knn_raw_importance = {
        feature: float(value)
        for feature, value in zip(selected_features, knn_result.importances_mean)
    }
    knn_importance = normalize_importance(knn_raw_importance)

    print("SVM permutation importance hesaplanıyor...")

    svm_result = permutation_importance(
        svm_model,
        X_test_scaled,
        y_test,
        n_repeats=3,
        random_state=42,
        n_jobs=-1
    )

    svm_raw_importance = {
        feature: float(value)
        for feature, value in zip(selected_features, svm_result.importances_mean)
    }
    svm_importance = normalize_importance(svm_raw_importance)

    feature_importance = {
        "random_forest": sort_importance(rf_importance),
        "knn": sort_importance(knn_importance),
        "svm": sort_importance(svm_importance)
    }

    print("Model dosyaları kaydediliyor...")

    joblib.dump(rf_model, MODEL_RF_PATH)
    joblib.dump(knn_model, MODEL_KNN_PATH)
    joblib.dump(svm_model, MODEL_SVM_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(selected_features, FEATURES_PATH)

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)

    with open(IMPORTANCE_PATH, "w", encoding="utf-8") as f:
        json.dump(feature_importance, f, indent=4, ensure_ascii=False)

    print("Eğitim tamamlandı.")
    print(json.dumps(metrics, indent=4, ensure_ascii=False))


if __name__ == "__main__":
    main()