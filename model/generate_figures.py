import json
from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    precision_recall_curve
)


BASE_DIR = Path(__file__).resolve().parent
FIGURE_DIR = BASE_DIR / "figures"
FIGURE_DIR.mkdir(exist_ok=True)

METRICS_PATH = BASE_DIR / "model_metrics.json"
IMPORTANCE_PATH = BASE_DIR / "feature_importance.json"
TEST_DATASET_PATH = BASE_DIR / "test_dataset.csv"

RF_MODEL_PATH = BASE_DIR / "rf_model.pkl"
KNN_MODEL_PATH = BASE_DIR / "knn_model.pkl"
SVM_MODEL_PATH = BASE_DIR / "svm_model.pkl"
SCALER_PATH = BASE_DIR / "scaler.pkl"
FEATURES_PATH = BASE_DIR / "selected_features.pkl"


def save_metric_charts(metrics):
    models = ["Random Forest", "KNN", "SVM"]
    keys = ["random_forest", "knn", "svm"]

    metric_names = ["accuracy", "precision", "recall", "f1_score"]

    for metric in metric_names:
        values = [metrics[key][metric] for key in keys]

        plt.figure(figsize=(8, 5))
        plt.bar(models, values)
        plt.ylim(80, 100)
        plt.title(f"{metric.replace('_', ' ').title()} Karşılaştırması")
        plt.ylabel("Yüzde (%)")

        for i, v in enumerate(values):
            plt.text(i, v + 0.3, f"{v}%", ha="center")

        plt.tight_layout()
        plt.savefig(FIGURE_DIR / f"{metric}_comparison.png", dpi=300)
        plt.close()


def save_all_metrics_chart(metrics):
    models = ["Random Forest", "KNN", "SVM"]
    keys = ["random_forest", "knn", "svm"]

    data = {
        "Accuracy": [metrics[key]["accuracy"] for key in keys],
        "Precision": [metrics[key]["precision"] for key in keys],
        "Recall": [metrics[key]["recall"] for key in keys],
        "F1-Score": [metrics[key]["f1_score"] for key in keys],
    }

    df = pd.DataFrame(data, index=models)

    ax = df.plot(kind="bar", figsize=(10, 6))
    ax.set_title("Model Performans Metrikleri Karşılaştırması")
    ax.set_ylabel("Yüzde (%)")
    ax.set_ylim(80, 100)
    ax.legend(loc="lower right")

    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "all_metrics_comparison.png", dpi=300)
    plt.close()


def save_feature_importance_charts(importance):
    model_titles = {
        "random_forest": "Random Forest Feature Importance",
        "knn": "KNN Permutation Importance",
        "svm": "SVM Permutation Importance"
    }

    for model_key, title in model_titles.items():
        top_items = list(importance[model_key].items())[:10]

        features = [item[0] for item in top_items]
        values = [item[1] for item in top_items]

        plt.figure(figsize=(10, 6))
        plt.barh(features[::-1], values[::-1])
        plt.title(title)
        plt.xlabel("Katkı Oranı (%)")

        plt.tight_layout()
        plt.savefig(FIGURE_DIR / f"{model_key}_feature_importance.png", dpi=300)
        plt.close()


def save_confusion_matrices(models, X_test, X_test_scaled, y_test):
    for model_name, model in models.items():
        if model_name == "random_forest":
            y_pred = model.predict(X_test)
        else:
            y_pred = model.predict(X_test_scaled)

        cm = confusion_matrix(y_test, y_pred)

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["Normal", "Attack"]
        )

        disp.plot()
        plt.title(f"{model_name.upper()} Confusion Matrix")
        plt.tight_layout()
        plt.savefig(FIGURE_DIR / f"{model_name}_confusion_matrix.png", dpi=300)
        plt.close()


def save_roc_curves(models, X_test, X_test_scaled, y_test):
    plt.figure(figsize=(8, 6))

    for model_name, model in models.items():
        if model_name == "random_forest":
            y_score = model.predict_proba(X_test)[:, 1]
        else:
            y_score = model.predict_proba(X_test_scaled)[:, 1]

        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, label=f"{model_name.upper()} AUC = {roc_auc:.4f}")

    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title("ROC Curve Karşılaştırması")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "roc_curve_comparison.png", dpi=300)
    plt.close()


def save_precision_recall_curves(models, X_test, X_test_scaled, y_test):
    plt.figure(figsize=(8, 6))

    for model_name, model in models.items():
        if model_name == "random_forest":
            y_score = model.predict_proba(X_test)[:, 1]
        else:
            y_score = model.predict_proba(X_test_scaled)[:, 1]

        precision, recall, _ = precision_recall_curve(y_test, y_score)

        plt.plot(recall, precision, label=model_name.upper())

    plt.title("Precision-Recall Curve Karşılaştırması")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="lower left")

    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "precision_recall_curve_comparison.png", dpi=300)
    plt.close()


def main():
    print("Metrikler okunuyor...")
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    print("Feature importance okunuyor...")
    with open(IMPORTANCE_PATH, "r", encoding="utf-8") as f:
        importance = json.load(f)

    print("Grafikler oluşturuluyor...")
    save_metric_charts(metrics)
    save_all_metrics_chart(metrics)
    save_feature_importance_charts(importance)

    print("Modeller yükleniyor...")
    rf_model = joblib.load(RF_MODEL_PATH)
    knn_model = joblib.load(KNN_MODEL_PATH)
    svm_model = joblib.load(SVM_MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    selected_features = joblib.load(FEATURES_PATH)

    models = {
        "random_forest": rf_model,
        "knn": knn_model,
        "svm": svm_model
    }

    print("Test dataset okunuyor...")
    test_df = pd.read_csv(TEST_DATASET_PATH)

    X_test = test_df[selected_features]
    y_test = test_df["Label"]

    X_test_scaled = scaler.transform(X_test)

    print("Confusion matrix oluşturuluyor...")
    save_confusion_matrices(models, X_test, X_test_scaled, y_test)

    print("ROC curve oluşturuluyor...")
    save_roc_curves(models, X_test, X_test_scaled, y_test)

    print("Precision-Recall curve oluşturuluyor...")
    save_precision_recall_curves(models, X_test, X_test_scaled, y_test)

    print("Tüm görseller oluşturuldu.")
    print(f"Klasör: {FIGURE_DIR}")


if __name__ == "__main__":
    main()