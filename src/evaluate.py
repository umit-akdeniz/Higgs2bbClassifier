import yaml
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from keras.models import load_model
from sklearn.preprocessing import StandardScaler

from src.data_loader import get_features_labels
from src.model import build_residual_model

def evaluate_from_config(config_path="configs/config.yaml"):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    X_test, y_test = get_features_labels(config["test_file"])
    scaler = StandardScaler()
    X_test = scaler.fit_transform(X_test)

    model = build_residual_model(input_dim=X_test.shape[1], output_dim=y_test.shape[1])
    model.load_weights("outputs/keras_model_best.h5")

    y_pred = model.predict(X_test)
    fpr, tpr, _ = roc_curve(y_test[:,1], y_pred[:,1])
    roc_auc = auc(fpr, tpr)

    print(f"AUC Score: {roc_auc:.4f}")

    plt.figure()
    plt.plot(tpr, fpr, label=f"AUC = {roc_auc:.2%}")
    plt.xlabel("True Positive Rate")
    plt.ylabel("False Positive Rate")
    plt.semilogy()
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/ROC.png")
    plt.show()

if __name__ == "__main__":
    evaluate_from_config()
