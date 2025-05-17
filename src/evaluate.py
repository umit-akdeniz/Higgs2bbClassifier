import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from keras.models import load_model
from src.data_loader import get_features_labels
from sklearn.preprocessing import StandardScaler

X_test, y_test = get_features_labels("data/test/ntuple_merged_0.h5", remove_mass_pt_window=True)
scaler = StandardScaler()
X_test = scaler.fit_transform(X_test)

model = load_model("outputs/keras_model_best.h5")
predictions = model.predict(X_test)

fpr, tpr, _ = roc_curve(y_test[:, 1], predictions[:, 1])
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(tpr, fpr, lw=2.5, label="AUC = {:.2f}%".format(roc_auc * 100))
plt.xlabel('True Positive Rate')
plt.ylabel('False Positive Rate')
plt.semilogy()
plt.ylim(0.001, 1)
plt.xlim(0, 1)
plt.grid(True)
plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig("outputs/ROC_residual.png")
