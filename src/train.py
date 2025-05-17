# src/train.py
import yaml
from keras.callbacks import EarlyStopping, ModelCheckpoint
from src.model import build_residual_model
from src.data_loader import load_multiple_files

# Konfigürasyonu yükle
with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Veriyi yükle
X_train, y_train = load_multiple_files(config["train_files"])

# Modeli oluştur
model = build_residual_model(
    input_dim=X_train.shape[1],
    output_dim=y_train.shape[1],
    dropout_rate=config.get("dropout_rate", 0.3)
)

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Callback'ler
callbacks = [
    EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
    ModelCheckpoint("outputs/keras_model_best.h5", monitor="val_loss", save_best_only=True)
]

# Eğitimi başlat
model.fit(
    X_train,
    y_train,
    epochs=config["epochs"],
    batch_size=config["batch_size"],
    validation_split=0.2,
    callbacks=callbacks,
    shuffle=True
)

# Eğitim tamamlandıktan sonra modeli kaydet
model.save("outputs/final_model.h5")
