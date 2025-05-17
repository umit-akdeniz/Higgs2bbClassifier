# src/train.py

import yaml
import os
from datetime import datetime
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.optimizers import Adam

from src.data_loader import get_features_labels
from src.model import build_residual_model

def train_from_config(config_path="configs/config.yaml"):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    X, y = get_features_labels(config["train_file"], remove_mass_pt_window=False)

    # Normalize
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model = build_residual_model(
        input_dim=X.shape[1],
        output_dim=y.shape[1],
        dropout_rate=config.get("dropout_rate", 0.3)
    )

    model.compile(
        optimizer=Adam(learning_rate=config["learning_rate"]),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # TensorBoard ve diğer callback'ler
    log_dir = os.path.join("logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ModelCheckpoint('outputs/keras_model_best.h5', monitor='val_loss', save_best_only=True),
        TensorBoard(log_dir=log_dir)
    ]

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        callbacks=callbacks,
        verbose=1
    )

if __name__ == "__main__":
    train_from_config()
