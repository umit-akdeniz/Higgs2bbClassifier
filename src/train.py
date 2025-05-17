import yaml
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.optimizers import Adam
from keras.metrics import AUC
from datetime import datetime
import os

from src.model import build_residual_model
from src.data_loader import get_features_labels

# YAML'den parametreleri oku
with open('configs/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

batch_size = config['batch_size']
epochs = config['epochs']
learning_rate = config['learning_rate']
dropout_rate = config['dropout_rate']
train_file = config['train_file']

features, labels = get_features_labels(train_file, remove_mass_pt_window=False)

scaler = StandardScaler()
features = scaler.fit_transform(features)

X_train, X_val, y_train, y_val = train_test_split(
    features, labels, test_size=0.2, random_state=42, stratify=labels)

model = build_residual_model(input_dim=X_train.shape[1], output_dim=y_train.shape[1], dropout_rate=dropout_rate)
model.compile(optimizer=Adam(learning_rate=learning_rate),
              loss='categorical_crossentropy',
              metrics=['accuracy', AUC(name='auc')])

log_dir = os.path.join("logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
callbacks = [
    EarlyStopping(monitor='val_auc', patience=10, restore_best_weights=True, mode='max'),
    ModelCheckpoint('outputs/keras_model_best.h5', monitor='val_auc', save_best_only=True, mode='max'),
    TensorBoard(log_dir=log_dir, histogram_freq=1)
]

model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    batch_size=batch_size,
    epochs=epochs,
    callbacks=callbacks,
    verbose=1
)
