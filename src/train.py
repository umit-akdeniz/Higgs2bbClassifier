import numpy as np
import tables
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from keras.metrics import AUC
from src.model import build_residual_model
from src.data_loader import get_features_labels

features, labels = get_features_labels("data/train/ntuple_merged_10.h5", remove_mass_pt_window=False)

scaler = StandardScaler()
features = scaler.fit_transform(features)

X_train, X_val, y_train, y_val = train_test_split(
    features, labels, test_size=0.2, random_state=42, stratify=labels)

model = build_residual_model(input_dim=X_train.shape[1], output_dim=y_train.shape[1])
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy', AUC(name='auc')])

callbacks = [
    EarlyStopping(monitor='val_auc', patience=10, restore_best_weights=True, mode='max'),
    ModelCheckpoint('outputs/keras_model_best.h5', monitor='val_auc', save_best_only=True, mode='max')
]

model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    batch_size=1024,
    epochs=100,
    callbacks=callbacks,
    verbose=1
)
