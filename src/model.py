from keras.models import Model
from keras.layers import Input, Dense, BatchNormalization, Dropout, Add, Activation

def build_residual_model(input_dim, output_dim, dropout_rate=0.3):
    inputs = Input(shape=(input_dim,), name='input')
    x = BatchNormalization()(inputs)
    x = Dense(128, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    res = Dense(128, activation='relu')(x)
    res = Dropout(dropout_rate)(res)
    x = Add()([x, res])
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    res = Dense(128, activation='relu')(x)
    res = Dropout(dropout_rate)(res)
    x = Add()([x, res])
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(output_dim, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model
