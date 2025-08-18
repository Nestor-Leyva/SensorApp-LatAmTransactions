import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

"""# MLP"""
model = keras.Sequential([
    layers.Dense(32, activation="relu", input_shape=(X.shape[1],)),
    layers.Dense(16, activation="relu"),
    layers.Dense(5, activation="linear")
])
model.compile(optimizer="adam", loss="mse", metrics=["mae"])
history = model.fit(X_train, y_train, epochs=100, batch_size=8, validation_data=(X_test, y_test), verbose=1)
test_loss, test_mae = model.evaluate(X_test, y_test)
test_loss, test_mae


"""# CNN"""
X_cnn = X.reshape(X.shape[0], X.shape[1], 1)
model_cnn = keras.Sequential([
    layers.Conv1D(filters=32, kernel_size=3, activation="relu", input_shape=(X_cnn.shape[1], 1)),
    layers.MaxPooling1D(pool_size=2),
    layers.Conv1D(filters=64, kernel_size=3, activation="relu"),
    layers.MaxPooling1D(pool_size=2),
    layers.Flatten(),
    layers.Dense(32, activation="relu"),
    layers.Dense(16, activation="relu"),
    layers.Dense(5, activation="linear")
])
model_cnn.compile(optimizer="adam", loss="mse", metrics=["mae"])
history_cnn = model_cnn.fit(X_cnn, y, epochs=100, batch_size=8, validation_split=0.2, verbose=1)
test_loss_cnn, test_mae_cnn = model_cnn.evaluate(X_cnn, y)
test_loss_cnn, test_mae_cnn


"""# LSTM"""
X_lstm = X.reshape(X.shape[0], 1, X.shape[1])
model_lstm = keras.Sequential([
    layers.LSTM(64, return_sequences=True, input_shape=(X_lstm.shape[1], X_lstm.shape[2])),
    layers.LSTM(32, return_sequences=False),
    layers.Dense(16, activation="relu"),
    layers.Dense(5, activation="linear")
])
model_lstm.compile(optimizer="adam", loss="mse", metrics=["mae"])
history_lstm = model_lstm.fit(X_lstm, y, epochs=100, batch_size=8, validation_split=0.2, verbose=1)
test_loss_lstm, test_mae_lstm = model_lstm.evaluate(X_lstm, y)
test_loss_lstm, test_mae_lstm


"""# Transformer"""
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim)
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
X_transformer = X.reshape(X.shape[0], 1, X.shape[1])
embed_dim = X_transformer.shape[2]
num_heads = 4
ff_dim = 32

inputs = layers.Input(shape=(1, embed_dim))
transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
x = transformer_block(inputs)
x = layers.GlobalAveragePooling1D()(x)
x = layers.Dense(32, activation="relu")(x)
x = layers.Dense(16, activation="relu")(x)
outputs = layers.Dense(5, activation="linear")(x)

model_transformer = keras.Model(inputs=inputs, outputs=outputs)
model_transformer.compile(optimizer="adam", loss="mse", metrics=["mae"])

history_transformer = model_transformer.fit(X_transformer, y, epochs=100, batch_size=8, validation_split=0.2, verbose=1)

test_loss_transformer, test_mae_transformer = model_transformer.evaluate(X_transformer, y)
test_loss_transformer, test_mae_transformer