
# Pruebas Redes (IPIP - Dominio de tiempo-Frecuencia)

## MLP
X = df_completo_proc.drop(columns=["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]).values
y = df_completo_proc[["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]].values
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = keras.Sequential([
    layers.Dense(64, activation="relu", input_shape=(X_train.shape[1],)),
    layers.Dense(32, activation="relu"),
    layers.Dense(5, activation="linear")
])
model.compile(optimizer="adam", loss="mse", metrics=["mae"])
history = model.fit(X_train, y_train, epochs=100, batch_size=8, validation_data=(X_test, y_test), verbose=1)
test_loss, test_mae = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}, Test MAE: {test_mae}")
X = df_completo_proc.drop(columns=["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]).values
y = df_completo_proc[["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]].values
scaler = StandardScaler()
X = scaler.fit_transform(X)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
mae_scores = []
mse_scores = []

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    clear_session()
    model = keras.Sequential([
        layers.Dense(64, activation="relu", input_shape=(X_train.shape[1],)),
        layers.Dense(32, activation="relu"),
        layers.Dense(5, activation="linear")
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    history = model.fit(X_train, y_train, epochs=100, batch_size=8, verbose=1)
    loss, mae = model.evaluate(X_test, y_test, verbose=0)
    mse_scores.append(loss)
    mae_scores.append(mae)


"""## CNN"""
X = df_completo_proc.drop(columns=["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]).values
y = df_completo_proc[["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]].values
scaler = StandardScaler()
X = scaler.fit_transform(X)
X = X.reshape(X.shape[0], X.shape[1], 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = keras.Sequential([
    layers.Conv1D(64, 3, activation="relu", input_shape=(X_train.shape[1], 1)),
    layers.MaxPooling1D(2),    layers.Conv1D(32, 3, activation="relu"),
    layers.MaxPooling1D(2),    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dense(5, activation="linear")
])
model.compile(optimizer="adam", loss="mse", metrics=["mae"])
history = model.fit(X_train, y_train, epochs=100, batch_size=8, validation_data=(X_test, y_test), verbose=1)
test_loss, test_mae = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}, Test MAE: {test_mae}")


X = df_completo_proc.drop(columns=["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]).values
y = df_completo_proc[["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]].values
scaler = StandardScaler()
X = scaler.fit_transform(X)
X = X.reshape(X.shape[0], X.shape[1], 1)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
mae_scores = []
mse_scores = []

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    clear_session()
    model = keras.Sequential([
        layers.Conv1D(64, 3, activation="relu", input_shape=(X_train.shape[1], 1)),
        layers.MaxPooling1D(2),
        layers.Conv1D(32, 3, activation="relu"),
        layers.MaxPooling1D(2),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dense(5, activation="linear")
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    history = model.fit(X_train, y_train, epochs=100, batch_size=8, verbose=1, validation_data=(X_test, y_test))
    loss, mae = model.evaluate(X_test, y_test, verbose=0)
    mse_scores.append(loss)
    mae_scores.append(mae)


"""## LSTM"""
X = df_completo_proc.drop(columns=["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]).values
y = df_completo_proc[["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]].values
scaler = StandardScaler()
X = scaler.fit_transform(X)
X = X.reshape(X.shape[0], 1, X.shape[1])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = keras.Sequential([
    layers.LSTM(64, activation='tanh', input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False),
    layers.Dense(32, activation="relu"),
    layers.Dense(5, activation="linear")
])

model.compile(optimizer="adam", loss="mse", metrics=["mae"])
history = model.fit(X_train, y_train, epochs=100, batch_size=8, validation_data=(X_test, y_test), verbose=1)
test_loss, test_mae = model.evaluate(X_test, y_test)

X = df_completo_proc.drop(columns=["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]).values
y = df_completo_proc[["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]].values
scaler = StandardScaler()
X = scaler.fit_transform(X)
X = X.reshape(X.shape[0], 1, X.shape[1])
kf = KFold(n_splits=5, shuffle=True, random_state=42)
mae_scores = []
mse_scores = []

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    clear_session()
    model = keras.Sequential([
        layers.LSTM(64, activation='tanh', input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False),
        layers.Dense(32, activation="relu"),
        layers.Dense(5, activation="linear")
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    history = model.fit(X_train, y_train, epochs=100, batch_size=8, verbose=1, validation_data=(X_test, y_test))
    loss, mae = model.evaluate(X_test, y_test, verbose=0)
    mse_scores.append(loss)
    mae_scores.append(mae)


"""## Transformer"""
X = df_completo_proc.drop(columns=["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]).values
y = df_completo_proc[["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]].values
scaler = StandardScaler()
X = scaler.fit_transform(X)
X = X.reshape(X.shape[0], 1, X.shape[1])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
def transformer_encoder(inputs, head_size, num_heads, ff_dim):
    attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=head_size)(inputs, inputs)
    attention = layers.LayerNormalization()(attention)
    ffn = layers.Dense(ff_dim, activation="relu")(attention)
    ffn = layers.Dense(head_size)(ffn)
    ffn = layers.LayerNormalization()(ffn)
    return ffn
inputs = layers.Input(shape=(X_train.shape[1], X_train.shape[2]))
x = transformer_encoder(inputs=inputs, head_size=64, num_heads=4, ff_dim=128)
x = layers.GlobalAveragePooling1D()(x)
x = layers.Dense(32, activation="relu")(x)
outputs = layers.Dense(5, activation="linear")(x)

model = keras.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer="adam", loss="mse", metrics=["mae"])
history = model.fit(X_train, y_train, epochs=100, batch_size=8, validation_data=(X_test, y_test), verbose=1)
test_loss, test_mae = model.evaluate(X_test, y_test)

X = df_completo_proc.drop(columns=["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]).values
y = df_completo_proc[["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]].values
scaler = StandardScaler()
X = scaler.fit_transform(X)
X = X.reshape(X.shape[0], 1, X.shape[1])
kf = KFold(n_splits=5, shuffle=True, random_state=42)
mae_scores = []
mse_scores = []

def transformer_encoder(inputs, head_size, num_heads, ff_dim):
    attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=head_size)(inputs, inputs)
    attention = layers.LayerNormalization()(attention)
    ffn = layers.Dense(ff_dim, activation="relu")(attention)
    ffn = layers.Dense(head_size)(ffn)
    ffn = layers.LayerNormalization()(ffn)
    return ffn
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    clear_session()
    inputs = layers.Input(shape=(X_train.shape[1], X_train.shape[2]))
    x = transformer_encoder(inputs=inputs, head_size=64, num_heads=4, ff_dim=128)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(32, activation="relu")(x)
    outputs = layers.Dense(5, activation="linear")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    history = model.fit(X_train, y_train, epochs=100, batch_size=8, verbose=1, validation_data=(X_test, y_test))
    loss, mae = model.evaluate(X_test, y_test, verbose=0)
    mse_scores.append(loss)
    mae_scores.append(mae)