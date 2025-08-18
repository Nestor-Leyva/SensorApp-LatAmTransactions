from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import json
import pandas as pd
import numpy as np

file_path_sintetizado = "path"

with open(file_path_sintetizado, "r", encoding="utf-8") as file:
    data_sintetizado = json.load(file)

def calcular_metrica(sensor_data):
    if sensor_data:
        valores_x = [d["x"] for d in sensor_data]
        valores_y = [d["y"] for d in sensor_data]
        valores_z = [d["z"] for d in sensor_data]
        return {
            "mean_x": np.mean(valores_x), "std_x": np.std(valores_x),
            "mean_y": np.mean(valores_y), "std_y": np.std(valores_y),
            "mean_z": np.mean(valores_z), "std_z": np.std(valores_z),
        }
    return {"mean_x": np.nan, "std_x": np.nan, "mean_y": np.nan, "std_y": np.nan, "mean_z": np.nan, "std_z": np.nan}
usuarios_procesados = []

for usuario in data_sintetizado:
    user_data = {"userId": usuario["userId"], "name": usuario["name"]}
    accel_metrics = calcular_metrica(usuario.get("accel", []))
    gyro_metrics = calcular_metrica(usuario.get("gyro", []))
    
    for key, value in accel_metrics.items():
        user_data[f"accel_{key}"] = value
    for key, value in gyro_metrics.items():
        user_data[f"gyro_{key}"] = value
    
    ocean = usuario.get("OCEAN", {})
    user_data["Openness"] = round(ocean.get("Openness", np.nan), 1)
    user_data["Conscientiousness"] = round(ocean.get("Conscientiousness", np.nan), 1)
    user_data["Extraversion"] = round(ocean.get("Extraversion", np.nan), 1)
    user_data["Agreeableness"] = round(ocean.get("Agreeableness", np.nan), 1)
    user_data["Neuroticism"] = round(ocean.get("Neuroticism", np.nan), 1)

    usuarios_procesados.append(user_data)

df_sintetizado = pd.DataFrame(usuarios_procesados)

columnas_numericas = df_sintetizado.select_dtypes(include=[np.number]).columns
df_sintetizado[columnas_numericas] = df_sintetizado[columnas_numericas].fillna(df_sintetizado[columnas_numericas].mean())
columnas_sensores = [col for col in df_sintetizado.columns if "accel_" in col or "gyro_" in col]
columnas_target = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]

X = df_sintetizado[columnas_sensores].values
y = df_sintetizado[columnas_target].values

scaler = StandardScaler()
X = scaler.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)