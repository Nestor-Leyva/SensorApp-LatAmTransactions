"""# Cálculo de métricas estadísticas (Features)"""

## Métricas en dominio de tiempo

### Varianza
def calcular_varianza(sensor_data):
    if not sensor_data:
        return {"variance_x": np.nan, "variance_y": np.nan, "variance_z": np.nan}
    valores_x = [d["x"] for d in sensor_data]
    valores_y = [d["y"] for d in sensor_data]
    valores_z = [d["z"] for d in sensor_data]
    variance_x = np.var(valores_x)
    variance_y = np.var(valores_y)
    variance_z = np.var(valores_z)
    return {
        "variance_x": variance_x,
        "variance_y": variance_y,
        "variance_z": variance_z
    }
usuarios_procesados_varianza = []

for usuario in data_sintetizado:
    user_data = {"userId": usuario["userId"], "name": usuario["name"]}
    accel_variance = calcular_varianza(usuario.get("accel", []))
    gyro_variance = calcular_varianza(usuario.get("gyro", []))
    for key, value in accel_variance.items():
        user_data[f"accel_{key}"] = value
    for key, value in gyro_variance.items():
        user_data[f"gyro_{key}"] = value
    usuarios_procesados_varianza.append(user_data)
df_varianza = pd.DataFrame(usuarios_procesados_varianza)


### Rango intercuartil
def calcular_iqr(sensor_data):
    if not sensor_data:
        return {"iqr_x": np.nan, "iqr_y": np.nan, "iqr_z": np.nan}
    valores_x = [d["x"] for d in sensor_data]
    valores_y = [d["y"] for d in sensor_data]
    valores_z = [d["z"] for d in sensor_data]
    iqr_x = np.percentile(valores_x, 75) - np.percentile(valores_x, 25)
    iqr_y = np.percentile(valores_y, 75) - np.percentile(valores_y, 25)
    iqr_z = np.percentile(valores_z, 75) - np.percentile(valores_z, 25)
    return {
        "iqr_x": iqr_x,
        "iqr_y": iqr_y,
        "iqr_z": iqr_z
    }
usuarios_procesados_iqr = []

for usuario in data_sintetizado:
    user_data = {"userId": usuario["userId"], "name": usuario["name"]}
    accel_iqr = calcular_iqr(usuario.get("accel", []))
    gyro_iqr = calcular_iqr(usuario.get("gyro", []))
    for key, value in accel_iqr.items():
        user_data[f"accel_{key}"] = value
    for key, value in gyro_iqr.items():
        user_data[f"gyro_{key}"] = value

    usuarios_procesados_iqr.append(user_data)
df_iqr = pd.DataFrame(usuarios_procesados_iqr)


### Media Cuadrática (RMS)
def calcular_rms(sensor_data):
    if not sensor_data:
        return {"rms_x": np.nan, "rms_y": np.nan, "rms_z": np.nan}
    valores_x = [d["x"] for d in sensor_data]
    valores_y = [d["y"] for d in sensor_data]
    valores_z = [d["z"] for d in sensor_data]
    rms_x = np.sqrt(np.mean(np.square(valores_x)))
    rms_y = np.sqrt(np.mean(np.square(valores_y)))
    rms_z = np.sqrt(np.mean(np.square(valores_z)))
    return {
        "rms_x": rms_x,
        "rms_y": rms_y,
        "rms_z": rms_z
    }
usuarios_procesados_rms = []

for usuario in data_sintetizado:
    user_data = {"userId": usuario["userId"], "name": usuario["name"]}
    accel_rms = calcular_rms(usuario.get("accel", []))
    gyro_rms = calcular_rms(usuario.get("gyro", []))
    for key, value in accel_rms.items():
        user_data[f"accel_{key}"] = value
    for key, value in gyro_rms.items():
        user_data[f"gyro_{key}"] = value
    usuarios_procesados_rms.append(user_data)
df_rms = pd.DataFrame(usuarios_procesados_rms)


### Entropía
def calcular_entropia(sensor_data):
    if not sensor_data:
        return {"entropy_x": np.nan, "entropy_y": np.nan, "entropy_z": np.nan}
    valores_x = [d["x"] for d in sensor_data]
    valores_y = [d["y"] for d in sensor_data]
    valores_z = [d["z"] for d in sensor_data]
    def calcular_entropia_vector(valores):
        _, counts = np.unique(valores, return_counts=True)
        probabilidad = counts / len(valores)
        return -np.sum(probabilidad * np.log2(probabilidad))
    entropy_x = calcular_entropia_vector(valores_x)
    entropy_y = calcular_entropia_vector(valores_y)
    entropy_z = calcular_entropia_vector(valores_z)
    return {
        "entropy_x": entropy_x,
        "entropy_y": entropy_y,
        "entropy_z": entropy_z
    }
usuarios_procesados_entropia = []

for usuario in data_sintetizado:
    user_data = {"userId": usuario["userId"], "name": usuario["name"]}
    accel_entropy = calcular_entropia(usuario.get("accel", []))
    gyro_entropy = calcular_entropia(usuario.get("gyro", []))
    for key, value in accel_entropy.items():
        user_data[f"accel_{key}"] = value
    for key, value in gyro_entropy.items():
        user_data[f"gyro_{key}"] = value
    usuarios_procesados_entropia.append(user_data)
df_entropia = pd.DataFrame(usuarios_procesados_entropia)


"""### Dataframe Dominio de tiempo"""

import numpy as np
import pandas as pd

def calcular_varianza(sensor_data):
    if not sensor_data:
        return {"variance_x": np.nan, "variance_y": np.nan, "variance_z": np.nan}
    valores_x = [d["x"] for d in sensor_data]
    valores_y = [d["y"] for d in sensor_data]
    valores_z = [d["z"] for d in sensor_data]
    return {
        "variance_x": np.var(valores_x),
        "variance_y": np.var(valores_y),
        "variance_z": np.var(valores_z)
    }
def calcular_iqr(sensor_data):
    if not sensor_data:
        return {"iqr_x": np.nan, "iqr_y": np.nan, "iqr_z": np.nan}
    valores_x = [d["x"] for d in sensor_data]
    valores_y = [d["y"] for d in sensor_data]
    valores_z = [d["z"] for d in sensor_data]
    return {
        "iqr_x": np.percentile(valores_x, 75) - np.percentile(valores_x, 25),
        "iqr_y": np.percentile(valores_y, 75) - np.percentile(valores_y, 25),
        "iqr_z": np.percentile(valores_z, 75) - np.percentile(valores_z, 25)
    }
def calcular_rms(sensor_data):
    if not sensor_data:
        return {"rms_x": np.nan, "rms_y": np.nan, "rms_z": np.nan}
    valores_x = [d["x"] for d in sensor_data]
    valores_y = [d["y"] for d in sensor_data]
    valores_z = [d["z"] for d in sensor_data]
    return {
        "rms_x": np.sqrt(np.mean(np.square(valores_x))),
        "rms_y": np.sqrt(np.mean(np.square(valores_y))),
        "rms_z": np.sqrt(np.mean(np.square(valores_z)))
    }
def calcular_entropia(sensor_data):
    if not sensor_data:
        return {"entropy_x": np.nan, "entropy_y": np.nan, "entropy_z": np.nan}
    valores_x = [d["x"] for d in sensor_data]
    valores_y = [d["y"] for d in sensor_data]
    valores_z = [d["z"] for d in sensor_data]
    def calcular_entropia_vector(valores):
        _, counts = np.unique(valores, return_counts=True)
        probabilidad = counts / len(valores)
        return -np.sum(probabilidad * np.log2(probabilidad))
    return {
        "entropy_x": calcular_entropia_vector(valores_x),
        "entropy_y": calcular_entropia_vector(valores_y),
        "entropy_z": calcular_entropia_vector(valores_z)
    }
usuarios_procesados = []

for usuario in data_sintetizado:
    user_data = {"userId": usuario["userId"], "name": usuario["name"]}
    accel_variance = calcular_varianza(usuario.get("accel", []))
    gyro_variance = calcular_varianza(usuario.get("gyro", []))
    accel_iqr = calcular_iqr(usuario.get("accel", []))
    gyro_iqr = calcular_iqr(usuario.get("gyro", []))
    accel_rms = calcular_rms(usuario.get("accel", []))
    gyro_rms = calcular_rms(usuario.get("gyro", []))
    accel_entropy = calcular_entropia(usuario.get("accel", []))
    gyro_entropy = calcular_entropia(usuario.get("gyro", []))
    for key, value in accel_variance.items():
        user_data[f"accel_{key}"] = value
    for key, value in gyro_variance.items():
        user_data[f"gyro_{key}"] = value
    for key, value in accel_iqr.items():
        user_data[f"accel_{key}"] = value
    for key, value in gyro_iqr.items():
        user_data[f"gyro_{key}"] = value
    for key, value in accel_rms.items():
        user_data[f"accel_{key}"] = value
    for key, value in gyro_rms.items():
        user_data[f"gyro_{key}"] = value
    for key, value in accel_entropy.items():
        user_data[f"accel_{key}"] = value
    for key, value in gyro_entropy.items():
        user_data[f"gyro_{key}"] = value
    usuarios_procesados.append(user_data)
df_metricas_tiempo = pd.DataFrame(usuarios_procesados)
df_metricas_tiempo


"""## Métricas en dominio de frecuencia"""


### Transformada rápida de Fourier discreta (DFFT)
def calcular_dfft(sensor_data):
    if not sensor_data:
        return {"dfft_x": np.nan, "dfft_y": np.nan, "dfft_z": np.nan}
    valores_x = [d["x"] for d in sensor_data]
    valores_y = [d["y"] for d in sensor_data]
    valores_z = [d["z"] for d in sensor_data]
    dfft_x = np.abs(np.fft.fft(valores_x))[:len(valores_x)//2]
    dfft_y = np.abs(np.fft.fft(valores_y))[:len(valores_y)//2]
    dfft_z = np.abs(np.fft.fft(valores_z))[:len(valores_z)//2]
    dfft_x_max = np.max(dfft_x) if dfft_x.size > 0 else np.nan
    dfft_y_max = np.max(dfft_y) if dfft_y.size > 0 else np.nan
    dfft_z_max = np.max(dfft_z) if dfft_z.size > 0 else np.nan
    return {
        "dfft_x": dfft_x_max,
        "dfft_y": dfft_y_max,
        "dfft_z": dfft_z_max
    }
usuarios_procesados_dfft = []

for usuario in data_sintetizado:
    user_data = {"userId": usuario["userId"], "name": usuario["name"]}
    accel_dfft = calcular_dfft(usuario.get("accel", []))
    gyro_dfft = calcular_dfft(usuario.get("gyro", []))
    for key, value in accel_dfft.items():
        user_data[f"accel_{key}"] = value
    for key, value in gyro_dfft.items():
        user_data[f"gyro_{key}"] = value
    usuarios_procesados_dfft.append(user_data)
df_dfft = pd.DataFrame(usuarios_procesados_dfft)


"""### Transformada discreta del coseno (DCT)"""
from scipy.fftpack import dct
def calcular_dct(sensor_data):
    if not sensor_data:
        return {"dct_x": np.nan, "dct_y": np.nan, "dct_z": np.nan}
    valores_x = [d["x"] for d in sensor_data]
    valores_y = [d["y"] for d in sensor_data]
    valores_z = [d["z"] for d in sensor_data]
    dct_x = np.abs(dct(valores_x))[:len(valores_x)//2]
    dct_y = np.abs(dct(valores_y))[:len(valores_y)//2]
    dct_z = np.abs(dct(valores_z))[:len(valores_z)//2]
    dct_x_max = np.max(dct_x) if dct_x.size > 0 else np.nan
    dct_y_max = np.max(dct_y) if dct_y.size > 0 else np.nan
    dct_z_max = np.max(dct_z) if dct_z.size > 0 else np.nan
    return {
        "dct_x": dct_x_max,
        "dct_y": dct_y_max,
        "dct_z": dct_z_max
    }
usuarios_procesados_dct = []

for usuario in data_sintetizado:
    user_data = {"userId": usuario["userId"], "name": usuario["name"]}
    accel_dct = calcular_dct(usuario.get("accel", []))
    gyro_dct = calcular_dct(usuario.get("gyro", []))
    for key, value in accel_dct.items():
        user_data[f"accel_{key}"] = value
    for key, value in gyro_dct.items():
        user_data[f"gyro_{key}"] = value
    usuarios_procesados_dct.append(user_data)
df_dct = pd.DataFrame(usuarios_procesados_dct)


"""### Energía"""
def calcular_energia(sensor_data):
    if not sensor_data:
        return {"energia_x": np.nan, "energia_y": np.nan, "energia_z": np.nan}
    valores_x = [d["x"] for d in sensor_data]
    valores_y = [d["y"] for d in sensor_data]
    valores_z = [d["z"] for d in sensor_data]
    energia_x = np.sum(np.square(valores_x))
    energia_y = np.sum(np.square(valores_y))
    energia_z = np.sum(np.square(valores_z))
    return {
        "energia_x": energia_x,
        "energia_y": energia_y,
        "energia_z": energia_z
    }
usuarios_procesados_energia = []

for usuario in data_sintetizado:
    user_data = {"userId": usuario["userId"], "name": usuario["name"]}
    accel_energia = calcular_energia(usuario.get("accel", []))
    gyro_energia = calcular_energia(usuario.get("gyro", []))
    for key, value in accel_energia.items():
        user_data[f"accel_{key}"] = value
    for key, value in gyro_energia.items():
        user_data[f"gyro_{key}"] = value
    usuarios_procesados_energia.append(user_data)
df_energia = pd.DataFrame(usuarios_procesados_energia)


"""### Rango de frecuencias de potencia"""
def calcular_rango_frecuencia(sensor_data):
    if not sensor_data:
        return {"freq_range_x": np.nan, "freq_range_y": np.nan, "freq_range_z": np.nan}
    valores_x = [d["x"] for d in sensor_data]
    valores_y = [d["y"] for d in sensor_data]
    valores_z = [d["z"] for d in sensor_data]
    dfft_x = np.abs(np.fft.fft(valores_x))[:len(valores_x)//2]
    dfft_y = np.abs(np.fft.fft(valores_y))[:len(valores_y)//2]
    dfft_z = np.abs(np.fft.fft(valores_z))[:len(valores_z)//2]
    freq_range_x = np.max(dfft_x) - np.min(dfft_x) if dfft_x.size > 0 else np.nan
    freq_range_y = np.max(dfft_y) - np.min(dfft_y) if dfft_y.size > 0 else np.nan
    freq_range_z = np.max(dfft_z) - np.min(dfft_z) if dfft_z.size > 0 else np.nan
    return {
        "freq_range_x": freq_range_x,
        "freq_range_y": freq_range_y,
        "freq_range_z": freq_range_z
    }
usuarios_procesados_freq_range = []

for usuario in data_sintetizado:
    user_data = {"userId": usuario["userId"], "name": usuario["name"]}
    accel_freq_range = calcular_rango_frecuencia(usuario.get("accel", []))
    gyro_freq_range = calcular_rango_frecuencia(usuario.get("gyro", []))
    for key, value in accel_freq_range.items():
        user_data[f"accel_{key}"] = value
    for key, value in gyro_freq_range.items():
        user_data[f"gyro_{key}"] = value
    usuarios_procesados_freq_range.append(user_data)
df_freq_range = pd.DataFrame(usuarios_procesados_freq_range)


"""### Dataframe Dominio de frecuencia"""

import numpy as np
import pandas as pd
from scipy.fftpack import dct

def calcular_dfft(sensor_data):
    if not sensor_data:
        return {"dfft_x": np.nan, "dfft_y": np.nan, "dfft_z": np.nan}
    valores_x = [d["x"] for d in sensor_data]
    valores_y = [d["y"] for d in sensor_data]
    valores_z = [d["z"] for d in sensor_data]
    dfft_x = np.abs(np.fft.fft(valores_x))[:len(valores_x)//2]
    dfft_y = np.abs(np.fft.fft(valores_y))[:len(valores_y)//2]
    dfft_z = np.abs(np.fft.fft(valores_z))[:len(valores_z)//2]
    dfft_x_max = np.max(dfft_x) if dfft_x.size > 0 else np.nan
    dfft_y_max = np.max(dfft_y) if dfft_y.size > 0 else np.nan
    dfft_z_max = np.max(dfft_z) if dfft_z.size > 0 else np.nan

    return {
        "dfft_x": dfft_x_max,
        "dfft_y": dfft_y_max,
        "dfft_z": dfft_z_max
    }
def calcular_dct(sensor_data):
    if not sensor_data:
        return {"dct_x": np.nan, "dct_y": np.nan, "dct_z": np.nan}
    valores_x = [d["x"] for d in sensor_data]
    valores_y = [d["y"] for d in sensor_data]
    valores_z = [d["z"] for d in sensor_data]
    dct_x = np.abs(dct(valores_x))[:len(valores_x)//2]
    dct_y = np.abs(dct(valores_y))[:len(valores_y)//2]
    dct_z = np.abs(dct(valores_z))[:len(valores_z)//2]
    dct_x_max = np.max(dct_x) if dct_x.size > 0 else np.nan
    dct_y_max = np.max(dct_y) if dct_y.size > 0 else np.nan
    dct_z_max = np.max(dct_z) if dct_z.size > 0 else np.nan
    return {
        "dct_x": dct_x_max,
        "dct_y": dct_y_max,
        "dct_z": dct_z_max
    }
def calcular_energia(sensor_data):
    if not sensor_data:
        return {"energia_x": np.nan, "energia_y": np.nan, "energia_z": np.nan}
    valores_x = [d["x"] for d in sensor_data]
    valores_y = [d["y"] for d in sensor_data]
    valores_z = [d["z"] for d in sensor_data]
    energia_x = np.sum(np.square(valores_x))
    energia_y = np.sum(np.square(valores_y))
    energia_z = np.sum(np.square(valores_z))
    return {
        "energia_x": energia_x,
        "energia_y": energia_y,
        "energia_z": energia_z
    }
def calcular_rango_frecuencia(sensor_data):
    if not sensor_data:
        return {"freq_range_x": np.nan, "freq_range_y": np.nan, "freq_range_z": np.nan}
    valores_x = [d["x"] for d in sensor_data]
    valores_y = [d["y"] for d in sensor_data]
    valores_z = [d["z"] for d in sensor_data]
    dfft_x = np.abs(np.fft.fft(valores_x))[:len(valores_x)//2]
    dfft_y = np.abs(np.fft.fft(valores_y))[:len(valores_y)//2]
    dfft_z = np.abs(np.fft.fft(valores_z))[:len(valores_z)//2]
    freq_range_x = np.max(dfft_x) - np.min(dfft_x) if dfft_x.size > 0 else np.nan
    freq_range_y = np.max(dfft_y) - np.min(dfft_y) if dfft_y.size > 0 else np.nan
    freq_range_z = np.max(dfft_z) - np.min(dfft_z) if dfft_z.size > 0 else np.nan
    return {
        "freq_range_x": freq_range_x,
        "freq_range_y": freq_range_y,
        "freq_range_z": freq_range_z
    }
usuarios_procesados_frecuencia = []

for usuario in data_sintetizado:
    user_data = {"userId": usuario["userId"], "name": usuario["name"]}
    accel_dfft = calcular_dfft(usuario.get("accel", []))
    gyro_dfft = calcular_dfft(usuario.get("gyro", []))
    accel_dct = calcular_dct(usuario.get("accel", []))
    gyro_dct = calcular_dct(usuario.get("gyro", []))
    accel_energia = calcular_energia(usuario.get("accel", []))
    gyro_energia = calcular_energia(usuario.get("gyro", []))
    accel_freq_range = calcular_rango_frecuencia(usuario.get("accel", []))
    gyro_freq_range = calcular_rango_frecuencia(usuario.get("gyro", []))
    for key, value in accel_dfft.items():
        user_data[f"accel_{key}"] = value
    for key, value in gyro_dfft.items():
        user_data[f"gyro_{key}"] = value
    for key, value in accel_dct.items():
        user_data[f"accel_{key}"] = value
    for key, value in gyro_dct.items():
        user_data[f"gyro_{key}"] = value
    for key, value in accel_energia.items():
        user_data[f"accel_{key}"] = value
    for key, value in gyro_energia.items():
        user_data[f"gyro_{key}"] = value
    for key, value in accel_freq_range.items():
        user_data[f"accel_{key}"] = value
    for key, value in gyro_freq_range.items():
        user_data[f"gyro_{key}"] = value
    usuarios_procesados_frecuencia.append(user_data)
df_metricas_frecuencia = pd.DataFrame(usuarios_procesados_frecuencia)
df_metricas_frecuencia


"""## Otros"""

### Coeficiente de relación de ejes
def calcular_coeficiente_relacion(sensor_data):
    if not sensor_data:
        return {"coef_x_y": np.nan, "coef_x_z": np.nan, "coef_y_z": np.nan}
    valores_x = [d["x"] for d in sensor_data]
    valores_y = [d["y"] for d in sensor_data]
    valores_z = [d["z"] for d in sensor_data]
    coef_x_y = np.mean(np.divide(valores_x, valores_y)) if np.all(valores_y != 0) else np.nan
    coef_x_z = np.mean(np.divide(valores_x, valores_z)) if np.all(valores_z != 0) else np.nan
    coef_y_z = np.mean(np.divide(valores_y, valores_z)) if np.all(valores_z != 0) else np.nan
    return {
        "coef_x_y": coef_x_y,
        "coef_x_z": coef_x_z,
        "coef_y_z": coef_y_z
    }
usuarios_procesados_coef = []

for usuario in data_sintetizado:
    user_data = {"userId": usuario["userId"], "name": usuario["name"]}
    accel_coef = calcular_coeficiente_relacion(usuario.get("accel", []))
    gyro_coef = calcular_coeficiente_relacion(usuario.get("gyro", []))
    for key, value in accel_coef.items():
        user_data[f"accel_{key}"] = value
    for key, value in gyro_coef.items():
        user_data[f"gyro_{key}"] = value
    usuarios_procesados_coef.append(user_data)
df_coef = pd.DataFrame(usuarios_procesados_coef)


"""## Dataframe completo (Dominio de tiempo - frecuencia - otros)"""
def calcular_varianza(sensor_data):
    if not sensor_data:
        return {"variance_x": np.nan, "variance_y": np.nan, "variance_z": np.nan}
    valores_x = [d["x"] for d in sensor_data]
    valores_y = [d["y"] for d in sensor_data]
    valores_z = [d["z"] for d in sensor_data]
    return {
        "variance_x": np.var(valores_x),
        "variance_y": np.var(valores_y),
        "variance_z": np.var(valores_z)
    }
def calcular_iqr(sensor_data):
    if not sensor_data:
        return {"iqr_x": np.nan, "iqr_y": np.nan, "iqr_z": np.nan}
    valores_x = [d["x"] for d in sensor_data]
    valores_y = [d["y"] for d in sensor_data]
    valores_z = [d["z"] for d in sensor_data]
    return {
        "iqr_x": np.percentile(valores_x, 75) - np.percentile(valores_x, 25),
        "iqr_y": np.percentile(valores_y, 75) - np.percentile(valores_y, 25),
        "iqr_z": np.percentile(valores_z, 75) - np.percentile(valores_z, 25)
    }
def calcular_rms(sensor_data):
    if not sensor_data:
        return {"rms_x": np.nan, "rms_y": np.nan, "rms_z": np.nan}
    valores_x = [d["x"] for d in sensor_data]
    valores_y = [d["y"] for d in sensor_data]
    valores_z = [d["z"] for d in sensor_data]
    return {
        "rms_x": np.sqrt(np.mean(np.square(valores_x))),
        "rms_y": np.sqrt(np.mean(np.square(valores_y))),
        "rms_z": np.sqrt(np.mean(np.square(valores_z)))
    }
def calcular_entropia(sensor_data):
    if not sensor_data:
        return {"entropy_x": np.nan, "entropy_y": np.nan, "entropy_z": np.nan}
    valores_x = [d["x"] for d in sensor_data]
    valores_y = [d["y"] for d in sensor_data]
    valores_z = [d["z"] for d in sensor_data]
    def calcular_entropia_vector(valores):
        _, counts = np.unique(valores, return_counts=True)
        probabilidad = counts / len(valores)
        return -np.sum(probabilidad * np.log2(probabilidad))
    return {
        "entropy_x": calcular_entropia_vector(valores_x),
        "entropy_y": calcular_entropia_vector(valores_y),
        "entropy_z": calcular_entropia_vector(valores_z)
    }
def calcular_dfft(sensor_data):
    if not sensor_data:
        return {"dfft_x": np.nan, "dfft_y": np.nan, "dfft_z": np.nan}
    valores_x = [d["x"] for d in sensor_data]
    valores_y = [d["y"] for d in sensor_data]
    valores_z = [d["z"] for d in sensor_data]
    dfft_x = np.abs(np.fft.fft(valores_x))[:len(valores_x)//2]
    dfft_y = np.abs(np.fft.fft(valores_y))[:len(valores_y)//2]
    dfft_z = np.abs(np.fft.fft(valores_z))[:len(valores_z)//2]
    return {
        "dfft_x": np.max(dfft_x),
        "dfft_y": np.max(dfft_y),
        "dfft_z": np.max(dfft_z)
    }
def calcular_dct(sensor_data):
    if not sensor_data:
        return {"dct_x": np.nan, "dct_y": np.nan, "dct_z": np.nan}
    valores_x = [d["x"] for d in sensor_data]
    valores_y = [d["y"] for d in sensor_data]
    valores_z = [d["z"] for d in sensor_data]
    dct_x = np.abs(dct(valores_x))[:len(valores_x)//2]
    dct_y = np.abs(dct(valores_y))[:len(valores_y)//2]
    dct_z = np.abs(dct(valores_z))[:len(valores_z)//2]
    return {
        "dct_x": np.max(dct_x),
        "dct_y": np.max(dct_y),
        "dct_z": np.max(dct_z)
    }
def calcular_energia(sensor_data):
    if not sensor_data:
        return {"energia_x": np.nan, "energia_y": np.nan, "energia_z": np.nan}
    valores_x = [d["x"] for d in sensor_data]
    valores_y = [d["y"] for d in sensor_data]
    valores_z = [d["z"] for d in sensor_data]
    energia_x = np.sum(np.square(valores_x))
    energia_y = np.sum(np.square(valores_y))
    energia_z = np.sum(np.square(valores_z))
    return {
        "energia_x": energia_x,
        "energia_y": energia_y,
        "energia_z": energia_z
    }
def calcular_rango_frecuencia(sensor_data):
    if not sensor_data:
        return {"freq_range_x": np.nan, "freq_range_y": np.nan, "freq_range_z": np.nan}
    valores_x = [d["x"] for d in sensor_data]
    valores_y = [d["y"] for d in sensor_data]
    valores_z = [d["z"] for d in sensor_data]
    dfft_x = np.abs(np.fft.fft(valores_x))[:len(valores_x)//2]
    dfft_y = np.abs(np.fft.fft(valores_y))[:len(valores_y)//2]
    dfft_z = np.abs(np.fft.fft(valores_z))[:len(valores_z)//2]
    return {
        "freq_range_x": np.max(dfft_x) - np.min(dfft_x),
        "freq_range_y": np.max(dfft_y) - np.min(dfft_y),
        "freq_range_z": np.max(dfft_z) - np.min(dfft_z)
    }
def calcular_coef_relacion_ejes(sensor_data):
    if not sensor_data:
        return {"axis_relation_x": np.nan, "axis_relation_y": np.nan, "axis_relation_z": np.nan}
    valores_x = [d["x"] for d in sensor_data]
    valores_y = [d["y"] for d in sensor_data]
    valores_z = [d["z"] for d in sensor_data]
    relation_x = np.corrcoef(valores_x, valores_y)[0, 1] if len(valores_x) > 1 else np.nan
    relation_y = np.corrcoef(valores_y, valores_z)[0, 1] if len(valores_y) > 1 else np.nan
    relation_z = np.corrcoef(valores_x, valores_z)[0, 1] if len(valores_x) > 1 else np.nan
    return {
        "axis_relation_x": relation_x,
        "axis_relation_y": relation_y,
        "axis_relation_z": relation_z
    }
usuarios_procesados = []

for usuario in data_sintetizado:
    user_data = {"userId": usuario["userId"], "name": usuario["name"]}
    accel_variance = calcular_varianza(usuario.get("accel", []))
    gyro_variance = calcular_varianza(usuario.get("gyro", []))
    accel_iqr = calcular_iqr(usuario.get("accel", []))
    gyro_iqr = calcular_iqr(usuario.get("gyro", []))
    accel_rms = calcular_rms(usuario.get("accel", []))
    gyro_rms = calcular_rms(usuario.get("gyro", []))
    accel_entropy = calcular_entropia(usuario.get("accel", []))
    gyro_entropy = calcular_entropia(usuario.get("gyro", []))
    accel_dfft = calcular_dfft(usuario.get("accel", []))
    gyro_dfft = calcular_dfft(usuario.get("gyro", []))
    accel_dct = calcular_dct(usuario.get("accel", []))
    gyro_dct = calcular_dct(usuario.get("gyro", []))
    accel_energia = calcular_energia(usuario.get("accel", []))
    gyro_energia = calcular_energia(usuario.get("gyro", []))
    accel_freq_range = calcular_rango_frecuencia(usuario.get("accel", []))
    gyro_freq_range = calcular_rango_frecuencia(usuario.get("gyro", []))
    accel_coef_rel = calcular_coef_relacion_ejes(usuario.get("accel", []))
    gyro_coef_rel = calcular_coef_relacion_ejes(usuario.get("gyro", []))
    for key, value in accel_variance.items():
        user_data[f"accel_{key}"] = value
    for key, value in gyro_variance.items():
        user_data[f"gyro_{key}"] = value
    for key, value in accel_iqr.items():
        user_data[f"accel_{key}"] = value
    for key, value in gyro_iqr.items():
        user_data[f"gyro_{key}"] = value
    for key, value in accel_rms.items():
        user_data[f"accel_{key}"] = value
    for key, value in gyro_rms.items():
        user_data[f"gyro_{key}"] = value
    for key, value in accel_entropy.items():
        user_data[f"accel_{key}"] = value
    for key, value in gyro_entropy.items():
        user_data[f"gyro_{key}"] = value
    for key, value in accel_dfft.items():
        user_data[f"accel_{key}"] = value
    for key, value in gyro_dfft.items():
        user_data[f"gyro_{key}"] = value
    for key, value in accel_dct.items():
        user_data[f"accel_{key}"] = value
    for key, value in gyro_dct.items():
        user_data[f"gyro_{key}"] = value
    for key, value in accel_energia.items():
        user_data[f"accel_{key}"] = value
    for key, value in gyro_energia.items():
        user_data[f"gyro_{key}"] = value
    for key, value in accel_freq_range.items():
        user_data[f"accel_{key}"] = value
    for key, value in gyro_freq_range.items():
        user_data[f"gyro_{key}"] = value
    for key, value in accel_coef_rel.items():
        user_data[f"accel_{key}"] = value
    for key, value in gyro_coef_rel.items():
        user_data[f"gyro_{key}"] = value
    usuarios_procesados.append(user_data)
df_freq_time_ot = pd.DataFrame(usuarios_procesados)
df_completo = pd.merge(df_sintetizado, df_freq_time_ot, on=["userId", "name"], how="left")