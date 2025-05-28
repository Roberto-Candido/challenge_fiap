import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
import joblib  # Para salvar o modelo
from sklearn.datasets import fetch_california_housing


# Carregar o dataset
data = fetch_california_housing(as_frame=True)
df = data.frame

print(df.head(10))

# Separar em X (variáveis independentes) e y (variável alvo)
X = df.drop(columns=["MedHouseVal"])  # Todas as variáveis exceto a target
y = df["MedHouseVal"]

# Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinar modelo
model = LinearRegression()
model.fit(X_train, y_train)

# Previsões
y_pred = model.predict(X_test)

# Métricas
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"MAE : {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²  : {r2:.4f}")


# Dados de entrada (1 casa)
nova_casa = pd.DataFrame([{
    'MedInc': 5.0,
    'HouseAge': 20,
    'AveRooms': 5.0,
    'AveBedrms': 1.0,
    'Population': 1000,
    'AveOccup': 3.0,
    'Latitude': 34.0,
    'Longitude': -118.0
}])

# Previsão
previsao = model.predict(nova_casa)

print(f"Valor estimado da casa: {previsao[0]:.2f} (em centenas de milhares de dólares)")


plt.figure(figsize=(10, 6))

# Pontos reais (azul)
plt.scatter(range(len(y_test)), y_test, color='blue', label='Valor Real', alpha=0.6)

# Pontos previstos (laranja)
plt.scatter(range(len(y_pred)), y_pred, color='orange', label='Valor Previsto', alpha=0.6)

plt.title('Comparação entre valores reais e previstos')
plt.xlabel('Índice da Amostra')
plt.ylabel('Valor médio da casa')
plt.legend()
plt.grid(True)
plt.show()




