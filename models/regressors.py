from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor


def get_model(model_type):
    if model_type == "Regressão Linear":
        return LinearRegression()
    elif model_type == "Árvore de Decisão":
        return DecisionTreeRegressor()
    else:
        raise ValueError("Tipo de modelo não suportado")
