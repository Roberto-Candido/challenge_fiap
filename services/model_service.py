import os
import pandas as pd
import joblib
import statsmodels.api as sm
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from models.regressors import get_model
from utils.logger import setup_logger
from utils.config_loader import carregar_config_yaml
from pymongo import MongoClient

logger = setup_logger()


def train_and_evaluate_model(df, model_type):
    X = df.drop("encargos", axis=1)
    y = df["encargos"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = get_model(model_type)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Validação estatística com statsmodels
    X_train_sm = sm.add_constant(X_train)
    sm_model = sm.OLS(y_train, X_train_sm).fit()

    metrics = {
        "r2": r2_score(y_test, y_pred),
        "mse": mean_squared_error(y_test, y_pred),
        "mae": mean_absolute_error(y_test, y_pred),
        "pvalues": sm_model.pvalues.to_dict(),
        "conf_intervals": sm_model.conf_int().to_dict()
    }

    logger.info(f"Modelo treinado com {model_type}")

    # Salvar modelo e resultados
    model_path = save_model(model, model_type.lower().replace(" ", "_"))
    csv_path = save_predictions(y_test, y_pred)

    logger.info(f"Modelo salvo em {model_path}")
    logger.info(f"Resultados salvos em {csv_path}")

    # Salvar relatório no MongoDB
    salvar_relatorio_mongo(model_type, metrics, model_path, csv_path)

    return {
        "model": model,
        "y_test": y_test,
        "y_pred": y_pred,
        "metrics": metrics
    }


def save_model(model, name):
    os.makedirs("medical_costs_app/saved_models", exist_ok=True)
    path = f"medical_costs_app/saved_models/{name}.pkl"
    joblib.dump(model, path)
    return path


def save_predictions(y_test, y_pred):
    os.makedirs("medical_costs_app/results", exist_ok=True)
    df_result = pd.DataFrame({"Real": y_test, "Previsto": y_pred})
    path = "medical_costs_app/results/predictions.csv"
    df_result.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def salvar_relatorio_mongo(model_name, metrics, model_path, csv_path):
    config = carregar_config_yaml()
    mongo_conf = config["mongodb"]

    try:
        client = MongoClient(mongo_conf["host"], mongo_conf["port"])
        db = client[mongo_conf["database"]]
        collection = db[mongo_conf["collection"]]

        # Convertendo chaves numéricas para string (necessário para MongoDB)
        intervalos_convertido = {str(k): v for k, v in metrics["conf_intervals"].items()}

        documento = {
            "modelo": model_name,
            "metrica_r2": metrics["r2"],
            "metrica_mae": metrics["mae"],
            "metrica_mse": metrics["mse"],
            "pvalues": metrics["pvalues"],
            "intervalos_confianca": intervalos_convertido,
            "arquivo_modelo": model_path,
            "arquivo_resultado": csv_path,
            "data_execucao": datetime.now().isoformat()
        }

        collection.insert_one(documento)
        logger.info("Relatório salvo no MongoDB.")
    except Exception as e:
        logger.error(f"Erro ao salvar no MongoDB: {e}")
