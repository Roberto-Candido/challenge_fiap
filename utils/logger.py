import logging
import os


def setup_logger():
    os.makedirs('medical_costs_app/logs', exist_ok=True)
    logger = logging.getLogger('medical_costs_logger')
    logger.setLevel(logging.INFO)

    # Evita múltiplos handlers duplicados ao recarregar em Streamlit
    if not logger.handlers:
        file_handler = logging.FileHandler('medical_costs_app/logs/app.log', encoding='utf-8')
        formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
