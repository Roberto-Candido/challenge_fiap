import pandas as pd
from sklearn.preprocessing import LabelEncoder


def load_data(file):
    return pd.read_csv(file)


def preprocess_data(df):
    df = df.copy()
    label_cols = ['gênero', 'fumante', 'região']
    for col in label_cols:
        df[col] = LabelEncoder().fit_transform(df[col])
    return df
