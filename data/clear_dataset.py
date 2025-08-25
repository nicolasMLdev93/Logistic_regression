import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_csv('../data/data_set.csv') 

def clear_dataset(df):
    # Convertir a float
    cols_to_float = ['LIMIT_BAL', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
    for col in cols_to_float:
        df[col] = df[col].astype(float).round(2)

    # Eliminar duplicados y valores nulos
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)

    # Eliminar outliers de columnas numéricas
    numeric_cols = cols_to_float
    df = df[(np.abs(stats.zscore(df[numeric_cols])) < 3).all(axis=1)]

    # Eliminar columnas que no aportan al modelo como sexo,educacion y si es casada o no la perspona
    df.drop(columns=['EDUCATION','SEX','MARRIAGE','AGE'],inplace=True)

    return df

result = clear_dataset(df)
result.to_csv('clean_data.csv', index=False)
