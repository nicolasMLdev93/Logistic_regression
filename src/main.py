import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.preprocessing import StandardScaler
import pandas as pd
from classes.fraud_class import Fraud_detect

# Dataset limpio
df_train = pd.read_csv('../data/clean_data.csv')
# Dataset para realizar las predicciones
df_test = pd.read_csv('../data/x_test.csv')

def get_variables(df_train, df_test):
    # Entrenamiento:
    # Variables independientes (x1,x2 ... xn)
    x_train = df_train.iloc[:, 0:19]
    # Variable dependiente (y)
    y_train = df_train.iloc[:, -1]

    # Predicciones:
    # Variables independientes (x1,x2 ... xn)
    x_test = df_test.iloc[:, 0:19]
    # Variable dependiente (y)
    y_test = df_test.iloc[:, -1]

    return x_train, y_train, x_test, y_test

# Obtener variables
x_train, y_train, x_test, y_test = get_variables(df_train, df_test)

# Instancio una nueva detección de fraudes
detect_instance: Fraud_detect = Fraud_detect()

# Escalar los datos
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(x_train)  
X_test_scaled  = scaler.transform(x_test)        

# Entrenar modelo
detect_instance.train_model(X_train_scaled, y_train)

# Predicciones y Probabilidades
probabilities = detect_instance.probability_detector(X_test_scaled) * 100 
predictions   = detect_instance.predict_data(X_test_scaled)

# Mostrar resultados individuales
def show_evaluation(probabilities, predictions):
    for i, pred in enumerate(predictions):
        status = "FRAUDULENTO" if pred == 1 else "NO FRAUDULENTO"
        print(f"--- Input {i + 1} ---")
        print(f"Clasificación: {status}")
        print(f"Probabilidad NO FRAUDE: {probabilities[i][0].round(2)}%")
        print(f"Probabilidad FRAUDE:    {probabilities[i][1].round(2)}%\n")

show_evaluation(probabilities, predictions)
