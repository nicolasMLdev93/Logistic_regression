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

def get_variables(df_train,df_test):
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

    return x_test,x_train,y_test,y_train

x_test,x_train,y_test,y_train = get_variables(df_train,df_test)

# Instancio una nueva detección de fraudes
detect_instance:Fraud_detect = Fraud_detect()

# Escalar los datos
# El escalado de datos es un proceso que transforma las variables numéricas para que todas tengan la misma escala. 
# Esto evita que columnas con valores muy grandes dominen a otras con valores más pequeños y ayuda a que los modelos
# aprendan correctamente.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(x_train)

# Entrenar modelo
detect_instance.train_model(X_scaled,y_train)

X_scaled_train = scaler.fit_transform(x_test)

# Predicciones y Probabilidades
probability = detect_instance.probability_detector(X_scaled_train) * 100
predic = detect_instance.predict_data(X_scaled_train)

def show_evaluation(probabilities, predictions):
    for i in range(len(predictions)):
        print(f"El input n° {i} {'es fraudulento' if predictions[i] == 1 else 'no es fraudulento'}")
        print(f"Probabilidad NO FRAUDE: {probabilities[i][0].round(2)} %, FRAUDE: {probabilities[i][1].round(2)} %\n")
        
show_evaluation(probability,predic)