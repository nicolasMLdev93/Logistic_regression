import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import matplotlib.pyplot as plt
from classes.fraud_class import Fraud_detect

# Dataset limpio
df = pd.read_csv('../data/clean_data.csv')

def get_variables(df):
    x = df.iloc[:, 0:19]  # Variables independientes
    y = df.iloc[:, -1]    # Variable dependiente
    return x, y

X_train, Y_train = get_variables(df)

# Instancio una nueva detección de fraudes
detect_instance:Fraud_detect = Fraud_detect()

# Entrenar modelo
detect_instance.train_model(X_train,Y_train)

# Calcular AUC : valor numérico que mide el área bajo la ROC (entre 0 y 1)
auc:float = detect_instance.get_auc_score(X_train,Y_train)
print(f"AUC: {auc:.2f}")

# Calcular curva ROC
fpr, tpr, thresholds = detect_instance.get_roc_curve(X_train,Y_train)

# Graficar curva ROC
# ROC = Receiver Operating Characteristic: es una curva que muestra el desempeño de un modelo de
# clasificación según cómo cambia el umbral de decisión.
plt.figure(figsize=(8,6))
# Línea azul: cada punto de la curva azul corresponde a un resultado de la evaluación de los datos entre 0 y 1.
plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {auc:.2f}%)')
# Línea roja: representa un modelo que no distingue entre positivos y negativos.
plt.plot([0, 1], [0, 1], color='red', linestyle='--')  # línea diagonal de referencia
plt.xlabel('Falsos positivos')
plt.ylabel('Verdaderos positivos')
plt.title('Curva ROC - Regresión Logística')
plt.grid()
plt.savefig('../reports/roc_auc.png') 
plt.show()
