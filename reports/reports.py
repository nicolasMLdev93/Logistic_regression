import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score

# Dataset limpio
df = pd.read_csv('../data/clean_data.csv')

def get_variables(df):
    x = df.iloc[:, 0:19]  # Variables independientes
    y = df.iloc[:, -1]    # Variable dependiente
    return x, y

X, Y = get_variables(df)

# Entrenar modelo
clf = LogisticRegression(solver="newton-cholesky", random_state=0)
clf.fit(X, Y)

# Probabilidades clase positiva
y_prob = clf.predict_proba(X)[:, 1]

# Calcular AUC : valor numérico que mide el área bajo la ROC (entre 0 y 1)
auc = roc_auc_score(Y, y_prob) * 100  # convertir a porcentaje
print(f"AUC: {auc:.2f} %")

# Calcular curva ROC
fpr, tpr, thresholds = roc_curve(Y, y_prob)

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
plt.savefig('../reports/roc_auc.png')  # guardar imagen en reports
plt.show()
