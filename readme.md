📝 Proyecto: Regresión Logística para Detección de Fraude:

Para este proyecto se descargó el archivo .xls con el dataset completo desde:

Default of Credit Card Clients - UCI Machine Learning Repository
URL de descarga => https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients

En esa página se explica detalladamente cada uno de los features y su funcionalidad dentro del dataset.

El proyecto implementa un modelo de regresión logística utilizando Scikit-learn para detectar registros fraudulentos en datos financieros. Además, se calcula la curva ROC y el AUC, métricas que permiten evaluar el desempeño y la capacidad del modelo para distinguir correctamente entre casos fraudulentos y no fraudulentos.

⚡ Instalación y uso: 

1 - Clonar el repo:

git clone https://github.com/nicolasMLdev93/Logistic_regression_Python.git

2 - Crear y activar entorno virtual:

python -m venv .venv

.venv\Scripts\activate

3 - Instalar dependencias:

pip install -r requirements.txt

4 - Entrenar el modelo:

python src/model.py

5 - Predicción de resultados:

python src/evaluate.py

6 - Gráficos:

python reports/reports.py

📈 Curva ROC y AUC:

ROC (Receiver Operating Characteristic):
Curva que muestra cómo cambia la tasa de verdaderos positivos (TPR) frente a la tasa de falsos positivos (FPR) según el umbral de decisión del modelo.

AUC (Area Under the Curve):
Mide la capacidad del modelo de distinguir entre registros positivos y negativos.

AUC = 1 → modelo perfecto

AUC = 0.5 → modelo al azar

AUC = 0.72 → modelo distingue bien, pero puede mejorar

Interpretación de los puntos de la curva:
Cada punto azul corresponde a un umbral específico para clasificar un registro como positivo o negativo.
La curva completa muestra el rendimiento del modelo para todos los posibles umbrales.

Línea diagonal punteada roja:
Representa un modelo que no distingue mejor que azar.

👀 Referencias:

Scikit-learn Logistic Regression:
https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression

Scikit-learn roc_auc_score:
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html


🧑 Mi perfil:

https://www.linkedin.com/in/nicol%C3%A1s-bauz%C3%A1-48a8a0244/ 👈 – Sígueme para ver mis proyectos de desarrollo y ML. 🚀
