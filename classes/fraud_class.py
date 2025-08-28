from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score

class Fraud_detect:
    def __init__(self):
        self.model = LogisticRegression(solver="newton-cholesky", random_state=0)
    def train_model(self,x_train,y_train):
        # Método para entrenar el modelo de Regresión Logística
        self.model.fit(x_train,y_train)
    def predict_data(self,x_test):
        # Predicción de datos (0,1)
        return self.model.predict(x_test)
    def probability_detector(self,x_test):
        # Predicción de probabilidad de c/u de las clases
        return self.model.predict_proba(x_test)
    def get_score(self,x_test,y_test):
        # Score nos dice qué tan bien el modelo predijo las clases correctas en un conjunto de datos.
        return self.model.score(x_test,y_test)
    def get_auc_score(self,x_test,y_test):
        # Calcular AUC : valor numérico que mide el área bajo la ROC (entre 0 y 1)
        y_prob = self.model.predict_proba(x_test)[:, 1] # obtenemos la columna de la clase positiva
        return roc_auc_score(y_test,y_prob)
    def get_roc_curve(self,x_test,y_test):
        # Calcular curva ROC
        y_prob = self.model.predict_proba(x_test)[:, 1] # obtenemos la columna de la clase positiva
        return roc_curve(y_test, y_prob)