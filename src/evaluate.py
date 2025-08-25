from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import pandas as pd
from model import trained_model, scaler


def predict_score():
    df = pd.read_csv('../data/x_test.csv') 
    x_test = df
    
    # Escalar los datos de test
    x_test_scaled = scaler.transform(x_test)

    # Predecir con el modelo entrenado
    probability = trained_model.predict_proba(x_test_scaled) * 100
    predic = trained_model.predict(x_test_scaled)
    return probability,predic

result = predict_score()

def show_evaluation():
    probabilities, predictions = result
    for i in range(len(predictions)):
        print(f"El input n° {i} {'es fraudulento' if predictions[i] == 1 else 'no es fraudulento'}")
        print(f"Probabilidad NO FRAUDE: {probabilities[i][0].round(2)} %, FRAUDE: {probabilities[i][1].round(2)} %\n")

        
show_evaluation()


