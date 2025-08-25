from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Dataset limpio
df = pd.read_csv('../data/clean_data.csv')

def get_variables(df):
    # Variables independientes (x1,x2 ... xn)
    x = df.iloc[:, 0:19]
    # Variable dependiente (y)
    y = df.iloc[:, -1]
    return x, y

X, Y = get_variables(df)

# Escalar los datos
# El escalado de datos es un proceso que transforma las variables numéricas para que todas tengan la misma escala. 
# Esto evita que columnas con valores muy grandes dominen a otras con valores más pequeños y ayuda a que los modelos
# aprendan correctamente.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

def train_model(x, y):  
    # clf -> clasificador (0 o 1 en regresión logística)
    # Se entrena el modelo con nuestro dataset
    clf = LogisticRegression(class_weight='balanced')
    clf.fit(x, y)
    return clf

trained_model = train_model(X_scaled, Y)