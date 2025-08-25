import pandas as pd

# El dataset origianl que es .xls se convierte en un archivo .csv para manipulación más eficiente de los datos
df = pd.read_excel('data.xls', index_col=0)  
df.to_csv('data_set.csv', index=False) 