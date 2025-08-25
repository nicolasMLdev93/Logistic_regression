import matplotlib.pyplot as plt
import numpy as np
from src.evaluate import result

probabilities, predictions = result
n_samples = len(predictions)
indices = np.arange(n_samples)

# Probabilidades de no fraude y fraude
no_fraud = [probabilities[i][0] for i in range(n_samples)]
fraud = [probabilities[i][1] for i in range(n_samples)]

plt.figure(figsize=(12,6))
plt.bar(indices - 0.2, no_fraud, width=0.4, label='No fraude')
plt.bar(indices + 0.2, fraud, width=0.4, label='Fraude')
plt.xlabel('Registro')
plt.ylabel('Probabilidad (%)')
plt.title('Probabilidad de fraude vs no fraude por registro')
plt.legend()
plt.show()
