import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Base de dados de exemplo (apenas para ilustração)
sintomas = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0]]) # Febre, tosse, dor abdominal
doencas = np.array(["Gripe", "Apendicite", "Enxaqueca"])

# Dividindo os dados
X_train, X_test, y_train, y_test = train_test_split(sintomas, doencas, test_size=0.2)

# Treinando o modelo
modelo = RandomForestClassifier()
modelo.fit(X_train, y_train)

# Entrada de exemplo do usuário
entrada_usuario = np.array([[0, 1, 0]]) # Usuário reporta febre e dor abdominal

# Prevendo a doença
doenca_predita = modelo.predict(entrada_usuario)
print("Doença mais provável:", doenca_predita)

