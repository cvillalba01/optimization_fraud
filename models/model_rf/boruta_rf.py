from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Cargar el conjunto de datos y las etiquetas
X = ...  # Características
y = ...  # Variable objetivo

# Crear un modelo Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Crear el selector Boruta
boruta_selector = BorutaPy(rf, n_estimators='auto', random_state=42)

# Ajustar el selector Boruta al conjunto de datos
boruta_selector.fit(X, y)

# Seleccionar las características importantes
important_features = X.columns[boruta_selector.support_]

print("Características seleccionadas por Boruta:")
print(important_features)
