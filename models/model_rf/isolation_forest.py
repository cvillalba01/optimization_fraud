import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import IsolationForest

# Cargar los datos
#data = pd.read_csv('../../data/PreprocessData/prediction_results_202409081602_copy.csv')
#data = pd.read_csv('../../data/PreprocessData/copy_training_data_latest.csv')
data = pd.read_csv('../../data/PreprocessData/copy_training_data.csv')
#data = pd.read_csv('../../data/PreprocessData/prediction_results_total.csv')

# Preprocesamiento (asegúrate de eliminar o transformar columnas irrelevantes)
X = data.drop(columns=['rrn', 'fraud_label', 'created', 'wallet_number', 'fraud_probability', 'ip_clean', 'created',
                       #'service_id',
                       #'is_ip_associated_with_other_wallet',
                       #'is_atypical_amount',
                       #'application_id',
                       #'transaction_frequency_last_30_days',
                       #'total_amount_last_7_days',
                       #'is_atypical_amount',
                       #'transactions_to_same_wallet_last_30_days',
                       'get_z_score'
                    ])  # Quitar columnas no útiles
X.fillna(0, inplace=True)  # Llenar valores nulos

le = LabelEncoder()
X['jcard_type'] = le.fit_transform(X['jcard_type'])
# Aplicar Isolation Forest
model = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
model.fit(X)

# Predecir si una transacción es normal (-1 indica anomalía, 1 indica normalidad)
data['anomaly'] = model.predict(X)

# Mostrar las transacciones que son anomalías
anomalies = data[data['anomaly'] == -1]
print(anomalies)

# Guardar las anomalías detectadas
anomalies.to_csv('anomalous_transactions.csv', index=False)
