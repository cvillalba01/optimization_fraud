import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, classification_report, roc_curve, auc, RocCurveDisplay
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE,BorderlineSMOTE  # Sobremuestreo con SMOTE
from imblearn.over_sampling import ADASYN,RandomOverSampler 
from imblearn.under_sampling import RandomUnderSampler
import joblib

# Cargar los datos
#data = pd.read_csv('../../data/PreprocessData/prediction_results_202409081602_copy.csv')
#data = pd.read_csv('../../data/PreprocessData/copy_training_data_latest.csv')
#data = pd.read_csv('../../data/PreprocessData/copy_training_data.csv')
#data = pd.read_csv('../../data/PreprocessData/prediction_results_total.csv')
data = pd.read_csv('../../models/model_rf/final_anomaly_scores_with_data.csv')
# Preprocesamiento 
# Eliminar columnas innecesarias
X = data.drop(columns=[#'rrn', 
                       'fraud_label', 'created', #'wallet_number',
                       'fraud_probability', 'ip_clean', 'created',
                       #'service_id',
                       'is_atypical_amount',#-->al quitar esto sube la precisión pero baja el recall
                       #'application_id',
                       #'transaction_frequency_last_30_days',
                       #'jcard_type',
                       'transactions_to_same_wallet_last_30_days',
                       'get_z_score',
                        #"get_transaction_type_probability",
                        #"get_time_diff",
                        #"is_active_client",
                        #"is_distribution_wallet_customer",
                        "ip_clean",
                        "has_ip_changed",
                        #"is_ip_associated_with_other_wallet",
                        #"service_id",
                        #"is_service_habitual_at_transaction",
                        "total_amount_last_7_days",
                        #"avg_amount_by_transaction_type",
                        #"get_std_dev_by_transaction_type",
                        "get_z_score_by_jcard_type",
                        #"anomaly_score"
                        ])

# Llenar valores nulos si es necesario
X.fillna(0, inplace=True)

# Convertir las columnas categóricas a numéricas
le = LabelEncoder()
X['jcard_type'] = le.fit_transform(X['jcard_type'])
X['rrn'] = le.fit_transform(X['rrn'])
X['wallet_number'] = le.fit_transform(X['wallet_number'])

# Variable objetivo
y = data['fraud_label'].astype(int)

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Aplicar SMOTE solo al conjunto de entrenamiento
ada = SMOTE(random_state=42) 
X_train_resampled, y_train_resampled = ada.fit_resample(X_train, y_train)

print("Tamaño original de X_train:", X_train.shape)
print("Tamaño después de aplicar SMOTE en X_train:", X_train_resampled.shape)
print("Tamaño original de y_train:", y_train.shape)
print("Tamaño después de aplicar SMOTE en y_train:", y_train_resampled.shape)

# Crear el modelo Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_resampled, y_train_resampled)

# Hacer predicciones
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probabilidad de fraude

# Guardar las predicciones en un DataFrame
predictions_df = pd.DataFrame({
    'predicted_label': y_pred,         # Predicción binaria (0/1)
    'fraud_probability': y_pred_proba, # Probabilidad de fraude
    'actual_label': y_test             # Etiqueta real
})

# Matriz de Confusión
cm = confusion_matrix(y_test, y_pred)
cmd = ConfusionMatrixDisplay(cm, display_labels=[0, 1])
cmd.plot(cmap=plt.cm.Blues)
plt.title("Matriz de Confusión")
plt.savefig('matriz_confusion2.png')
plt.close()

# Mostrar las primeras filas de las predicciones con probabilidad
print(predictions_df.head())

# Guardar las predicciones en un archivo CSV
predictions_df.to_csv('/home/avelazquez/projects/fraud-detection-pry/models/model_rf/predicciones_con_probabilidad.csv', index=False)

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred)
print("Matriz de confusión:")
print(cm)

# Guardar matriz de confusión como imagen
plt.figure(figsize=(6, 6))
plt.matshow(cm, fignum=1)
plt.title('Matriz de Confusión')
plt.colorbar()
plt.ylabel('Etiqueta real')
plt.xlabel('Etiqueta predicha')
plt.savefig('/home/avelazquez/projects/fraud-detection-pry/models/model_rf/confusion_matrix.png')

# Reporte de clasificación
print("Reporte de clasificación:")
print(classification_report(y_test, y_pred))

# Curva ROC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 6))
RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='Random Forest').plot()
plt.title('Curva ROC')
plt.savefig('/home/avelazquez/projects/fraud-detection-pry/models/model_rf/roc_curve.png')

# Gráficos de Recall, Precisión, F1-Score
report = classification_report(y_test, y_pred, output_dict=True)
metrics_df = pd.DataFrame(report).transpose()

plt.figure(figsize=(10, 6))
metrics_df[['precision', 'recall', 'f1-score']].iloc[:-3].plot(kind='bar')
plt.title('Precision, Recall, F1-Score por clase')
plt.xticks(rotation=45)
plt.savefig('/home/avelazquez/projects/fraud-detection-pry/models/model_rf/metrics_bar_chart.png')
