import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, RocCurveDisplay
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Cargar los datos
data = pd.read_csv('../../data/PreprocessData/prediction_results_202409081602_copy.csv')

# Preprocesamiento
# Eliminar columnas innecesarias
X = data.drop(columns=['rrn', 'fraud_label', 'created', 'wallet_number', 'fraud_probability', 'ip_clean', 'created'
                       #,'service_id',
                       #'is_ip_associated_with_other_wallet','is_atypical_amount','application_id','total_amount_last_7_days',
                        #'transaction_frequency_last_30_days','total_amount_last_7_days','is_atypical_amount','transactions_to_same_wallet_last_30_days'
                        ])

#"rrn","wallet_number","jcard_type","amount","created","promedio","desviacion_estandar","get_z_score",
# "get_transaction_type_probability","get_time_diff","is_active_client","is_distribution_wallet_customer",
# "ip_clean","application_id","has_ip_changed","is_ip_associated_with_other_wallet","service_id","is_service_habitual_at_transaction",
# "transaction_frequency_last_30_days","total_amount_last_7_days","is_atypical_amount","transactions_to_same_wallet_last_30_days",
# "avg_amount_by_transaction_type",# "get_std_dev_by_transaction_type","get_z_score_by_jcard_type","fraud_probability","fraud_label"


# Llenar valores nulos si es necesario
X.fillna(0, inplace=True)

# Convertir las columnas categóricas a numéricas
le = LabelEncoder()
X['jcard_type'] = le.fit_transform(X['jcard_type'])


# Variable objetivo
y = data['fraud_label'].astype(int)

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Crear el modelo Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Hacer predicciones
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

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