import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, RocCurveDisplay
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

from imblearn.over_sampling import SMOTE  # Sobremuestreo con SMOTE
from imblearn.over_sampling import ADASYN
from imblearn.under_sampling import RandomUnderSampler

# Cargar los datos
data = pd.read_csv('../../data/PreprocessData/prediction_results_202409081602_copy.csv')

# Preprocesamiento
# Eliminar columnas innecesarias
X = data.drop(columns=['rrn', 'fraud_label', 'created', 'wallet_number',
                        'fraud_probability', 'ip_clean', 'created'#,
                       #'service_id', 'is_ip_associated_with_other_wallet', 'is_atypical_amount', 'application_id',
                       #'total_amount_last_7_days', 'transaction_frequency_last_30_days', 'total_amount_last_7_days',
                       #'is_atypical_amount', 'transactions_to_same_wallet_last_30_days'
                       ])

# Llenar valores nulos si es necesario
X.fillna(0, inplace=True)

# Convertir las columnas categóricas a numéricas
le = LabelEncoder()
X['jcard_type'] = le.fit_transform(X['jcard_type'])
#X['wallet_number'] = le.fit_transform(X['wallet_number'].astype(str))

# Variable objetivo
y = data['fraud_label'].astype(int)

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Aplicar SMOTE solo al conjunto de entrenamiento
ada = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = ada.fit_resample(X_train, y_train)

# Crear el modelo XGBoost
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
model.fit(X_train_resampled, y_train_resampled)

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
plt.savefig('/home/avelazquez/projects/fraud-detection-pry/models/model_xgb/confusion_matrix.png')

# Reporte de clasificación
print("Reporte de clasificación:")
print(classification_report(y_test, y_pred))

# Curva ROC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 6))
RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='XGBoost').plot()
plt.title('Curva ROC')
plt.savefig('/home/avelazquez/projects/fraud-detection-pry/models/model_xgb/roc_curve.png')

# Gráficos de Recall, Precisión, F1-Score
report = classification_report(y_test, y_pred, output_dict=True)
metrics_df = pd.DataFrame(report).transpose()

plt.figure(figsize=(10, 6))
metrics_df[['precision', 'recall', 'f1-score']].iloc[:-3].plot(kind='bar')
plt.title('Precision, Recall, F1-Score por clase')
plt.xticks(rotation=45)
plt.savefig('/home/avelazquez/projects/fraud-detection-pry/models/model_xgb/metrics_bar_chart.png')
