import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

# Función principal
def main():
    # Cargar los datos originales
    #data = pd.read_csv('../../data/PreprocessData/prediction_results_202409081602_copy.csv')
    #data = pd.read_csv('../../data/PreprocessData/copy_training_data_latest.csv')
    data = pd.read_csv('../../data/PreprocessData/copy_training_data.csv')
    #data = pd.read_csv('../../data/PreprocessData/prediction_results_total.csv')

    # Preprocesamiento: Seleccionar columnas relevantes
    X = data[['amount', 'promedio', 'desviacion_estandar', 'get_z_score', 
               'transaction_frequency_last_30_days', 'wallet_number', 
               'application_id', 'is_active_client', 'is_distribution_wallet_customer', 
               'has_ip_changed', 'is_ip_associated_with_other_wallet', 
               'service_id', 'is_service_habitual_at_transaction']]  # Incluir las columnas deseadas

    # Comprobar valores nulos antes de cualquier operación
    print("Valores nulos en el conjunto de datos original:")
    print(X.isnull().sum())

    # Convertir todas las columnas a numéricas, ignorando errores
    X = X.apply(pd.to_numeric, errors='coerce')

    # Verificar si hay valores NaN después de la conversión
    print("Valores nulos después de convertir a numérico:")
    print(X.isnull().sum())

    # Comprobar si hay valores NaN y actuar en consecuencia
    if X.isnull().values.any():
        print("Se encontraron valores NaN en el conjunto de datos. Rellenando con 0...")

        # Rellenar valores nulos con 0
        X.fillna(0, inplace=True)

        # Verificar nuevamente si hay NaN
        if X.isnull().values.any():
            print("Todavía hay valores NaN después de la imputación.")
            return
        else:
            print("Todos los valores NaN han sido manejados.")

    # Entrenar el modelo Isolation Forest
    isolation_forest = IsolationForest(contamination=0.1, random_state=42)  # Ajusta el contamination si es necesario
    anomaly_scores_proba = isolation_forest.fit(X).decision_function(X)  # Puntajes de anomalía

    # Agregar los puntajes de anomalía al DataFrame original
    data['anomaly_score'] = anomaly_scores_proba

    # Guardar los resultados en un archivo CSV temporal con 'rrn' y 'anomaly_score'
    temp_output_file = '/home/avelazquez/projects/fraud-detection-pry/models/model_rf/temp_anomaly_scores.csv'
    data[['rrn', 'anomaly_score']].to_csv(temp_output_file, index=False)

    print(f"Puntajes de anomalía guardados temporalmente en: {temp_output_file}")

    # Unir con el DataFrame original para tener la planilla completa
    original_data = pd.read_csv('../../data/PreprocessData/copy_training_data.csv')  # Cargar el original nuevamente
    merged_data = pd.merge(original_data, data[['rrn', 'anomaly_score']], on='rrn', how='left')

    # Guardar el DataFrame unido en una nueva planilla
    final_output_file = '/home/avelazquez/projects/fraud-detection-pry/models/model_rf/final_anomaly_scores_with_data.csv'
    merged_data.to_csv(final_output_file, index=False)

    print(f"Archivo final con puntajes de anomalía guardado en: {final_output_file}")

if __name__ == "__main__":
    main()
