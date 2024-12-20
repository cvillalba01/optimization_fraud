import pandas as pd

# Ruta del archivo CSV original
input_file = '../data/totalDataFrame.csv'

# Ruta del archivo CSV reducido para session_data
output_file_session = '/tmp/sessionData_reduced.csv'

# Leer el archivo CSV
df = pd.read_csv(input_file, dtype={'wallet_number': str, 'service_id': 'float'})

# Convertir valores no nulos a enteros, manteniendo NaN en service_id
df['service_id'] = df['service_id'].apply(lambda x: int(x) if pd.notna(x) else pd.NA)

# Seleccionar las columnas necesarias para la tabla session_data
columns_needed_session = [
    'rrn', 
    'login_date', 
    'application_id', 
    'remote_ip_clean'
]

# Verificar si todas las columnas necesarias están en el archivo CSV
missing_cols_session = [col for col in columns_needed_session if col not in df.columns]
if missing_cols_session:
    raise ValueError(f"Las siguientes columnas están faltando en el archivo CSV: {', '.join(missing_cols_session)}")

# Filtrar el DataFrame para mantener solo las columnas necesarias para session_data
df_session_reduced = df[columns_needed_session]

# Guardar el DataFrame reducido en un nuevo archivo CSV
df_session_reduced.to_csv(output_file_session, index=False)

print(f"Archivo CSV reducido para session_data guardado en: {output_file_session}")
