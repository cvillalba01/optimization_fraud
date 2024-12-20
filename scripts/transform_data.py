import pandas as pd

# Leer los datos especificando el tipo de datos para evitar advertencias
data = pd.read_csv('../data/Total_2022.csv', dtype={'wallet_number': str})
additional_data = pd.read_csv('../data/Transactions_amount_2022.csv')
bill_payment_data = pd.read_csv('../data/Transaction_amount_bill_payments_2022.csv')
bill_payment_service_data = pd.read_csv('../data/Transaction_amount_bill_payments_2022.csv')

# Realizar el merge con ambos DataFrames de amounts
merged_data = pd.merge(data, additional_data[['rrn', 'amount']], on='rrn', how='left')
merged_data = pd.merge(merged_data, bill_payment_data[['rrn', 'amount']], on='rrn', how='left', suffixes=('', '_bill_payment'))

# Priorizar el 'amount' del archivo 'bill_payment_data' si está presente
merged_data['amount'] = merged_data['amount_bill_payment'].combine_first(merged_data['amount'])

merged_data['remote_ip_clean'] = merged_data['remote_ip'].str.split(',').str[0]
merged_data['remote_ip_clean'] = merged_data['remote_ip_clean'].fillna("0.0.0.0")

# Convertir 'wallet_number' y 'application_id' a cadena de texto
merged_data['wallet_number'] = merged_data['wallet_number'].astype(str)
merged_data['application_id'] = merged_data['application_id'].astype(str)  # Asegurarse de que sea de tipo str

# Reemplazar valores 'nan' como string por '0' y manejar valores NaN reales
merged_data['application_id'] = merged_data['application_id'].replace('nan', '0').fillna('0')
merged_data['application_id'] = merged_data['application_id'].replace('0.0', '0').fillna('0')
merged_data['application_id'] = merged_data['application_id'].replace('10.0', '10').fillna('10')
merged_data['application_id'] = merged_data['application_id'].replace('9.0', '9').fillna('9')
merged_data['application_id'] = merged_data['application_id'].replace('15.0', '15').fillna('15')

# Imprimir valores únicos para verificar el cambio
print("Valores únicos en 'application_id' después de reemplazar:", merged_data['application_id'].unique())

merged_data['login_date'] = pd.to_datetime(merged_data['login_date'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
default_date = pd.to_datetime("1900-01-01 00:00:00")

# Rellenar las fechas nulas
merged_data['login_date'] = merged_data['login_date'].fillna(default_date)

# Añadir un '0' al inicio si falta en 'wallet_number'
merged_data['wallet_number'] = merged_data['wallet_number'].apply(lambda x: '0' + x if len(x) == 9 else x)

# Eliminar columnas no necesarias
merged_data.drop(['remote_ip', 'amount_bill_payment'], axis=1, inplace=True)

# Eliminar 'amount' de bill_payment_service_data (asegúrate de que 'service_id' esté presente)
bill_payment_service_data.drop('amount', axis=1, inplace=True)

# Agregar el 'service_id' al DataFrame combinado
merged_data = pd.merge(merged_data, bill_payment_service_data[['rrn', 'service_id']], on='rrn', how='left')

# Guardar el DataFrame combinado en un nuevo archivo CSV
merged_data.to_csv('../data/totalDataFrame.csv', index=False)

print('Se ha guardado el nuevo archivo')
