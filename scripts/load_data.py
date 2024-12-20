import psycopg2
import pandas as pd

# Conectar a la base de datos de PostgreSQL
conn = psycopg2.connect(
    dbname="fraud_detection",
    user="postgres",
    password="postgres",
    host="localhost",  # o la IP de tu servidor de base de datos
    port="5432"        # El puerto por defecto es 5432
)

# Crear un cursor
cur = conn.cursor()

# Leer los archivos CSV en DataFrames de Pandas
total_data = pd.read_csv('Ruta_para_extraer_datos')
wtransfer_data = pd.read_csv('Ruta_para_extraer_datos')

# Insertar datos en la tabla de transacciones
for index, row in total_data.iterrows():
    cur.execute("""
        INSERT INTO transactions (rrn, type, wallet_number, created, jcard_type, amount, service_id, wallet_number_origin, wallet_number_destiny, itc)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (rrn) DO NOTHING;
    """, (
        row['rrn'], row['type'], row['wallet_number'], row['created'], row['jcard_type'], row['amount'], 
        row['service_id'], row['wallet_number_origin'], row['wallet_number_destiny'], row['itc']
    ))

# Insertar datos en la tabla de sesiones
for index, row in total_data.iterrows():
    cur.execute("""
        INSERT INTO session_data (rrn, login_date, application_id, remote_ip_clean)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (rrn) DO NOTHING;
    """, (
        row['rrn'], row['login_date'], row['application_id'], row['remote_ip_clean']
    ))

# Insertar datos en la tabla de transacciones de billeteras de distribución
for index, row in wtransfer_data.iterrows():
    cur.execute("""
        INSERT INTO distribution_wallet_transactions (rrn, distribution_wallet, wallet_destiny, marca_origen, marca_destino, irc, itc)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (rrn) DO NOTHING;
    """, (
        row['rrn'], row['distribution_wallet'], row['wallet_destiny'], row['marca_origen'], 
        row['marca_destino'], row['irc'], row['itc']
    ))

# Hacer commit a la base de datos para guardar los cambios
conn.commit()

# Cerrar el cursor y la conexión
cur.close()
conn.close()

print("Datos insertados exitosamente")
