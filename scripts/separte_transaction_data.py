import pandas as pd

# Ruta del archivo CSV original
input_file = 'Ruta de acceso a datos'

# Ruta del archivo CSV reducido
output_file = 'Ruta de acceso a datos'

# Leer el archivo CSV, asegurándose de que service_id sea tratado como float
# Se setean los tipos de datos a string y float.
df = pd.read_csv(input_file, dtype={'wallet_number': str,'service_id': 'float'})

# Convertir valores no nulos a enteros, manteniendo NaN
df['service_id'] = df['service_id'].apply(lambda x: int(x) if pd.notna(x) else pd.NA)


# Seleccionar las columnas necesarias para la tabla transactions
columns_needed = [
    'rrn', 
    'type', 
    'wallet_number', 
    'created', 
    'jcard_type', 
    'amount', 
    'service_id', 
    'wallet_number_origin', 
    'wallet_number_destiny', 
    'itc'
]

# Verificar si todas las columnas necesarias están en el archivo CSV
missing_cols = [col for col in columns_needed if col not in df.columns]
if missing_cols:
    raise ValueError(f"Las siguientes columnas están faltando en el archivo CSV: {', '.join(missing_cols)}")

# Filtrar el DataFrame para mantener solo las columnas necesarias
df_reduced = df[columns_needed]

# Guardar el DataFrame reducido en un nuevo archivo CSV
df_reduced.to_csv(output_file, index=False)

print(f"Archivo CSV reducido guardado en: {output_file}")


# Contar las repeticiones de cada rrn
rrn_counts = df_reduced['rrn'].value_counts()

# Filtrar rrn que están repetidos
rrn_repeated = rrn_counts[rrn_counts > 1]

# Cantidad total de rrn repetidos
num_repeated_rrns = rrn_repeated.count()

# Mostrar los resultados
print(f"Cantidad de rrn repetidos: {num_repeated_rrns}")
print("Primeros 10 rrn repetidos y sus conteos:")
print(rrn_repeated)