import pandas as pd

# Leer el archivo 'unmatched_data.csv'
unmatched_data = pd.read_csv('Ruta de acceso a datos')

# Filtrar los registros no deseados en dos pasos
# Paso 1: Eliminar registros donde 'itc' sea 'tranlog::reverse' o 'tranlog::int_rev'
filtered_data = unmatched_data[~unmatched_data['itc'].isin(['tranlog::reverse', 'tranlog::int_rev'])]

# Paso 2: Eliminar registros donde 'itc' sea 'wcredit' y 'marca_origen' sea 75728.0
final_filtered_data = filtered_data[~((filtered_data['itc'] == 'wcredit') & (filtered_data['marca_origen'] == 75728.0))]

# Guardar el DataFrame limpio en un nuevo archivo CSV
final_filtered_data.to_csv('Ruta de acceso a datos', index=False)

print('Se ha guardado el archivo limpio final')
