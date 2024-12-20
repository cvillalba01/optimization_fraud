import pandas as pd

# Leer el archivo 'final_cleaned_unmatched_data.csv'
final_data = pd.read_csv('../data/final_cleaned_unmatched_data.csv')

# Filtrar las transacciones que deben ser movidas
wtransfer_data = final_data[(final_data['marca_origen'] == 75728.0) & (final_data['itc'] == 'wtransfer')]

# Eliminar esas transacciones del DataFrame principal
remaining_data = final_data[~((final_data['marca_origen'] == 75728.0) & (final_data['itc'] == 'wtransfer'))]

# Guardar las transacciones movidas en un nuevo archivo CSV
wtransfer_data.to_csv('../data/wtransfer_transactions.csv', index=False)

# Guardar el DataFrame restante en el archivo principal
remaining_data.to_csv('../data/final_cleaned_unmatched_data.csv', index=False)

print('Las transacciones han sido movidas y los archivos han sido actualizados')