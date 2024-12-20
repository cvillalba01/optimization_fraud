import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Datos proporcionados
data = pd.read_csv('../test/desviacion_estandar.csv')
# Crear el DataFrame
df = pd.DataFrame(data)

# Configuración de la gráfica
plt.figure(figsize=(14, 8))
sns.barplot(x="rrn", y="stddev_amount_up_to_now", data=df)

# Personalización
plt.title("Desviación Estándar de las Transacciones por RRN")
plt.xlabel("RRN")
plt.ylabel("Desviación Estándar")
plt.xticks(rotation=45, ha="right")  # Rotar las etiquetas del eje X y alinearlas a la derecha
plt.tight_layout()

# Guardar la gráfica en un archivo
plt.savefig("/home/avelazquez/projects/fraud-detection-pry/test/desviacion_estandar_fraude.png")
