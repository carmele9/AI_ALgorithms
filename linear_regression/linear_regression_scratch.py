import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. CREACIÓN DEL DATASET
np.random.seed(10)
X = np.random.uniform(-20, 80, 400)
noise = np.random.normal(0, 100, 400)
y = 5.5 * X - 12 + noise
df = pd.DataFrame({
    "X": X,
    "y": y
})

# Visualización
plt.scatter(df["X"], df["y"])
plt.show()


# ============================
# IMPLEMENTACIÓN DE LA REGRESIÓN LINEAL DESDE CERO
# ============================

# Función de pérdida (Loss Function)
# Calcula el Error Cuadrático Medio (MSE).
# Cuanto más pequeño sea este valor, mejor se ajusta la recta a los datos.
def loss_function(m, b, points):

    # Variable donde iremos acumulando el error total
    total_error = 0

    # Recorremos todos los puntos del dataset
    for i in range(len(points)):

        # Obtenemos la variable independiente (X)
        x = points.iloc[i].X

        # Obtenemos el valor real (y)
        y = points.iloc[i].y

        # Predicción de nuestro modelo:
        # y_pred = m*x + b
        prediction = m * x + b

        # Sumamos el error cuadrático
        total_error += (y - prediction) ** 2

    # Calculamos el error medio
    return total_error / float(len(points))

# Gradient Descent
# Ajusta poco a poco los parámetros m (pendiente)
# y b (intercepto) para minimizar el error.
def gradient_descent(m_now, b_now, points, L):

    # Gradiente respecto a la pendiente
    m_gradient = 0

    # Gradiente respecto al intercepto
    b_gradient = 0

    # Número de muestras
    n = len(points)

    # Recorremos todos los datos
    for i in range(n):

        x = points.iloc[i].X
        y = points.iloc[i].y

        # Calculamos cuánto debe cambiar la pendiente
        m_gradient += -(2/n) * x * (y - (m_now * x + b_now))

        # Calculamos cuánto debe cambiar el intercepto
        b_gradient += -(2/n) * (y - (m_now * x + b_now))

    # Actualizamos la pendiente
    m = m_now - L * m_gradient

    # Actualizamos el intercepto
    b = b_now - L * b_gradient

    return m, b

###### Entrenamiento

# Pendiente inicial
m = 0

# Intercepto inicial
b = 0

# Learning Rate
# Tamaño del paso que damos en cada iteración
L = 0.0001

# Número de iteraciones
epochs = 1000

# Entrenamiento
for i in range(epochs):

    # Mostrar el progreso cada 50 iteraciones
    if i % 50 == 0:
        print(f"Epoch: {i}")

    # Actualizar m y b usando Gradient Descent
    m, b = gradient_descent(m, b, df, L)

# Mostrar los parámetros finales
print("Pendiente:", m)
print("Intercepto:", b)

# Dibujamos los puntos originales
plt.scatter(df.X, df.y, color="black")

# Dibujamos la recta aprendida
plt.plot(
    df.X,
    m * df.X + b,
    color="red"
)

plt.xlabel("X")
plt.ylabel("y")
plt.title("Regresión Lineal")
plt.show()



