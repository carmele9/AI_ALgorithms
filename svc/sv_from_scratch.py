import numpy as np

# Inicializar pesos
#        │
#        ▼
# Recorrer todas las muestras
#        │
#        ▼
# Calcular: yᵢ(w·xᵢ - b)
#        │
#        ├───────────────┐
#        │               │
#      ≥ 1             < 1
#        │               │
#        ▼               ▼
# Solo regularizar   Regularizar + corregir error
#        │               │
#        └───────┬───────┘
#                ▼
#       Siguiente muestra
#                │
#                ▼
#     Repetir durante n_iters épocas

import numpy as np

class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        """
        Constructor de la clase SVM.

        Parámetros
        ----------
        learning_rate : float
            Tamaño del paso en cada actualización del Gradient Descent.

        lambda_param : float
            Parámetro de regularización L2. Cuanto mayor sea, más se penalizan
            los pesos grandes para evitar overfitting.

        n_iters : int
            Número de épocas (veces que recorreremos todo el dataset).
        """

        # Guardamos los hiperparámetros
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters

        # Los pesos y el bias se inicializarán durante el entrenamiento
        self.w = None
        self.b = None

    def fit(self, X, y):
        """
        Entrena una SVM lineal utilizando Stochastic Gradient Descent (SGD).

        La SVM busca encontrar el hiperplano que maximiza el margen entre
        las dos clases minimizando la función Hinge Loss.
        """

        # Número de muestras (filas) y número de variables (columnas)
        n_samples, n_features = X.shape

        # Las SVM trabajan con etiquetas {-1, +1}
        # Si la etiqueta es menor que 0 -> -1
        # En cualquier otro caso -> +1
        y_ = np.where(y < 0, -1, 1)

        # Inicializamos todos los pesos a cero
        self.w = np.zeros(n_features)

        # Inicializamos el bias
        self.b = 0

        # Recorremos el dataset varias veces
        for _ in range(self.n_iters):

            # En SGD actualizamos los pesos muestra a muestra
            for idx, x_i in enumerate(X):

                # Calculamos el margen:
                #
                # y * (w·x - b)
                #
                # Si es mayor o igual que 1:
                #   -> la muestra está correctamente clasificada
                #      y fuera del margen.
                #
                # Si es menor que 1:
                #   -> está mal clasificada o dentro del margen.
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1

                # Caso 1:
                # La muestra ya está correctamente clasificada.
                if condition:

                    # Solo aplicamos regularización para evitar
                    # que los pesos crezcan demasiado.
                    self.w -= self.lr * (2 * self.lambda_param * self.w)

                # Caso 2:
                # La muestra está mal clasificada
                # o dentro del margen.
                else:

                    # Actualizamos los pesos teniendo en cuenta:
                    #
                    # 1. Regularización
                    # 2. Error de clasificación
                    self.w -= self.lr * (
                        2 * self.lambda_param * self.w
                        - x_i * y_[idx]
                    )

                    # También actualizamos el bias
                    self.b -= self.lr * y_[idx]

    def predict(self, X):
        """
        Realiza predicciones sobre nuevos datos.

        Devuelve:

            +1  -> clase positiva
            -1  -> clase negativa
        """

        # Calculamos la función de decisión
        #
        # f(x) = w·x - b
        #
        # El signo indica el lado del hiperplano
        # donde cae cada observación.
        approx = np.dot(X, self.w) - self.b

        # np.sign devuelve:
        #
        # positivo -> 1
        # negativo -> -1
        # cero     -> 0
        return np.sign(approx)