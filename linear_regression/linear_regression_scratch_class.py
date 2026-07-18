import numpy as np

class LinearRegressionScratch:

    def __init__(self):
        self.weights = None
        self.bias = None

    def predict(self, X):
        """
        Realiza las predicciones utilizando la ecuación:

        y = X·w + b

        Parámetros
        ----------
        X : ndarray
            Variables independientes.

        Devuelve
        --------
        y_pred : ndarray
            Predicciones del modelo.
        """

        return np.dot(X, self.weights) + self.bias
    
    def fit_batch_gradient_descent(self, X, y, learning_rate=0.01, epochs=1000):
        """
        Entrena una regresión lineal utilizando Batch Gradient Descent.

        En cada iteración se utilizan TODAS las muestras del dataset
        para calcular el gradiente.

        Es el algoritmo más estable, aunque también el más lento cuando
        el dataset es muy grande.
        """

        # Número de muestras y variables
        n_samples, n_features = X.shape

        # Inicializamos pesos y bias en cero
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Entrenamiento
        for epoch in range(epochs):

            # Predicciones actuales
            y_pred = np.dot(X, self.weights) + self.bias

            # Error
            error = y_pred - y

            # Derivadas parciales de la función de coste
            dw = (2 / n_samples) * np.dot(X.T, error)
            db = (2 / n_samples) * np.sum(error)

            # Actualización de parámetros
            self.weights -= learning_rate * dw
            self.bias -= learning_rate * db

    def fit_stochastic_gradient_descent(self, X, y, learning_rate=0.01, epochs=100):
        """
        Entrena utilizando Stochastic Gradient Descent.

        En lugar de utilizar todo el dataset,
        actualiza los pesos utilizando UNA sola muestra.

        Es mucho más rápido aunque las actualizaciones
        son bastante más ruidosas.
        """

        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0

        for epoch in range(epochs):

            # Recorremos muestra por muestra
            for i in range(n_samples):

                x_i = X[i]
                y_i = y[i]

                # Predicción para una única muestra
                y_pred = np.dot(x_i, self.weights) + self.bias

                error = y_pred - y_i

                # Gradientes
                dw = 2 * x_i * error
                db = 2 * error

                # Actualización inmediata
                self.weights -= learning_rate * dw
                self.bias -= learning_rate * db

    def fit_mini_batch_gradient_descent(self, X, y, learning_rate=0.01, epochs=100, batch_size=32):
        """
        Entrena utilizando Mini Batch Gradient Descent.

        El dataset se divide en pequeños lotes (batches).

        Cada actualización utiliza únicamente un batch.

        Es el algoritmo más utilizado actualmente en Machine Learning.
        """

        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0

        for epoch in range(epochs):

            # Mezclamos el dataset en cada época
            permutation = np.random.permutation(n_samples)

            X_shuffled = X[permutation]
            y_shuffled = y[permutation]

            # Recorremos el dataset por lotes
            for i in range(0, n_samples, batch_size):

                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]

                # Predicciones del lote
                y_pred = np.dot(X_batch, self.weights) + self.bias

                error = y_pred - y_batch

                m = len(X_batch)

                dw = (2 / m) * np.dot(X_batch.T, error)
                db = (2 / m) * np.sum(error)

                # Actualización
                self.weights -= learning_rate * dw
                self.bias -= learning_rate * db

    def fit_normal_equation(self, X, y):
        """
        Entrena utilizando la ecuación normal.

        No necesita Gradient Descent.

        Calcula directamente los pesos óptimos mediante
        álgebra lineal.

        Es muy rápido para datasets pequeños,
        pero no escala bien cuando el número de variables
        es muy elevado.
        """

        # Añadimos una columna de unos para representar el bias
        X_b = np.c_[np.ones((X.shape[0], 1)), X]

        # Aplicamos la ecuación normal
        theta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y

        # Separamos bias y pesos
        self.bias = theta[0]
        self.weights = theta[1:]