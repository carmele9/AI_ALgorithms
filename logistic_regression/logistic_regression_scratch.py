import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# ----------------------------------------------------
# Función Sigmoide
# ----------------------------------------------------
def sigmoid(z):
    """
    Convierte cualquier número real en una probabilidad
    comprendida entre 0 y 1.
    """

    # Evitamos overflow de exp()
    z = np.clip(z, -500, 500)

    return 1 / (1 + np.exp(-z))


# ----------------------------------------------------
# Cálculo del gradiente
# ----------------------------------------------------
def calculate_gradient(theta, X, y):
    """
    Calcula el gradiente de la función de coste.
    Parámetros
    ----------
    theta : pesos actuales
    X : matriz de características (ya incluye bias)
    y : etiquetas reales
    Devuelve
    --------
    Gradiente respecto a todos los parámetros.
    """

    m = len(y)

    # Predicciones actuales
    predictions = sigmoid(X @ theta)

    # Error cometido
    error = predictions - y

    # Gradiente
    gradient = (X.T @ error) / m

    return gradient


# ----------------------------------------------------
# Gradient Descent
# ----------------------------------------------------
def gradient_descent( X, y, alpha=0.1, num_iter=1000, tol=1e-7):
    """
    Entrena la regresión logística utilizando
    Batch Gradient Descent.
    """

    # Añadimos columna de unos (bias)
    X_b = np.c_[np.ones((X.shape[0], 1)), X]

    # Inicializamos pesos
    theta = np.zeros(X_b.shape[1])

    # Entrenamiento
    for i in range(num_iter):
        grad = calculate_gradient(theta, X_b, y)
        theta -= alpha * grad

        # Si el gradiente prácticamente no cambia, asumimos convergencia.
        if np.linalg.norm(grad) < tol:
            print(f"Convergencia alcanzada en iteración {i}")
            break

    return theta


# ----------------------------------------------------
# Predicción de probabilidades
# ----------------------------------------------------
def predict_proba(X, theta):
    """
    Devuelve P(y=1|x)
    """
    X_b = np.c_[np.ones((X.shape[0], 1)), X]

    return sigmoid(X_b @ theta)


# ----------------------------------------------------
# Predicción final
# ----------------------------------------------------
def predict(X, theta, threshold=0.5):
    """
    Convierte probabilidades en clases. Si -> P >= 0.5 entonces clase = 1 si no clase = 0
    """

    probabilities = predict_proba(X, theta)

    return (probabilities >= threshold).astype(int)

X, y = load_breast_cancer(return_X_y= True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

theta_hat = gradient_descent(X_train, y_train, alpha=0.001)

y_pred_train = predict (X_train_scaled, theta_hat)
y_pred_test = predict(X_test, theta_hat)

train_acc = accuracy_score(y_train, y_pred_train)
test_acc = accuracy_score(y_test, y_pred_test)

print(train_acc)
print(test_acc)