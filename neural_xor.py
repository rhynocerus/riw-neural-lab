import numpy as np

# Datos: dos interruptores y estado esperado de la luz.
X = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=float)
Y = np.array([[0], [1], [1], [0]], dtype=float)

# Pesos iniciales: nuestra red todavía no sabe nada.
rng = np.random.default_rng(42)
W1 = rng.normal(size=(2, 4))
b1 = np.zeros((1, 4))
W2 = rng.normal(size=(4, 1))
b2 = np.zeros((1, 1))

def sigmoide(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

# Entrenamiento mediante descenso de gradiente.
for epoca in range(20000):
    oculta = sigmoide(X @ W1 + b1)
    salida = sigmoide(oculta @ W2 + b2)

    # Error y retropropagación.
    delta2 = (salida - Y) * salida * (1 - salida)
    delta1 = (delta2 @ W2.T) * oculta * (1 - oculta)

    tasa = 1.0
    W2 -= tasa * oculta.T @ delta2 / len(X)
    b2 -= tasa * delta2.mean(axis=0, keepdims=True)
    W1 -= tasa * X.T @ delta1 / len(X)
    b1 -= tasa * delta1.mean(axis=0, keepdims=True)

    if epoca % 4000 == 0:
        error = np.mean((salida - Y) ** 2)
        print(f"Época {epoca:5d} | Error: {error:.6f}")

print("\n=== EXAMEN FINAL ===")
oculta = sigmoide(X @ W1 + b1)
salida = sigmoide(oculta @ W2 + b2)

for entrada, pred, esperado in zip(X, salida, Y):
    resultado = int(pred[0] >= 0.5)
    print(f"{entrada.astype(int)} -> "
          f"{pred[0]:.4f} -> {resultado} "
          f"(esperado: {int(esperado[0])})")

assert np.array_equal((salida >= 0.5).astype(int), Y.astype(int))
print("\nRIW Neural LAB: ¡red entrenada correctamente!")
