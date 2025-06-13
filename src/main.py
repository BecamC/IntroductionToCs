import cv2
import numpy as np
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

# Cargar el dataset
digits = load_digits()
data = digits.data
targets = digits.target

# Inicializar matrices para promedios y contadores
promedios = np.zeros((10, 64))
enumerador = np.zeros(10)

# Calcular promedios de los dígitos
for i in range(len(targets)):
    promedios[targets[i]] += data[i]
    enumerador[targets[i]] += 1

promedios = promedios / enumerador[:, None]

# Mostrar matrices de promedios en consola
for i in range(10):
    print(f"Dígito {i} promedio:\n", promedios[i].reshape(8, 8).astype(int))

# Mostrar promedios como imágenes
fig, axs = plt.subplots(2, 5, figsize=(10, 5))
fig.suptitle("Promedios de los Dígitos del 0 al 9")
for i in range(10):
    ax = axs[i // 5, i % 5]
    ax.imshow(promedios[i].reshape(8, 8), cmap='Blues', interpolation='nearest')
    ax.set_title(f"Dígito {i}")
    ax.axis('off')
plt.show()

# Cargar imagen a clasificar (desde carpeta assets)
imagen = cv2.imread("./assets/cero.png", cv2.IMREAD_GRAYSCALE)

# Redimensionar a 8x8
imagen_pequena = cv2.resize(imagen, (8, 8))

# Invertir colores
for i in range(8):
    for j in range(8):
        imagen_pequena[i][j] = 255 - imagen_pequena[i][j]

# Reducir valores de 0–255 a 0–16
for i in range(8):
    for j in range(8):
        imagen_pequena[i][j] = imagen_pequena[i][j] // 16

# Mostrar imagen procesada
print("Imagen pequeña:\n", imagen_pequena)

plt.figure(figsize=(2, 2))
plt.imshow(imagen_pequena, cmap='Blues', interpolation='nearest')
plt.title('Imagen Ingresada (Redimensionada)')
plt.axis('off')
plt.show()

# Aplanar imagen
imagen_pequeña_aplanada = imagen_pequena.flatten()

# Calcular distancias a todas las imágenes del dataset
distancias = np.linalg.norm(data - imagen_pequeña_aplanada, axis=1)
numero_cercano = np.argsort(distancias)[:3]
digitos_mas_cercanos = targets[numero_cercano]

# Imprimir los 3 dígitos más cercanos
print("Los 3 numeros mas cercanos son:", digitos_mas_cercanos)

# Clasificación por voto
unique, counts = np.unique(digitos_mas_cercanos, return_counts=True)
if np.any(counts >= 2):
    numero_final = unique[np.argmax(counts)]
    print(f"Soy la inteligencia artificial, y he detectado que el dígito ingresado corresponde al número {numero_final}")
else:
    distancias_promedias = np.linalg.norm(promedios - imagen_pequeña_aplanada, axis=1)
    numero_final = np.argmin(distancias_promedias)
    print(f"Soy la inteligencia artificial versión 2, y he detectado que el dígito ingresado corresponde al número {numero_final}")

# Imprimir resultados
print(f"Versión 1: {numero_final}")
print(f"Versión 2: {numero_final}")
