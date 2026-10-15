"""
pipeline_roto.py
Reto B del módulo 1.2. Este script normaliza un dataset y deja el conjunto
de prueba contaminado. La tarea consiste en localizar la línea culpable,
demostrar la contaminación y corregirla con el mínimo cambio posible.

Ejecutar con:  python pipeline_roto.py
"""
import numpy as np

SEMILLA = 42
rng = np.random.default_rng(SEMILLA)


def cargar_datos(n_filas=10_000, n_columnas=20):
    """Simula un dataset con escalas muy distintas entre columnas."""
    escalas = rng.uniform(1, 500, size=n_columnas)
    return rng.normal(loc=100, scale=escalas, size=(n_filas, n_columnas))


def preparar(X, corte=8000):
    """Parte el dataset y centra el bloque de entrenamiento."""
    entrenamiento = X[:corte]
    prueba = X[corte:]

    entrenamiento -= entrenamiento.mean(axis=0)
    entrenamiento /= entrenamiento.std(axis=0)

    return entrenamiento, prueba


if __name__ == "__main__":
    X = cargar_datos()

    media_original = X.mean()
    print(f"Media del dataset antes de preparar : {media_original:10.4f}")

    entrenamiento, prueba = preparar(X)

    print(f"Media del dataset despues de preparar: {X.mean():10.4f}")
    print(f"Media del bloque de entrenamiento    : {entrenamiento.mean():10.4f}")
    print(f"Media del bloque de prueba           : {prueba.mean():10.4f}")
    print()
    print("Pistas para la investigacion:")
    print(f"  entrenamiento comparte memoria con X: {np.shares_memory(X, entrenamiento)}")
    print(f"  prueba comparte memoria con X       : {np.shares_memory(X, prueba)}")
    print(f"  entrenamiento.base is X             : {entrenamiento.base is X}")
