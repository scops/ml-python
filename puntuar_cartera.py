"""Puntúa una cartera de clientes con los modelos del proyecto integrador.

Uso:
    uv run python puntuar_cartera.py datos/cartera.parquet resultados/puntuaciones.parquet
"""
import sys
from pathlib import Path

import joblib
import pandas as pd

from utilidades import log_pos, puntuar  # noqa: F401  gasto.joblib busca log_pos en __main__


def main(entrada, salida):
    clientes = pd.read_parquet(entrada)
    abandono = joblib.load("modelos/abandono.joblib")
    gasto = joblib.load("modelos/gasto.joblib")
    segmentos = joblib.load("modelos/segmentos.joblib")
    smear = joblib.load("modelos/smear.joblib")
    puntos = puntuar(clientes, abandono, gasto, segmentos, smear)
    Path(salida).parent.mkdir(exist_ok=True)
    puntos.to_parquet(salida)
    print(f"{len(puntos)} clientes puntuados, {int(puntos['llamar'].sum())} llamadas, "
          f"valor esperado {round(puntos['valor_esperado'].sum())} €")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
