"""
Generador de los datos del Día 6 del curso de Machine Learning con Python.

El Día 6 vuelve a la pregunta del Día 4, quién abandona, y la convierte en un problema de
clasificación de verdad. Para eso basta con los cuatro ficheros que guardó el guiado 4.2
(datos/X_train.parquet, X_test, y_train e y_test). Este script añade dos cosas que esos ficheros
no traen.

1. La tabla datos/clientes_dia6.parquet, que es la del Día 5 con una columna nueva:

    contracargo   True si el cliente ha pedido a su banco que anule un cargo de tarjeta en los
                  seis meses siguientes. La tienda pierde el importe y paga además una comisión.
                  Le pasa a uno de cada treinta y ocho clientes, el 2,6 %, y es el
                  caso con el que estudiamos las clases desbalanceadas en el módulo 6.2.

2. El fichero datos/verdad_abandono.parquet, con la probabilidad de abandono que usó
   datos_dia4.py para decidir quién abandona. Ningún modelo puede conocerla en un proyecto real.
   La usamos una sola vez, en la slide 09, para saber cuál es la mejor nota alcanzable con
   estos datos. Nunca entra como columna en un modelo, porque sería la respuesta.

Si no existen los ficheros del Día 5 o del Día 4, el script ejecuta antes los generadores de
esos días, que a su vez reconstruyen lo que falte del Día 3. Si no existen los cuatro ficheros del
guiado 4.2, los genera con el mismo código del cuaderno de soluciones del Día 4.

Los datos son sintéticos y reproducibles con la semilla 6. El contracargo sale de un modelo
conocido, escrito en generar_contracargo, para poder comprobar qué columnas encuentra cada
clasificador.

Uso:
    python datos_dia6.py

Requisitos: numpy, pandas, pyarrow y scikit-learn. Necesita datos_dia3.py, datos_dia4.py y
datos_dia5.py en la misma carpeta.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATOS = Path("datos")
FIN_HISTORICO = pd.Timestamp("2025-12-31")

# Coeficientes del modelo que genera el contracargo, en la escala del logit.
VERDAD_CONTRACARGO = {
    "constante": -5.10,
    "log_ticket_100": 1.30,       # por cada vez que el ticket medio se multiplica por e, sobre 100 €
    "devoluciones": 9.00,         # proporción de pedidos devueltos, entre 0 y 1
    "alta_reciente": 1.40,        # cliente dado de alta en el último año del histórico
    "ingresos_falta": 1.00,       # ficha sin ingresos declarados
    "canal_web": 0.70,
    "log_pedidos": -0.50,         # centrado en la media
}


def asegurar_dia5():
    if not (DATOS / "clientes_dia5.parquet").exists():
        subprocess.run([sys.executable, "datos_dia5.py"], check=True)


def probabilidad_abandono_real():
    """Repite datos_dia4.py paso a paso y devuelve la probabilidad que usó, antes del sorteo."""
    from datos_dia4 import generar_codigos_postales

    rng = np.random.default_rng(4)
    df = pd.read_parquet(DATOS / "clientes_features.parquet")
    df["codigo_postal"] = generar_codigos_postales(rng, df["provincia"].astype(str).to_numpy())
    df["codigo_postal"] = df["codigo_postal"].astype("str")
    edad_c = (df["edad"] - 41) / 12
    efecto_cp = {c: rng.normal(0, 0.7) for c in df["codigo_postal"].unique()}
    logit = (-2.05
             + 1.10 * np.log1p(df["recencia"]) - 1.10 * np.log1p(df["recencia"]).mean()
             - 0.45 * np.log(df["pedidos"]) + 0.45 * np.log(df["pedidos"]).mean()
             + 4.0 * df["devoluciones"]
             - 0.55 * (df["nota_media"] - 4)
             + 0.85 * edad_c ** 2
             + 0.35 * (df["canal"] == "tienda")
             + df["codigo_postal"].map(efecto_cp).astype(float))
    p = 1 / (1 + np.exp(-logit))
    sorteo = rng.random(len(df)) < p
    return pd.DataFrame({"p_abandono": p, "abandona": sorteo}, index=df.index)


def generar_contracargo(rng, df):
    """Modelo conocido del contracargo. Lineal en el logit, sin interacciones."""
    v = VERDAD_CONTRACARGO
    antiguedad = (FIN_HISTORICO - pd.to_datetime(df["fecha_alta"])).dt.days / 365.25
    logit = (v["constante"]
             + v["log_ticket_100"] * np.log(df["ticket_medio"].clip(lower=10) / 100)
             + v["devoluciones"] * df["devoluciones"]
             + v["alta_reciente"] * (antiguedad < 1)
             + v["ingresos_falta"] * df["ingresos_falta"]
             + v["canal_web"] * (df["canal"] == "web")
             + v["log_pedidos"] * (np.log(df["pedidos"]) - np.log(df["pedidos"]).mean()))
    p = 1 / (1 + np.exp(-logit))
    return rng.random(len(df)) < p


def asegurar_ficheros_guiado_42(clientes):
    """Los cuatro ficheros del guiado 4.2, con el mismo código del cuaderno del Día 4."""
    if all((DATOS / f).exists() for f in
           ["X_train.parquet", "X_test.parquet", "y_train.parquet", "y_test.parquet"]):
        return
    from sklearn.compose import ColumnTransformer
    from sklearn.model_selection import StratifiedKFold, train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import (FunctionTransformer, KBinsDiscretizer, OneHotEncoder,
                                       OrdinalEncoder, RobustScaler, StandardScaler,
                                       TargetEncoder)

    base = clientes.drop(columns=["gasto_6m", "contracargo"], errors="ignore")
    train, test = train_test_split(base, test_size=0.25, random_state=4,
                                   stratify=base["abandona"])
    y_train, y_test = train["abandona"].astype(int), test["abandona"].astype(int)
    cv = StratifiedKFold(5, shuffle=True, random_state=4)
    log = FunctionTransformer(np.log1p, feature_names_out="one-to-one")
    preparar = ColumnTransformer([
        ("num",  StandardScaler(), ["edad", "ingresos", "ticket_medio", "nota_media"]),
        ("log",  make_pipeline(log, StandardScaler()), ["pedidos", "recencia"]),
        ("rob",  RobustScaler(), ["gasto", "devoluciones"]),
        ("cat",  OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                 ["canal", "favorita", "provincia"]),
        ("ord",  OrdinalEncoder(categories=[["C", "B", "A"]]), ["segmento"]),
        ("cp",   TargetEncoder(smooth=20, cv=cv), ["codigo_postal"]),
        ("edad", KBinsDiscretizer(n_bins=8, encode="onehot-dense"), ["edad"]),
    ]).set_output(transform="pandas")
    preparar.fit_transform(train, y_train).to_parquet(DATOS / "X_train.parquet")
    preparar.transform(test).to_parquet(DATOS / "X_test.parquet")
    y_train.to_frame().to_parquet(DATOS / "y_train.parquet")
    y_test.to_frame().to_parquet(DATOS / "y_test.parquet")
    print("ficheros del guiado 4.2 regenerados")


def main():
    asegurar_dia5()
    df = pd.read_parquet(DATOS / "clientes_dia5.parquet")

    rng = np.random.default_rng(6)
    df["contracargo"] = generar_contracargo(rng, df)
    df.to_parquet(DATOS / "clientes_dia6.parquet")

    verdad = probabilidad_abandono_real().loc[df.index]
    assert (verdad["abandona"] == df["abandona"]).all(), "la verdad no reproduce datos_dia4.py"
    verdad[["p_abandono"]].to_parquet(DATOS / "verdad_abandono.parquet")

    asegurar_ficheros_guiado_42(df)

    print(f"clientes_dia6.parquet  {df.shape[0]} filas, {df.shape[1]} columnas")
    print(f"tasa de abandono: {df['abandona'].mean():.4f}")
    print(f"tasa de contracargo: {df['contracargo'].mean():.4f} "
          f"({df['contracargo'].sum()} clientes)")
    print("verdad_abandono.parquet  solo para la slide 09, nunca como columna de un modelo")


if __name__ == "__main__":
    main()
