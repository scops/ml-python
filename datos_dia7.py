"""
Generador de los datos del Día 7 del curso de Machine Learning con Python.

El Día 7 tiene dos mitades y cada una necesita algo distinto.

1. El módulo 7.1 estudia el clustering, que busca grupos sin que nadie diga cuáles son. Para
   comprobar si un algoritmo los encuentra hace falta saber de antemano qué grupos hay, y eso solo
   es posible con datos generados. Este script crea datos/sesiones.parquet, 12 000 visitas a la web
   de la tienda, una por fila:

    hora               Hora de inicio de la visita, de 0 a 23.
    duracion_s         Segundos que dura la visita.
    paginas            Páginas vistas.
    busquedas          Veces que se usa el buscador.
    prop_producto      Proporción de las páginas vistas que son fichas de producto, entre 0 y 1.
    articulos_carrito  Artículos añadidos al carrito.
    compra             True si la visita termina en un pedido.

   Cada visita sale de uno de cuatro perfiles, escritos más abajo en PERFILES: curiosa, comparadora,
   decidida y bot. El perfil se guarda aparte, en datos/verdad_sesiones.parquet, y no entra nunca en
   un algoritmo de clustering, porque sería la respuesta. Lo usamos solo para medir cuánto se parecen
   los grupos encontrados a los verdaderos.

2. El módulo 7.2 es el proyecto integrador, que combina los dos modelos guardados en los días 5 y 6
   para estimar cuánto vale cada cliente. Para medir en euros una campaña de retención hace falta
   saber cuánto habría gastado un cliente que se fue si se hubiera quedado, algo que en la realidad
   nadie sabe. datos_dia5.py lo calculó para todos los clientes y después puso a cero a quien
   abandona. Este script repite ese cálculo y lo guarda en datos/verdad_gasto.parquet, en la columna
   gasto_si_se_queda. Igual que la otra verdad, solo sirve para medir.

Además, si faltan modelos/gasto.joblib o modelos/abandono.joblib, los reconstruye con el mismo código
de los cuadernos de soluciones de los días 5 y 6. Si faltan las tablas de días anteriores, llama a
sus generadores.

Los datos son sintéticos y reproducibles con la semilla 7.

Uso:
    python datos_dia7.py

Requisitos: numpy, pandas, pyarrow, scikit-learn y joblib. Necesita datos_dia3.py a datos_dia6.py
en la misma carpeta.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATOS = Path("datos")
MODELOS = Path("modelos")

# Los cuatro perfiles de visita. Las cifras son medianas aproximadas y probabilidades.
PERFILES = {
    #               peso  duración(s) páginas búsquedas  prop. producto  carrito  compra
    "curiosa":     (0.40,  110,        5,      0.4,      (3, 5),         0.1,     0.03),
    "comparadora": (0.30,  840,        28,     5.0,      (8, 2),         1.2,     0.25),
    "decidida":    (0.25,  260,        7,      1.0,      (6, 4),         1.2,     0.70),
    "bot":         (0.05,  2400,       700,    0.05,     (30, 2),        0.0,     0.00),
}


def generar_sesiones(rng, n=12_000):
    """Visitas con perfil conocido. Duraciones y páginas con cola larga, como en una web real."""
    nombres = list(PERFILES)
    pesos = np.array([PERFILES[p][0] for p in nombres])
    perfil = rng.choice(nombres, size=n, p=pesos)
    cols = {c: np.empty(n) for c in ["hora", "duracion_s", "paginas", "busquedas",
                                     "prop_producto", "articulos_carrito", "compra"]}
    for nombre, (_, dur, pag, bus, (a, b), car, compra) in PERFILES.items():
        m = perfil == nombre
        k = int(m.sum())
        cols["duracion_s"][m] = np.exp(rng.normal(np.log(dur), 0.5, k)).round()
        cols["paginas"][m] = np.maximum(1, np.exp(rng.normal(np.log(pag), 0.45, k)).round())
        cols["busquedas"][m] = rng.poisson(bus, k)
        cols["prop_producto"][m] = rng.beta(a, b, k).round(3)
        cols["articulos_carrito"][m] = rng.poisson(car, k) + (nombre == "decidida")
        cols["compra"][m] = rng.random(k) < compra
        if nombre == "bot":                                  # los bots no duermen
            cols["hora"][m] = rng.integers(0, 24, k)
        else:                                                # las personas, con dos picos
            pico = rng.choice([11, 21], size=k, p=[0.45, 0.55])
            cols["hora"][m] = np.round(rng.normal(pico, 2.5)) % 24
    df = pd.DataFrame(cols)
    for c in ["hora", "duracion_s", "paginas", "busquedas", "articulos_carrito"]:
        df[c] = df[c].astype(int)
    df["compra"] = df["compra"].astype(bool)
    df.index = pd.RangeIndex(1, n + 1, name="id_sesion")
    verdad = pd.DataFrame({"perfil": perfil}, index=df.index)
    return df, verdad


def gasto_si_se_queda(clientes):
    """Repite generar_gasto_6m de datos_dia5.py sin poner a cero a quien abandona."""
    from datos_dia5 import VERDAD as v

    rng = np.random.default_rng(5)
    df = clientes
    efecto_cp = {c: rng.normal(0, v["sd_codigo_postal"]) for c in sorted(df["codigo_postal"].unique())}
    log_g = (v["constante"]
             + v["log_pedidos"] * np.log(df["pedidos"])
             + v["log_ticket"] * np.log(df["ticket_medio"].clip(lower=10))
             + v["log1p_recencia"] * np.log1p(df["recencia"])
             + v["devoluciones"] * df["devoluciones"]
             + v["nota_media"] * (df["nota_media"] - 4)
             + v["ingresos_10k"] * (df["ingresos"] - df["ingresos"].mean()) / 10_000
             + v["edad_curva"] * ((df["edad"] - 52) / 12) ** 2
             + v["canal_app"] * (df["canal"] == "app")
             + v["canal_tienda"] * (df["canal"] == "tienda")
             + df["codigo_postal"].map(efecto_cp).astype(float)
             + rng.normal(0, v["sd_ruido"], len(df)))
    return np.exp(log_g).round(2)


def log_pos(x):
    """La misma función del Día 5. modelos/gasto.joblib guarda una referencia a ella."""
    return np.log(np.clip(x, 1, None))


def reconstruir_modelos(clientes):
    """Guiado 5.2 y guiado 6.2, con los hiperparámetros que eligieron sus búsquedas."""
    import joblib
    from sklearn.compose import ColumnTransformer
    from sklearn.frozen import FrozenEstimator
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.metrics import make_scorer
    from sklearn.model_selection import (FixedThresholdClassifier, KFold, StratifiedKFold,
                                         TunedThresholdClassifierCV, train_test_split)
    from sklearn.pipeline import Pipeline, make_pipeline
    from sklearn.preprocessing import (FunctionTransformer, KBinsDiscretizer, OneHotEncoder,
                                       OrdinalEncoder, RobustScaler, StandardScaler,
                                       TargetEncoder)

    MODELOS.mkdir(exist_ok=True)
    train, _ = train_test_split(clientes, test_size=0.25, random_state=4,
                                stratify=clientes["abandona"])
    comunes = [("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ["canal", "favorita", "provincia"]),
               ("ord", OrdinalEncoder(categories=[["C", "B", "A"]]), ["segmento"])]

    if not (MODELOS / "gasto.joblib").exists():
        quedan = train[~train["abandona"]]
        prep = ColumnTransformer([
            ("num", StandardScaler(), ["edad", "ingresos", "nota_media"]),
            ("log", make_pipeline(FunctionTransformer(log_pos, feature_names_out="one-to-one"),
                                  StandardScaler()), ["pedidos", "ticket_medio", "gasto"]),
            ("rec", make_pipeline(FunctionTransformer(np.log1p, feature_names_out="one-to-one"),
                                  StandardScaler()), ["recencia"]),
            ("rob", RobustScaler(), ["devoluciones"]),
            *comunes,
            ("cp", TargetEncoder(smooth=20, cv=KFold(5, shuffle=True, random_state=5)),
             ["codigo_postal"]),
            ("edad", KBinsDiscretizer(n_bins=32, encode="onehot-dense"), ["edad"]),
        ]).set_output(transform="pandas")
        modelo = Pipeline([("prep", prep), ("ridge", Ridge(alpha=0.1))])
        modelo.fit(quedan, np.log(quedan["gasto_6m"]))
        joblib.dump(modelo, MODELOS / "gasto.joblib")
        print("modelos/gasto.joblib reconstruido")

    if not (MODELOS / "abandono.joblib").exists():
        y = train["abandona"].astype(int)
        prep = ColumnTransformer([
            ("num", StandardScaler(), ["edad", "ingresos", "ticket_medio", "nota_media"]),
            ("log", make_pipeline(FunctionTransformer(np.log1p, feature_names_out="one-to-one"),
                                  StandardScaler()), ["pedidos", "recencia"]),
            ("rob", RobustScaler(), ["gasto", "devoluciones"]),
            *comunes,
            ("cp", TargetEncoder(smooth=20, cv=StratifiedKFold(5, shuffle=True, random_state=4)),
             ["codigo_postal"]),
            ("edad", KBinsDiscretizer(n_bins=12, encode="onehot-dense"), ["edad"]),
        ]).set_output(transform="pandas")
        mejor = Pipeline([("prep", prep), ("clf", LogisticRegression(C=10, max_iter=3000))])
        mejor.fit(train, y)

        def coste(yv, pred):
            yv, pred = np.asarray(yv), np.asarray(pred)
            return int(15 * (pred == 1).sum() + 75 * ((pred == 0) & (yv == 1)).sum())

        decisor = TunedThresholdClassifierCV(
            mejor, scoring=make_scorer(lambda a, b: -coste(a, b)),
            cv=StratifiedKFold(5, shuffle=True, random_state=6), random_state=6).fit(train, y)
        final = FixedThresholdClassifier(FrozenEstimator(mejor), threshold=decisor.best_threshold_)
        joblib.dump(final.fit(train, y), MODELOS / "abandono.joblib")
        print("modelos/abandono.joblib reconstruido")


def main():
    if not (DATOS / "clientes_dia6.parquet").exists():
        subprocess.run([sys.executable, "datos_dia6.py"], check=True)
    clientes = pd.read_parquet(DATOS / "clientes_dia6.parquet")

    rng = np.random.default_rng(7)
    sesiones, verdad = generar_sesiones(rng)
    sesiones.to_parquet(DATOS / "sesiones.parquet")
    verdad.to_parquet(DATOS / "verdad_sesiones.parquet")

    g = gasto_si_se_queda(clientes)
    quedan = ~clientes["abandona"]
    assert np.allclose(g[quedan], clientes.loc[quedan, "gasto_6m"]), "no reproduce datos_dia5.py"
    pd.DataFrame({"gasto_si_se_queda": g}, index=clientes.index).to_parquet(
        DATOS / "verdad_gasto.parquet")

    reconstruir_modelos(clientes)

    print(f"sesiones.parquet  {sesiones.shape[0]} filas, {sesiones.shape[1]} columnas")
    print("perfiles:", verdad["perfil"].value_counts().to_dict())
    print(f"verdad_gasto.parquet  {len(g)} clientes, gasto medio si se quedaran {g.mean():.2f} €")
    print("las dos verdades solo sirven para medir, nunca como columna de un modelo")


if __name__ == "__main__":
    main()
