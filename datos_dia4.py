"""
Generador de los datos del Día 4 del curso de Machine Learning con Python.

El Día 4 empieza donde terminó el Día 3, con la tabla de una fila por cliente que guardamos en
datos/clientes_features.parquet. Este script la completa con dos columnas que necesitamos hoy y
la guarda como datos/clientes_dia4.parquet:

    codigo_postal   Texto de cinco cifras que empieza por el prefijo de la provincia. Hay unos
                    pocos miles de códigos distintos, muchos con muy pocos clientes. Es la columna
                    de muchas categorías con la que ponemos a prueba los codificadores.
    abandona        True si el cliente no vuelve a comprar en los seis meses siguientes al final
                    del histórico. Es el objetivo que usamos para colorear gráficos y para el
                    TargetEncoder, y el que retomaremos en el Día 6 al clasificar.

Si clientes_features.parquet no existe, el script lo reconstruye a partir de los datos del Día 3,
y si tampoco existen esos datos, los genera antes con datos_dia3.py. Así cualquiera puede
incorporarse al Día 4 sin haber terminado el guiado 3.2.

Los datos son sintéticos y reproducibles con la semilla 4. El objetivo abandona sale de un modelo
conocido, escrito más abajo en generar_abandono, de modo que sabemos de antemano qué relaciones
tienen que aparecer en los gráficos. Entre ellas hay una relación en forma de U con la edad que la
correlación lineal no detecta, puesta a propósito.

Uso:
    python datos_dia4.py

Requisitos: numpy, pandas y pyarrow. Necesita datos_dia3.py en la misma carpeta.
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATOS = Path("datos")

# prefijo postal oficial de cada provincia del dataset del Día 3
PREFIJO = {
    "Madrid": "28", "Barcelona": "08", "Valencia": "46", "Sevilla": "41", "Málaga": "29",
    "Alicante": "03", "Murcia": "30", "Bizkaia": "48", "Zaragoza": "50", "Asturias": "33",
    "Baleares": "07", "Las Palmas": "35", "A Coruña": "15", "Santa Cruz de Tenerife": "38",
    "Cádiz": "11", "Pontevedra": "36", "Granada": "18", "Tarragona": "43", "Córdoba": "14",
    "Girona": "17", "Navarra": "31", "Valladolid": "47", "Almería": "04", "Toledo": "45",
    "Badajoz": "06",
}


def reconstruir_features():
    """Rehace el guiado 3.2 completo. Mismo código que el cuaderno de soluciones del Día 3."""
    if not (DATOS / "pedidos.parquet").exists():
        import subprocess, sys
        subprocess.run([sys.executable, "datos_dia3.py"], check=True)
    from datos_dia3 import PROVINCIAS

    pedidos = pd.read_parquet(DATOS / "pedidos.parquet")
    pedidos["fecha"] = pd.to_datetime(pedidos["fecha"], format="%Y-%m-%d %H:%M")
    pedidos["categoria"] = pedidos["categoria"].astype("category")
    clientes = pd.read_csv(DATOS / "clientes.csv", na_values={"edad": [-1]})

    fichas = clientes.drop_duplicates("id_cliente").copy()
    canonica = {p.lower(): p for p in PROVINCIAS}
    fichas["provincia"] = (fichas["provincia"].str.strip().str.lower()
                           .map(canonica).astype("category"))
    seg = fichas.groupby("segmento")
    fichas["ingresos_falta"] = fichas["ingresos"].isna()
    fichas["ingresos"] = fichas["ingresos"].fillna(seg["ingresos"].transform("mean"))
    fichas["edad"] = fichas["edad"].fillna(seg["edad"].transform("median"))
    fichas["canal"] = fichas["canal"].fillna("desconocido").astype("category")
    fichas["segmento"] = fichas["segmento"].astype("category")

    referencia = pedidos["fecha"].max()
    resumen = (pedidos.assign(devuelto=pedidos["importe"] < 0)
               .groupby("id_cliente")
               .agg(ultima=("fecha", "max"), pedidos=("id_pedido", "size"),
                    gasto=("importe", "sum"), ticket_medio=("importe", "mean"),
                    devoluciones=("devuelto", "mean"), nota_media=("valoracion", "mean")))
    resumen["recencia"] = (referencia - resumen.pop("ultima")).dt.days
    cuenta = pedidos.groupby(["id_cliente", "categoria"], observed=True).size()
    resumen["favorita"] = cuenta.unstack(fill_value=0).idxmax(axis=1).astype("category")

    final = (fichas.merge(resumen, left_on="id_cliente", right_index=True,
                          how="inner", validate="one_to_one").set_index("id_cliente"))
    final["valora"] = final["nota_media"].notna()
    final["nota_media"] = final["nota_media"].fillna(final["nota_media"].median())
    final.to_parquet(DATOS / "clientes_features.parquet")
    return final


def generar_codigos_postales(rng, provincia):
    """Cada provincia tiene su bolsa de códigos, y dentro de ella unos pocos acumulan casi todo."""
    cp = np.empty(len(provincia), dtype=object)
    for prov, pref in PREFIJO.items():
        filas = np.flatnonzero(provincia == prov)
        if len(filas) == 0:
            continue
        n_codigos = min(900, max(20, len(filas) // 8))
        sufijos = rng.choice(np.arange(1, 1000), size=n_codigos, replace=False)
        pesos = 1 / np.arange(1, n_codigos + 1) ** 0.9          # reparto con cola larga
        elegidos = rng.choice(sufijos, size=len(filas), p=pesos / pesos.sum())
        cp[filas] = [f"{pref}{s:03d}" for s in elegidos]
    return cp


def generar_abandono(rng, df):
    """Modelo conocido del abandono. Lo que aquí se escribe es lo que los gráficos deben revelar."""
    edad_c = (df["edad"] - 41) / 12
    efecto_cp = {c: rng.normal(0, 0.7) for c in df["codigo_postal"].unique()}
    logit = (-2.05
             + 1.10 * np.log1p(df["recencia"]) - 1.10 * np.log1p(df["recencia"]).mean()
             - 0.45 * np.log(df["pedidos"]) + 0.45 * np.log(df["pedidos"]).mean()
             + 4.0 * df["devoluciones"]
             - 0.55 * (df["nota_media"] - 4)
             + 0.85 * edad_c ** 2                                 # forma de U, a propósito
             + 0.35 * (df["canal"] == "tienda")
             + df["codigo_postal"].map(efecto_cp).astype(float))
    p = 1 / (1 + np.exp(-logit))
    return rng.random(len(df)) < p


def main():
    rng = np.random.default_rng(4)
    ruta = DATOS / "clientes_features.parquet"
    df = pd.read_parquet(ruta) if ruta.exists() else reconstruir_features()

    df["codigo_postal"] = generar_codigos_postales(rng, df["provincia"].astype(str).to_numpy())
    df["codigo_postal"] = df["codigo_postal"].astype("str")
    df["abandona"] = generar_abandono(rng, df)

    df.to_parquet(DATOS / "clientes_dia4.parquet")
    print(f"clientes_dia4.parquet  {df.shape[0]} filas, {df.shape[1]} columnas")
    print(f"códigos postales distintos: {df['codigo_postal'].nunique()}")
    print(f"tasa de abandono: {df['abandona'].mean():.3f}")


if __name__ == "__main__":
    main()
