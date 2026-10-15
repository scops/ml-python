"""
Generador de los datos del Día 5 del curso de Machine Learning con Python.

El Día 5 cambia de pregunta. Hasta ahora el objetivo era saber si un cliente abandona. Hoy
queremos saber cuánto gastará en los próximos seis meses si se queda, que es un número en euros
y, por tanto, un problema de regresión. Este script parte de datos/clientes_dia4.parquet, le
añade esa columna y guarda el resultado como datos/clientes_dia5.parquet:

    gasto_6m    Euros que el cliente gasta en los seis meses siguientes al final del histórico.
                Vale 0 para quien abandona, porque abandonar significa justo no volver a
                comprar en esos seis meses. Para quien se queda es siempre positivo.

Si clientes_dia4.parquet no existe, el script ejecuta antes datos_dia4.py, que a su vez
reconstruye lo que haga falta del Día 3. Así cualquiera puede incorporarse al Día 5 sin haber
terminado los días anteriores.

Los datos son sintéticos y reproducibles con la semilla 5. El gasto futuro sale de un modelo
conocido, escrito en generar_gasto_6m, de modo que sabemos de antemano qué coeficientes debería
encontrar una regresión. El modelo es multiplicativo, como suele serlo el gasto real, y por eso
se vuelve lineal al tomar logaritmos. Lleva además una curva suave con la edad, un efecto propio
de cada código postal y un ruido grande, porque el gasto de una persona nunca es del todo
predecible.

Uso:
    python datos_dia5.py

Requisitos: numpy, pandas y pyarrow. Necesita datos_dia4.py y datos_dia3.py en la misma carpeta.
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATOS = Path("datos")

# Coeficientes del modelo que genera el logaritmo del gasto a seis meses.
# Son los que una regresión sobre log(gasto_6m) debería acercarse a recuperar.
VERDAD = {
    "constante": 0.55,
    "log_pedidos": 0.75,          # elasticidad: un 10 % más de pedidos, un 7,5 % más de gasto
    "log_ticket": 0.80,
    "log1p_recencia": -0.12,
    "devoluciones": -1.50,
    "nota_media": 0.12,           # por punto de nota, centrada en 4
    "ingresos_10k": 0.10,         # por cada 10 000 € de ingresos, centrados en la media
    "edad_curva": -0.10,          # por ((edad - 52) / 12) al cuadrado, máximo a los 52 años
    "canal_app": 0.12,
    "canal_tienda": -0.08,
    "sd_codigo_postal": 0.25,
    "sd_ruido": 0.50,
}


def generar_gasto_6m(rng, df):
    """Modelo conocido del gasto futuro. Multiplicativo en euros, lineal en logaritmos."""
    v = VERDAD
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
    gasto = np.exp(log_g).round(2)
    return gasto.where(~df["abandona"], 0.0)


def main():
    rng = np.random.default_rng(5)
    ruta = DATOS / "clientes_dia4.parquet"
    if not ruta.exists():
        import subprocess, sys
        subprocess.run([sys.executable, "datos_dia4.py"], check=True)
    df = pd.read_parquet(ruta)

    df["gasto_6m"] = generar_gasto_6m(rng, df)
    df.to_parquet(DATOS / "clientes_dia5.parquet")

    quedan = df.loc[~df["abandona"], "gasto_6m"]
    print(f"clientes_dia5.parquet  {df.shape[0]} filas, {df.shape[1]} columnas")
    print(f"clientes que se quedan: {len(quedan)}")
    print(f"gasto_6m de quien se queda: mediana {quedan.median():.2f} €, "
          f"media {quedan.mean():.2f} €, máximo {quedan.max():.2f} €")


if __name__ == "__main__":
    main()
