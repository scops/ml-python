"""
Generador de los datos del Día 3 del curso de Machine Learning con Python.

Crea, dentro de la carpeta ./datos, los ficheros con los que trabajamos durante la jornada:

    clientes.csv     50 500 filas. Ficha de clientes de una tienda online, con huecos,
                     valores centinela, provincias mal escritas y algunos duplicados.
    pedidos.csv      1 000 000 filas. Un pedido por fila, con importes, unidades,
                     categoría, método de pago y valoración opcional.
    pedidos.parquet  Los mismos pedidos en formato columnar, para comparar la carga.
    clientes.xlsx    La misma ficha de clientes en Excel, para medir lo que cuesta leerla.
    crm.xlsx         Extracto de 5 000 clientes tal como lo exportaría un CRM, con el
                     identificador guardado como texto con ceros a la izquierda.

Los datos son sintéticos y reproducibles: con la misma semilla salen siempre iguales, de
modo que las cifras del deck y las de la sala coinciden fila a fila. Las imperfecciones
están puestas a propósito y en proporciones conocidas, porque el día entero consiste en
encontrarlas.

Uso:
    python datos_dia3.py                 # genera todo con la semilla 3
    python datos_dia3.py --pequeno       # versión de 100 000 pedidos para portátiles justos

Requisitos: numpy, pandas, pyarrow (para Parquet) y openpyxl (para Excel).
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PROVINCIAS = [
    "Madrid", "Barcelona", "Valencia", "Sevilla", "Málaga", "Alicante", "Murcia",
    "Bizkaia", "Zaragoza", "Asturias", "Baleares", "Las Palmas", "A Coruña",
    "Santa Cruz de Tenerife", "Cádiz", "Pontevedra", "Granada", "Tarragona",
    "Córdoba", "Girona", "Navarra", "Valladolid", "Almería", "Toledo", "Badajoz",
]
PESOS_PROV = np.array([30, 24, 11, 8, 7, 7, 6, 5, 4.5, 4, 4, 4, 4, 3.5, 3.5, 3,
                       3, 3, 2.5, 2.5, 2.5, 2.2, 2, 2, 2], dtype=float)
CATEGORIAS = ["electronica", "hogar", "moda", "deporte", "libros", "juguetes",
              "alimentacion", "belleza"]
PRECIO_MEDIO = {"electronica": 180, "hogar": 65, "moda": 45, "deporte": 55,
                "libros": 18, "juguetes": 30, "alimentacion": 25, "belleza": 28}
PAGOS = ["tarjeta", "paypal", "bizum", "transferencia", "contrareembolso"]


def generar_clientes(rng, n=50_000):
    ids = np.arange(1, n + 1)
    prov = rng.choice(PROVINCIAS, size=n, p=PESOS_PROV / PESOS_PROV.sum())
    # un 3 % de provincias escritas a mano, con espacios o en minúsculas
    sucias = rng.random(n) < 0.03
    estilos = rng.integers(0, 3, n)
    prov = prov.astype(object)
    for i in np.flatnonzero(sucias):
        p = prov[i]
        prov[i] = [p.lower(), p + " ", p.upper()][estilos[i]]

    edad = rng.normal(41, 12, n).clip(18, 90).round().astype(float)
    edad[rng.random(n) < 0.06] = np.nan          # 6 % sin edad
    edad[rng.random(n) < 0.01] = -1              # 1 % con el centinela -1

    alta = pd.Timestamp("2019-01-01") + pd.to_timedelta(rng.integers(0, 2400, n), unit="D")
    segmento = rng.choice(["A", "B", "C"], size=n, p=[0.2, 0.5, 0.3])

    # los ingresos dependen del segmento y algo de la edad
    efecto_seg = np.select([segmento == "A", segmento == "C"], [0.45, -0.30], 0.0)
    edad_ref = np.nan_to_num(edad, nan=41).clip(18, 90)
    ingresos = np.exp(rng.normal(10.15 + efecto_seg + 0.008 * (edad_ref - 41), 0.38)).round(-2)
    # los huecos NO son al azar: el segmento C deja el campo vacío mucho más a menudo
    p_hueco = np.select([segmento == "A", segmento == "C"], [0.04, 0.16], 0.08)
    ingresos[rng.random(n) < p_hueco] = np.nan
    canal = rng.choice(["web", "app", "tienda"], size=n, p=[0.55, 0.35, 0.10])
    canal = canal.astype(object)
    canal[rng.random(n) < 0.04] = None           # 4 % sin canal

    df = pd.DataFrame({
        "id_cliente": ids,
        "fecha_alta": alta.strftime("%Y-%m-%d"),
        "provincia": prov,
        "edad": edad,
        "ingresos": ingresos,
        "segmento": segmento,
        "canal": canal,
    })
    # 500 filas duplicadas, como las que deja una migración de CRM mal cerrada
    dup = df.sample(500, random_state=3)
    df = pd.concat([df, dup], ignore_index=True).sample(frac=1, random_state=3)
    return df.reset_index(drop=True)


def generar_pedidos(rng, n_clientes=50_000, n=1_000_000):
    # unos pocos clientes compran mucho más que el resto: actividad con cola larga
    actividad = rng.lognormal(0, 1.0, n_clientes)
    actividad /= actividad.sum()
    id_cliente = rng.choice(np.arange(1, n_clientes + 1), size=n, p=actividad)
    # un 0,5 % de pedidos apuntan a clientes que no están en la ficha
    huerfanos = rng.random(n) < 0.005
    id_cliente[huerfanos] = rng.integers(n_clientes + 1, n_clientes + 3000, huerfanos.sum())

    categoria = rng.choice(CATEGORIAS, size=n, p=[.14, .16, .2, .1, .12, .08, .12, .08])
    base = np.array([PRECIO_MEDIO[c] for c in CATEGORIAS])
    idx_cat = pd.Categorical(categoria, categories=CATEGORIAS).codes
    unidades = rng.geometric(0.55, n)
    precio = base[idx_cat] * np.exp(rng.normal(0, 0.5, n))
    importe = (precio * unidades).round(2)
    devol = rng.random(n) < 0.03                 # 3 % devoluciones, importe negativo
    importe[devol] *= -1

    fecha = pd.Timestamp("2024-01-01") + pd.to_timedelta(rng.integers(0, 730 * 24 * 60, n), unit="min")
    pago = rng.choice(PAGOS, size=n, p=[.48, .22, .18, .07, .05])
    valoracion = rng.choice([1, 2, 3, 4, 5], size=n, p=[.05, .07, .15, .33, .40]).astype(float)
    valoracion[rng.random(n) < 0.62] = np.nan    # la mayoría no valora

    return pd.DataFrame({
        "id_pedido": np.arange(1, n + 1),
        "id_cliente": id_cliente,
        "fecha": fecha.strftime("%Y-%m-%d %H:%M"),
        "categoria": categoria,
        "unidades": unidades,
        "importe": importe,
        "metodo_pago": pago,
        "valoracion": valoracion,
    })


def generar_crm(clientes, rng, n=5_000):
    unicos = clientes.drop_duplicates("id_cliente")
    muestra = unicos.sample(n, random_state=7)
    return pd.DataFrame({
        "id_cliente": muestra["id_cliente"].map(lambda x: f"{x:07d}"),   # texto con ceros
        "acepta_publicidad": rng.choice(["Sí", "No"], size=n, p=[0.37, 0.63]),
        "puntos_fidelidad": rng.integers(0, 5000, n),
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pequeno", action="store_true", help="100 000 pedidos en lugar de 1 000 000")
    ap.add_argument("--semilla", type=int, default=3)
    args = ap.parse_args()

    rng = np.random.default_rng(args.semilla)
    destino = Path("datos")
    destino.mkdir(exist_ok=True)

    clientes = generar_clientes(rng)
    pedidos = generar_pedidos(rng, n=100_000 if args.pequeno else 1_000_000)
    crm = generar_crm(clientes, rng)

    clientes.to_csv(destino / "clientes.csv", index=False)
    pedidos.to_csv(destino / "pedidos.csv", index=False)
    pedidos.to_parquet(destino / "pedidos.parquet", index=False)
    crm.to_excel(destino / "crm.xlsx", index=False)
    clientes.to_excel(destino / "clientes.xlsx", index=False)   # para comparar con el CSV

    for f in sorted(destino.iterdir()):
        print(f"{f.name:18s} {f.stat().st_size / 1e6:8.1f} MB")


if __name__ == "__main__":
    main()
