# Machine Learning con Python: prácticas

Repositorio de trabajo del curso de Sinensia. Contiene el entorno de Python de todo el curso y
los ficheros que necesitamos para hacer los ejercicios. Las slides y el resto del material están
en el campus.

## Puesta en marcha (una sola vez)

Necesitamos `git` y [`uv`](https://docs.astral.sh/uv/). En las máquinas del curso ya vienen
instalados.

```bash
git clone <URL-DEL-REPO> ml-python-alumnos
cd ml-python-alumnos
uv sync
```

`uv sync` crea la carpeta `.venv/` con Python 3.13 y todas las librerías del curso, en las
versiones exactas que fija `uv.lock`. Es el único entorno que vamos a usar durante los siete
días, de modo que no hace falta instalar nada más ni crear entornos por laboratorio.

Configuramos también `git pull` para que guarde y reaplique nuestros cambios locales, y git para
que ignore lo que cambia al ejecutar un notebook (salidas, contadores y datos del kernel):

```bash
git config pull.rebase true
git config rebase.autoStash true
uv run nbstripout --install --attributes .gitattributes
git config filter.nbstripout.extrakeys "metadata.kernelspec metadata.language_info"
```

## Cómo trabajar

Todo se ejecuta **desde la raíz del repositorio**. Los scripts crean ahí las carpetas `datos/`,
`modelos/` y `resultados/`, y los notebooks las buscan en ese mismo sitio.

### Opción A: JupyterLab

```bash
uv run jupyter lab
```

Se abre el navegador con el repositorio a la izquierda. Los notebooks usan el kernel
`Python 3 (ipykernel)`, que es el del entorno del curso.

### Opción B: VS Code

1. `File > Open Folder…` y elegimos la carpeta `ml-python-alumnos`.
2. Si VS Code lo propone, instalamos las extensiones recomendadas (Python y Jupyter).
3. Al abrir un notebook, en `Select Kernel` elegimos `.venv (Python 3.13)`.

La configuración de `.vscode/` ya apunta al entorno `.venv` y hace que los notebooks se ejecuten
desde la raíz, igual que en JupyterLab.

### Scripts

```bash
uv run python datos_dia3.py
```

`uv run` usa el entorno del curso sin necesidad de activarlo.

## Qué hay en el repositorio

| Fichero | Para qué sirve |
|---|---|
| `ml-hello-world.ipynb` | Primer notebook de prueba del entorno |
| `notebook_sospechoso.ipynb` | Reto A del Día 1 |
| `pipeline_roto.py` | Reto B del Día 1 |
| `datos_dia3.py` … `datos_dia7.py` | Generan los datos de cada día en `datos/` |
| `utilidades.py` | Funciones propias del proyecto integrador |
| `puntuar_cartera.py` | Script final del proyecto integrador |

## Recibir correcciones y ejercicios nuevos

```bash
git pull
```

Para que la actualización nunca choque con nuestro trabajo:

- Ejecutar un notebook del repositorio no cuenta como cambio: git descarta las salidas al
  compararlo. VS Code o `git status` pueden marcarlo igualmente con una `M`, pero si al abrir
  el diff no aparece ninguna diferencia, es que no hemos cambiado nada.
- Los notebooks y scripts que creemos nosotros (ficheros nuevos) no molestan nunca.
- Si queremos modificar un fichero del repositorio, mejor trabajar sobre una copia, por ejemplo
  `pipeline_roto_mio.py`.
- La carpeta `mi-trabajo/` está ignorada por git: lo que guardemos ahí es solo nuestro.

Si aun así `git pull` se queja de cambios locales, guardamos una copia de lo que queramos
conservar y dejamos el repositorio como estaba con `git restore .` antes de volver a intentarlo.

Si tras una actualización cambia `uv.lock` (porque se ha añadido una librería), basta con
volver a lanzar `uv sync`.
