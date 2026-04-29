import os
import shutil
from pathlib import Path
from tqdm import tqdm

# ====== CONFIGURA ESTAS RUTAS ======
carpeta_origen = r"E:\Frames"
carpeta_destino = r"E:\FrameSeparado"

# Extensiones permitidas
extensiones = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# Crear carpeta destino
destino = Path(carpeta_destino)
destino.mkdir(exist_ok=True)

# Crear 4 carpetas
carpetas = []
for i in range(1, 5):
    carpeta = destino / f"grupo_{i}"
    carpeta.mkdir(exist_ok=True)
    carpetas.append(carpeta)

# ====== PASO 1: CONTAR ARCHIVOS (para barra precisa) ======
total_archivos = 0
for _, _, archivos in os.walk(carpeta_origen):
    total_archivos += len(archivos)

# ====== PASO 2: ESCANEO CON BARRA ======
imagenes = []
with tqdm(total=total_archivos, desc="Escaneando archivos") as pbar:
    for ruta, subcarpetas, archivos in os.walk(carpeta_origen):
        for archivo in archivos:
            archivo_path = Path(ruta) / archivo
            if archivo_path.suffix.lower() in extensiones:
                imagenes.append(archivo_path)
            pbar.update(1)

imagenes.sort()

print(f"Se encontraron {len(imagenes)} imágenes.")

# ====== PASO 3: COPIA CON BARRA ======
for idx, imagen in enumerate(tqdm(imagenes, desc="Copiando imágenes")):
    carpeta_actual = carpetas[idx % 4]

    # Evitar nombres repetidos
    nuevo_nombre = f"{idx+1:05d}_{imagen.name}"
    destino_final = carpeta_actual / nuevo_nombre

    shutil.copy2(imagen, destino_final)

print("Listo. Las imágenes fueron copiadas en 4 carpetas.")