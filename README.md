# SpermDetection

Sistema base para preparar dataset y ejecutar detección + tracking de espermatozoides con **YOLOv8s + DeepSort**.

## Entregables

- `1_extract_frames.py`: extracción de frames desde videos AVI
- `2_annotation_tool.py`: herramienta interactiva de anotación (OpenCV)
- `3_convert_to_yolo.py`: conversión de anotaciones a formato YOLO
- `4_yolo_deepsort.py`: pipeline de detección y tracking en tiempo real
- `requirements.txt`: dependencias del proyecto
- `.gitignore`: exclusiones de archivos pesados/temporales

## Requisitos

- Python 3.10+
- GPU NVIDIA con CUDA (opcional, recomendado)
- Videos AVI de entrada (30 fps, 2720x1536)

## Instalación

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Estructura sugerida

```text
SpermDetection/
├── datasets/
│   ├── raw_videos/      # AVI originales
│   ├── frames/          # Frames extraídos
│   ├── annotations/     # JSON de anotación
│   └── yolo_data/       # Dataset YOLO final
├── outputs/             # Videos con tracking
└── models/              # Pesos entrenados
```

## 1) Extracción de frames

```bash
python 1_extract_frames.py \
  --input-dir datasets/raw_videos \
  --output-dir datasets/frames \
  --frame-step 5
```

Notas:
- `--frame-step 5` en un video de 30 fps guarda exactamente 6 fps para anotación.
- Se crea `datasets/frames/extraction_metadata.csv` con estadísticas.

## 2) Anotación interactiva

```bash
python 2_annotation_tool.py \
  --images-dir datasets/frames \
  --annotations-dir datasets/annotations
```

Controles:
- **Mouse izquierdo (drag):** crear bounding box
- **Mouse derecho:** deshacer última caja
- **N / Space:** siguiente imagen (autosave)
- **P:** imagen anterior (autosave)
- **S:** guardar
- **U:** deshacer
- **D:** borrar anotación del frame actual
- **Q / Esc:** salir (guarda estado actual)

Formato de salida por imagen (`.json`):

```json
{
  "image": "video_001/frame_000010.jpg",
  "width": 2720,
  "height": 1536,
  "bboxes": [
    {"x1": 100, "y1": 200, "x2": 140, "y2": 260, "class_id": 0}
  ]
}
```

## 3) Conversión a YOLO

```bash
python 3_convert_to_yolo.py \
  --images-dir datasets/frames \
  --annotations-dir datasets/annotations \
  --output-dir datasets/yolo_data \
  --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1
```

Salida esperada:

```text
datasets/yolo_data/
├── dataset.yaml
├── images/{train,val,test}/...
└── labels/{train,val,test}/...
```

## 4) Pipeline YOLOv8 + DeepSort

```bash
python 4_yolo_deepsort.py \
  --model yolov8s.pt \
  --source datasets/raw_videos/video_001.avi \
  --output outputs/video_001_tracked.mp4 \
  --conf 0.25 --iou 0.45 --class-id 0 --device 0
```

Para webcam:

```bash
python 4_yolo_deepsort.py --source 0 --model yolov8s.pt
```

## Entrenamiento YOLOv8s (ejemplo)

```bash
yolo task=detect mode=train model=yolov8s.pt data=datasets/yolo_data/dataset.yaml epochs=100 imgsz=1536
```

## Recomendaciones para 300 videos

1. Extraer subset inicial (`frame-step` alto) para arrancar rápido.
2. Anotar manualmente un conjunto balanceado por condiciones de captura.
3. Entrenar baseline con YOLOv8s.
4. Iterar con hard-negative mining y más anotaciones.
