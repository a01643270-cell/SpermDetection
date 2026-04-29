from dataclasses import dataclass
from pathlib import Path
from ultralytics import YOLO


@dataclass
class TrainConfig:
    data_yaml: str = "datasets/yolo_data/dataset.yaml"
    model_name: str = "yolo26n.pt"
    epochs: int = 50
    imgsz: int = 640
    batch: int = 16
    device: int = 0          # GPU 0
    workers: int = 0         # importante en Windows
    project: str = "runs/sperm_yolo"
    name: str = "exp1"


def train_yolo(cfg: TrainConfig):
    data_yaml = Path(cfg.data_yaml)
    if not data_yaml.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {data_yaml}")

    print(f"Loading model: {cfg.model_name}")
    model = YOLO(cfg.model_name)

    print(f"Training with dataset: {data_yaml}")
    results = model.train(
        data=str(data_yaml),
        epochs=cfg.epochs,
        imgsz=cfg.imgsz,
        batch=cfg.batch,
        device=cfg.device,
        workers=cfg.workers,
        project=cfg.project,
        name=cfg.name,
        patience=20,
    )

    print("Training complete.")
    return model, results


if __name__ == "__main__":
    cfg = TrainConfig()
    train_yolo(cfg)