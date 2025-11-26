from dataclasses import dataclass
from pathlib import Path
from typing import Final

from ultralytics import YOLO


@dataclass(frozen=True)
class TrainingConfig:
    data_yaml: Path = Path("data.yaml")
    weights: str = "yolov8s.pt"
    epochs: int = 100
    imgsz: int = 640
    batch: int = 16
    project: str = "jetbot_project"
    run_name: str = "run_vehicle_3classes"
    device: int | str = 0


INFO_PREFIX: Final[str] = "[Info]"
ERROR_PREFIX: Final[str] = "[Error]"
SUCCESS_PREFIX: Final[str] = "[Success]"


def run_training(cfg: TrainingConfig = TrainingConfig()) -> None:
    if not cfg.data_yaml.is_file():
        print(f"{ERROR_PREFIX} 找不到檔案: {cfg.data_yaml}")
        return

    print(f"{INFO_PREFIX} 載入設定檔: {cfg.data_yaml}")

    model = YOLO(cfg.weights)

    print(f"{INFO_PREFIX} 開始訓練...")
    model.train(
        data=str(cfg.data_yaml),
        epochs=cfg.epochs,
        imgsz=cfg.imgsz,
        batch=cfg.batch,
        project=cfg.project,
        name=cfg.run_name,
        device=cfg.device,
    )

    print(f"{SUCCESS_PREFIX} 訓練完成！")


if __name__ == "__main__":
    run_training()
