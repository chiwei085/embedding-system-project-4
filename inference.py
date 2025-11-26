from dataclasses import dataclass
from pathlib import Path
from typing import Final

from ultralytics import YOLO

INFO_PREFIX: Final[str] = "[Info]"
ERROR_PREFIX: Final[str] = "[Error]"
SUCCESS_PREFIX: Final[str] = "[Success]"


@dataclass(frozen=True)
class InferenceConfig:
    weights: Path = Path("jetbot_project/run_vehicle_3classes/weights/best.pt")
    source: Path = Path("test2.mp4")
    project: str = "jetbot_project"
    run_name: str = "video_result"
    conf: float = 0.2
    device: int | str | None = 0


def run_inference(cfg: InferenceConfig = InferenceConfig()) -> None:
    if not cfg.weights.is_file():
        print(f"{ERROR_PREFIX} 找不到權重檔案: {cfg.weights}")
        return

    if isinstance(cfg.source, Path) and not cfg.source.is_file():
        print(f"{ERROR_PREFIX} 找不到輸入檔案: {cfg.source}")
        return

    print(f"{INFO_PREFIX} 載入模型權重: {cfg.weights}")
    model = YOLO(str(cfg.weights))

    print(f"{INFO_PREFIX} 開始進行推論...")
    model.predict(
        source=str(cfg.source),
        save=True,
        conf=cfg.conf,
        project=cfg.project,
        name=cfg.run_name,
        device=cfg.device,
    )

    print(
        f"{SUCCESS_PREFIX} 處理完成！請到 "
        f"{cfg.project}/{cfg.run_name}/ 資料夾查看輸出結果。"
    )


if __name__ == "__main__":
    run_inference()
