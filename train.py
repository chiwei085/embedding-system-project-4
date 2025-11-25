import os
from ultralytics import YOLO

def run_training():
    # 1. 指定您的 data.yaml 路徑
    # 這是您剛剛提供的路徑位置
    yaml_path = '/home/rvl1421/embedding_system_jetbot/dataset/data.yaml'

    # 再次檢查檔案是否存在，避免路徑打錯
    if not os.path.exists(yaml_path):
        print(f"[Error] 找不到檔案: {yaml_path}")
        return

    print(f"[Info] 載入設定檔: {yaml_path}")

    # 2. 載入模型
    # 使用 yolov8n.pt (Nano) 速度最快，適合 Jetbot 或邊緣裝置測試
    # 如果您的電腦顯卡不錯 (如 RTX 3060 以上)，可以改用 'yolov8s.pt'
    model = YOLO('yolov8s.pt') 

    # 3. 開始訓練
    print("[Info] 開始訓練...")
    results = model.train(
        data=yaml_path,
        epochs=100,      # 訓練 100 輪
        imgsz=640,       # 圖片大小 640
        batch=32,        # 如果顯卡記憶體不足 (OOM)，請改成 8 或 4
        project='jetbot_project', # 專案名稱
        name='run_vehicle_3classes',   # 實驗名稱
        device=0         # 指定使用第一張 GPU (若無 GPU 會自動切回 CPU)
    )
    
    print("[Success] 訓練完成！")

if __name__ == "__main__":
    run_training()