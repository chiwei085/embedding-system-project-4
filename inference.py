from ultralytics import YOLO

# 1. 載入模型
model = YOLO('/home/rvl1421/embedding_system_jetbot/jetbot_project/run_vehicle_3classes/weights/best.pt')

# 2. 一行指令完成所有事
# source: 輸入影片路徑
# save=True: 自動儲存結果影片
# conf=0.2: 設定信心門檻
model.predict(
    source='/home/rvl1421/embedding_system_jetbot/test.mp4', 
    save=True, 
    conf=0.2,
    project='jetbot_project', # 儲存的主資料夾
    name='video_result'       # 儲存的子資料夾
)

print("處理完成！請去 jetbot_project/video_result/ 找影片")