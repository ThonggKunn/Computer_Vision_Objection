import pandas as pd
import matplotlib.pyplot as plt

# đường dẫn 2 model
# csv_n = "runs/detect/results/yolov8n3/results.csv"
csv_s = "runs/detect/yolov8s_best/results.csv"

# đọc dữ liệu
# data_n = pd.read_csv(csv_n)
data_s = pd.read_csv(csv_s)

# vẽ loss
# plt.plot(data_n['epoch'], data_n['train/box_loss'], label="YOLOv8n train loss")
plt.plot(data_s['epoch'], data_s['train/box_loss'], label="YOLOv8s train loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Learning Curve Comparison")
plt.legend()

plt.show()