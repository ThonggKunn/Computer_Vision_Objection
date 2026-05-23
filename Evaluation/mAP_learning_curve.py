import pandas as pd
import matplotlib.pyplot as plt

# đường dẫn tới file results.csv
csv_path = "../Models/runs/detect/yolov8s_best/results.csv"

# đọc dữ liệu
data = pd.read_csv(csv_path)

# lấy epoch
epochs = data['epoch']

# lấy mAP
map50 = data['metrics/mAP50(B)']
map5095 = data['metrics/mAP50-95(B)']

# vẽ biểu đồ
plt.figure()

plt.plot(epochs, map50, label="mAP50")
plt.plot(epochs, map5095, label="mAP50-95")

plt.xlabel("Epoch")
plt.ylabel("Score")
plt.title("Learning Curve (mAP)")
plt.legend()

plt.grid()

plt.show()