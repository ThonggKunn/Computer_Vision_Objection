import pandas as pd

models = [
"runs/detect/results/yolov8n3/results.csv",
"runs/detect/yolov8s_best/results.csv",
"runs/detect/results/yolov8m/results.csv"
]

for m in models:

    data = pd.read_csv(m)

    best_map = data["metrics/mAP50-95(B)"].max()

    print(m, best_map)