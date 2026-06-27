import os
import json
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO

# =========================
# CẤU HÌNH
# =========================
MODEL_PATH = "models/player_best.pt"
VIDEO_PATH = "video/input_10min.mp4"
COURT_POINTS_PATH = "output/court_points.json"
OUTPUT_DIR = "output"

OUTPUT_CSV = os.path.join(OUTPUT_DIR, "positions.csv")
OUTPUT_VIDEO = os.path.join(OUTPUT_DIR, "annotated_video.mp4")
OUTPUT_HEATMAP_ALL = os.path.join(OUTPUT_DIR, "heatmap_all.png")
OUTPUT_HEATMAP_NEAR = os.path.join(OUTPUT_DIR, "heatmap_near.png")
OUTPUT_HEATMAP_FAR = os.path.join(OUTPUT_DIR, "heatmap_far.png")

CONF_THRESHOLD = 0.25
FRAME_STEP = 5  # xử lý mỗi 2 frame để nhanh hơn; nếu muốn chính xác hơn thì đổi thành 1

# Kích thước sân cầu lông đôi (mét)
COURT_WIDTH = 6.10
COURT_LENGTH = 13.40

# =========================
# HÀM VẼ SÂN TỪ TRÊN XUỐNG
# =========================
def draw_badminton_court(ax):
    # Khung ngoài sân đôi
    outer = np.array([
        [0, 0],
        [COURT_WIDTH, 0],
        [COURT_WIDTH, COURT_LENGTH],
        [0, COURT_LENGTH],
        [0, 0]
    ])
    ax.plot(outer[:, 0], outer[:, 1], color="white", linewidth=2)

    # Biên sân đơn
    single_margin = (COURT_WIDTH - 5.18) / 2  # 0.46 m
    ax.plot([single_margin, single_margin], [0, COURT_LENGTH], color="white", linewidth=1)
    ax.plot([COURT_WIDTH - single_margin, COURT_WIDTH - single_margin], [0, COURT_LENGTH], color="white", linewidth=1)

    # Lưới ở giữa sân
    net_y = COURT_LENGTH / 2
    ax.plot([0, COURT_WIDTH], [net_y, net_y], color="white", linewidth=1.5)

    # Vạch giao cầu ngắn
    short_service = 1.98
    ax.plot([0, COURT_WIDTH], [net_y - short_service, net_y - short_service], color="white", linewidth=1)
    ax.plot([0, COURT_WIDTH], [net_y + short_service, net_y + short_service], color="white", linewidth=1)

    # Vạch giao cầu dài cho đánh đôi
    long_service_margin = 0.76
    ax.plot([0, COURT_WIDTH], [long_service_margin, long_service_margin], color="white", linewidth=1)
    ax.plot([0, COURT_WIDTH], [COURT_LENGTH - long_service_margin, COURT_LENGTH - long_service_margin], color="white", linewidth=1)

    # Đường giữa
    center_x = COURT_WIDTH / 2
    ax.plot([center_x, center_x], [0, COURT_LENGTH], color="white", linewidth=1)

    ax.set_xlim(0, COURT_WIDTH)
    ax.set_ylim(COURT_LENGTH, 0)  # đảo trục y để phía xa nằm trên
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("#2e8b57")


# =========================
# HÀM TẠO HEATMAP
# =========================
def save_heatmap(points, save_path, title):
    if len(points) == 0:
        print(f"Không có điểm nào để tạo heatmap cho {title}")
        return

    xs = [p["court_x"] for p in points]
    ys = [p["court_y"] for p in points]

    heatmap, _, _ = np.histogram2d(
        xs,
        ys,
        bins=[60, 120],
        range=[[0, COURT_WIDTH], [0, COURT_LENGTH]]
    )

    heatmap = heatmap.astype(np.float32)
    heatmap = cv2.GaussianBlur(heatmap, (0, 0), sigmaX=4, sigmaY=4)

    fig, ax = plt.subplots(figsize=(6, 12))
    draw_badminton_court(ax)

    ax.imshow(
        heatmap.T,
        extent=[0, COURT_WIDTH, COURT_LENGTH, 0],
        cmap="jet",
        alpha=0.75,
        aspect="auto"
    )

    ax.set_title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Đã lưu heatmap: {save_path}")


# =========================
# ĐỌC 4 ĐIỂM SÂN & TÍNH HOMOGRAPHY
# =========================
os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(COURT_POINTS_PATH, "r") as f:
    src_points = np.array(json.load(f), dtype=np.float32)

dst_points = np.array([
    [0, 0],
    [COURT_WIDTH, 0],
    [COURT_WIDTH, COURT_LENGTH],
    [0, COURT_LENGTH]
], dtype=np.float32)

H, _ = cv2.findHomography(src_points, dst_points)

# =========================
# LOAD MODEL
# =========================
model = YOLO(MODEL_PATH)

# =========================
# ĐỌC VIDEO
# =========================
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("Không mở được video.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"FPS: {fps}")
print(f"Frame size: {width} x {height}")
print(f"Tổng frame: {total_frames}")

# Vì FRAME_STEP=2 nên video output sẽ ghi với fps giảm tương ứng
out_fps = max(1, fps / FRAME_STEP)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, out_fps, (width, height))

positions = []

frame_id = 0
processed_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_id % FRAME_STEP != 0:
        frame_id += 1
        continue

    results = model.predict(frame, conf=CONF_THRESHOLD, verbose=False)
    annotated = frame.copy()

    boxes = results[0].boxes
    if boxes is not None and len(boxes) > 0:
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()

        for i, box in enumerate(xyxy):
            x1, y1, x2, y2 = box
            conf = float(confs[i])

            foot_x = int((x1 + x2) / 2)
            foot_y = int(y2)

            # Biến đổi sang tọa độ sân thật
            img_point = np.array([[[foot_x, foot_y]]], dtype=np.float32)
            court_point = cv2.perspectiveTransform(img_point, H)[0][0]
            court_x = float(court_point[0])
            court_y = float(court_point[1])

            # Chỉ giữ điểm nằm trong sân (có chừa chút sai số)
            if -0.3 <= court_x <= COURT_WIDTH + 0.3 and -0.3 <= court_y <= COURT_LENGTH + 0.3:
                zone = "near" if court_y > COURT_LENGTH / 2 else "far"

                positions.append({
                    "frame_id": frame_id,
                    "conf": conf,
                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(x2),
                    "y2": float(y2),
                    "foot_x": foot_x,
                    "foot_y": foot_y,
                    "court_x": court_x,
                    "court_y": court_y,
                    "zone": zone
                })

                # Vẽ box
                cv2.rectangle(
                    annotated,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    (255, 0, 0),
                    2
                )

                # Vẽ điểm bàn chân
                cv2.circle(annotated, (foot_x, foot_y), 5, (0, 0, 255), -1)

                # Ghi text
                text = f"{zone} {conf:.2f}"
                cv2.putText(
                    annotated,
                    text,
                    (int(x1), max(20, int(y1) - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2
                )

    writer.write(annotated)

    processed_count += 1
    if processed_count % 100 == 0:
        print(f"Đã xử lý {processed_count} frame...")

    frame_id += 1

cap.release()
writer.release()

print(f"Đã lưu video chú thích: {OUTPUT_VIDEO}")

# =========================
# LƯU CSV
# =========================
df = pd.DataFrame(positions)
df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
print(f"Đã lưu CSV: {OUTPUT_CSV}")

# =========================
# TẠO HEATMAP
# =========================
all_points = positions
near_points = [p for p in positions if p["zone"] == "near"]
far_points = [p for p in positions if p["zone"] == "far"]

save_heatmap(all_points, OUTPUT_HEATMAP_ALL, "Badminton Activity Heatmap - All Players")
save_heatmap(near_points, OUTPUT_HEATMAP_NEAR, "Badminton Activity Heatmap - Near Player")
save_heatmap(far_points, OUTPUT_HEATMAP_FAR, "Badminton Activity Heatmap - Far Player")

print("Hoàn thành!")