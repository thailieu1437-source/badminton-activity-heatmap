import cv2
import os

video_path = "video/input_10min.mp4"
output_dir = "frames"

os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Không mở được video. Kiểm tra lại tên file hoặc đường dẫn.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = total_frames / fps

print("FPS:", fps)
print("Tổng frame:", total_frames)
print("Thời lượng:", duration, "giây")

# Lấy 1 frame mỗi 1 giây
frame_interval = int(fps * 1)

frame_id = 0
saved_id = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_id % frame_interval == 0:
        filename = f"frame_{saved_id:05d}.jpg"
        save_path = os.path.join(output_dir, filename)
        cv2.imwrite(save_path, frame)
        saved_id += 1

    frame_id += 1

cap.release()

print(f"Đã lưu {saved_id} ảnh vào thư mục {output_dir}")