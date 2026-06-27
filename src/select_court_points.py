import cv2
import json
import os

image_path = "frames/frame_00110.jpg"
output_path = "output/court_points.json"

points = []

def mouse_callback(event, x, y, flags, param):
    global points

    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 4:
            points.append([x, y])
            print(f"Point {len(points)}: x={x}, y={y}")

img = cv2.imread(image_path)

if img is None:
    print("Không đọc được ảnh. Kiểm tra lại image_path.")
    exit()

clone = img.copy()

cv2.namedWindow("Select 4 court corners", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Select 4 court corners", mouse_callback)

while True:
    display = clone.copy()

    for i, point in enumerate(points):
        x, y = point
        cv2.circle(display, (x, y), 8, (0, 0, 255), -1)
        cv2.putText(display, str(i + 1), (x + 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Select 4 court corners", display)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("r"):
        points = []
        print("Reset points")

    if key == ord("s"):
        if len(points) == 4:
            os.makedirs("output", exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(points, f, indent=4)
            print(f"Saved points to {output_path}")
            break
        else:
            print("Bạn cần chọn đủ 4 điểm.")

    if key == 27:
        break

cv2.destroyAllWindows()