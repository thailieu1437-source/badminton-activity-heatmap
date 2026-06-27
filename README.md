\# Badminton Activity Heatmap



\## 1. Giới thiệu



Đây là project xử lí ảnh tạo heatmap vùng hoạt động của vận động viên cầu lông từ video. Hệ thống phát hiện vận động viên trong từng frame, ước lượng vị trí bàn chân từ bounding box, chuyển tọa độ ảnh sang tọa độ mặt sân bằng Homography, sau đó tạo heatmap biểu diễn mật độ hoạt động của vận động viên.



Project không sử dụng thư viện phát hiện tư thế có sẵn như Mediapipe. Vị trí bàn chân được ước lượng bằng điểm đáy giữa của bounding box người chơi.



\## 2. Chức năng chính



\* Phát hiện vận động viên cầu lông trong video bằng YOLO.

\* Ước lượng vị trí bàn chân từ điểm đáy giữa bounding box.

\* Chuyển đổi tọa độ ảnh sang tọa độ mặt sân bằng Homography.

\* Phân chia người chơi thành Near Player và Far Player.

\* Lưu tọa độ sau xử lí vào file CSV.

\* Tạo heatmap cho toàn bộ người chơi, người chơi phía gần và người chơi phía xa.

\* Xuất video demo có chú thích kết quả phát hiện.



\## 3. Cấu trúc thư mục



```text

.

├── src/

│   └── detect\_and\_heatmap.py

├── models/

├── output/

│   ├── annotated\_video.mp4

│   ├── positions.csv

│   ├── heatmap\_all.png

│   ├── heatmap\_far.png

│   └── heatmap\_near.png

├── video/

├── extract\_frames.py

├── yolov8n.pt

├── requirements.txt

└── README.md

```



\## 4. Công cụ và thư viện



\* Python

\* OpenCV

\* Ultralytics YOLO

\* NumPy

\* Pandas

\* Matplotlib

\* SciPy



\## 5. Phương pháp thực hiện



Quy trình xử lí chính của project gồm các bước:



1\. Đọc video đầu vào.

2\. Tách video thành từng frame.

3\. Phát hiện vận động viên trong mỗi frame bằng YOLO.

4\. Lấy điểm đáy giữa của bounding box để ước lượng vị trí bàn chân.

5\. Chuyển tọa độ ảnh sang tọa độ mặt sân bằng Homography.

6\. Phân loại vị trí thành Near Player hoặc Far Player dựa trên tọa độ theo chiều dài sân.

7\. Lưu tọa độ vào file `positions.csv`.

8\. Tạo heatmap vùng hoạt động.

9\. Xuất video chú thích và ảnh kết quả.



\## 6. Cách chạy chương trình



Cài đặt các thư viện cần thiết:



```bash

pip install -r requirements.txt

```



Chạy chương trình:



```bash

python src/detect\_and\_heatmap.py

```



\## 7. Kết quả đầu ra



Sau khi chạy chương trình, các kết quả được lưu trong thư mục `output/`:



\* `annotated\_video.mp4`: video có chú thích phát hiện người chơi

\* `positions.csv`: tọa độ người chơi sau khi chuyển sang tọa độ sân

\* `heatmap\_all.png`: heatmap toàn bộ người chơi

\* `heatmap\_far.png`: heatmap người chơi phía xa

\* `heatmap\_near.png`: heatmap người chơi phía gần



\## 8. Kết quả minh họa



\### All Players Heatmap



!\[All Players Heatmap](output/heatmap\_all.png)



\### Far Player Heatmap



!\[Far Player Heatmap](output/heatmap\_far.png)



\### Near Player Heatmap



!\[Near Player Heatmap](output/heatmap\_near.png)



\## 9. Demo video



Video demo có chú thích được lưu tại:



```text

output/demo_video.mp4

```






