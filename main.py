import cv2
from ultralytics import YOLO
import os

if __name__ == '__main__':
    # 1. Tải mô hình đã huấn luyện của bạn
    model = YOLO(os.path.join('runs', 'detect', 'train5', 'weights', 'best.pt'))

    # 2. Mở video đầu vào
    video_path = 'traffic.mp4'  # Thay bằng tên file video của bạn
    cap = cv2.VideoCapture(video_path)

    # Lấy thông số video để lưu file nếu cần
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv2.VideoWriter('output_counting.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # 3. Chạy nhận diện trên từng khung hình (Dùng RTX 4050)
        results = model.predict(frame, conf=0.3, device=0, verbose=False)
        
        # Lấy danh sách các lớp đã nhận diện được
        clss = results[0].boxes.cls.cpu().tolist()
        names = model.names
        
        # Khởi tạo bộ đếm
        counts = {}
        for cls_idx in clss:
            name = names[int(cls_idx)]
            counts[name] = counts.get(name, 0) + 1

        # 4. Vẽ khung hình và viết chữ đếm số lượng
        annotated_frame = results[0].plot()
        
        # Gộp 2 loại xe máy (Motorcycle và motorcycle) nếu bạn muốn
        total_moto = counts.get('Motorcycle', 0) + counts.get('motorcycle', 0)
        
        y_pos = 40
        cv2.putText(annotated_frame, f"Tong Xe May: {total_moto}", (20, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        y_pos += 40
        cv2.putText(annotated_frame, f"O to: {counts.get('car', 0)}", (20, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        y_pos += 40
        cv2.putText(annotated_frame, f"Xe tai: {counts.get('truck', 0)}", (20, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # 5. Hiển thị và lưu
        cv2.imshow("YOLOv8 Counting - RTX 4050", annotated_frame)
        out.write(annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Video đã được xử lý và lưu thành 'output_counting.mp4'")