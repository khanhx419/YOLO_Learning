import cv2
from ultralytics import YOLO
import os
import sys

# ==========================================
# LỚP BẢO VỆ (INPUT VALIDATOR)
# ==========================================
def check_video_safe(file_path, max_size_mb=50):
    print(f"[*] Đang kiểm tra an toàn file đầu vào: {file_path}")
    
    # 1. Kiểm tra tồn tại
    if not os.path.exists(file_path):
        return False, "❌ Lỗi: Không tìm thấy file trong thư mục!"

    # 2. Kiểm tra định dạng (Chống file lạ/virus)
    valid_extensions = ['.mp4', '.avi', '.mov']
    _, ext = os.path.splitext(file_path)
    if ext.lower() not in valid_extensions:
        return False, f"❌ Lỗi: Đuôi file {ext} không hợp lệ. Chỉ nhận .mp4, .avi"

    # 3. Kiểm tra dung lượng (Chống tràn RAM)
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    if file_size_mb > max_size_mb:
        return False, f"❌ Lỗi: Dung lượng quá lớn ({file_size_mb:.1f}MB). Giới hạn là {max_size_mb}MB."

    # 4. Kiểm tra file hỏng bằng OpenCV
    try:
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            return False, "❌ Lỗi: File video bị hỏng hoặc sai chuẩn nén."
        
        success, frame = cap.read()
        cap.release()
        
        if not success:
            return False, "❌ Lỗi: Không thể đọc dữ liệu hình ảnh bên trong file."
            
    except Exception as e:
        return False, f"❌ Lỗi hệ thống khi quét file: {str(e)}"

    return True, f"✅ File an toàn và hợp lệ (Dung lượng: {file_size_mb:.1f}MB)."


# ==========================================
# CHƯƠNG TRÌNH CHÍNH
# ==========================================
if __name__ == '__main__':
    video_path = 'traffic.mp4'  # Tên video của bạn
    
    # BƯỚC 1: KIỂM TRA ĐẦU VÀO TRƯỚC
    is_safe, message = check_video_safe(video_path)
    print(message)
    
    # Nếu file có lỗi, dừng ngay lập tức để bảo vệ máy
    if not is_safe:
        print("-> Dừng chương trình.")
        sys.exit()

    # BƯỚC 2: TIẾN HÀNH AI NẾU FILE AN TOÀN
    print("-> Tiến hành khởi động AI YOLOv8...")
    model = YOLO(os.path.join('runs', 'detect', 'train5', 'weights', 'best.pt'))
    
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv2.VideoWriter('output_counting.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        results = model.predict(frame, conf=0.3, device=0, verbose=False)
        
        clss = results[0].boxes.cls.cpu().tolist()
        names = model.names
        
        counts = {}
        for cls_idx in clss:
            name = names[int(cls_idx)]
            counts[name] = counts.get(name, 0) + 1

        annotated_frame = results[0].plot()
        
        total_moto = counts.get('Motorcycle', 0) + counts.get('motorcycle', 0)
        
        y_pos = 40
        cv2.putText(annotated_frame, f"Tong Xe May: {total_moto}", (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        y_pos += 40
        cv2.putText(annotated_frame, f"O to: {counts.get('car', 0)}", (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        y_pos += 40
        cv2.putText(annotated_frame, f"Xe tai: {counts.get('truck', 0)}", (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow("YOLOv8 Counting - RTX 4050", annotated_frame)
        out.write(annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Video đã được xử lý và lưu thành 'output_counting.mp4'")