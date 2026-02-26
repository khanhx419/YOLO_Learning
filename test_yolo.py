from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolov8n.pt')
    model.train(
        data='vietnam-1/data.yaml', 
        epochs=50, 
        imgsz=640, 
        device=0,
        workers=0,      # BẮT BUỘC: Không tạo thêm tiến trình con để đỡ tốn RAM
        batch=4,        # GIẢM XUỐNG 4: Để mỗi lần máy chỉ xử lý 4 cái ảnh, cực nhẹ
        cache=False     # KHÔNG lưu ảnh vào RAM: Để dành RAM cho Windows chạy
    )