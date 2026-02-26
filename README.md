# YOLOv8 Vehicle Counting & Detection (Vietnam Traffic)

This repository contains a YOLOv8-based object detection and vehicle counting system, specifically trained on a Vietnamese traffic dataset. It is highly optimized for systems with limited system memory (8GB RAM) by utilizing an NVIDIA RTX GPU for heavy computations.

## 🚀 Key Features

- **Real-time Vehicle Counting:** Detects and counts Cars, Motorcycles, and Trucks from video inputs using OpenCV and YOLOv8.
- **Robust Input Validation:** Includes a `check_video_safe` layer to prevent system crashes from corrupted files, unsupported formats, or oversized videos (max 50MB).
- **Memory-Optimized Training Pipeline:** The training script is explicitly configured to prevent RAM overflow on 8GB systems by disabling multiprocessing (`workers=0`) and reducing batch sizes.
- **High Accuracy:** Achieved **mAP50 of 91.1%** after 50 epochs of training.

## 📂 Repository Structure

- `main.py`: The main inference script. It includes the input validation shield, runs the YOLOv8 model on a video, counts vehicles, and outputs an annotated video.
- `test_yolo.py`: The training script optimized for low-RAM Windows environments.
- `.gitignore`: Keeps the repository lightweight by excluding heavy `.pt` model weights, dataset images, and video files.

## ⚙️ Hardware & Environment

This project was developed and tested on:

- **CPU:** Intel Core i5
- **RAM:** 8GB (System Memory)
- **GPU:** NVIDIA GeForce RTX 4050 (6GB VRAM)
- **Frameworks:** PyTorch (CUDA 12.1), Ultralytics YOLOv8, OpenCV.

## 🛠️ How to Run

1.  **Clone the repository:**

    ```bash
    git clone [https://github.com/khanhx419/YOLO_Learning.git](https://github.com/khanhx419/YOLO_Learning.git)
    cd YOLO_Learning
    ```

2.  **Add your files:**
    - Place your trained model weight (e.g., `best.pt`) in the appropriate directory (e.g., `runs/detect/train5/weights/`).
    - Place a test video named `traffic.mp4` in the root directory.

3.  **Run the inference script:**
    ```bash
    python main.py
    ```
    _If the video is safe and valid, the script will output an annotated video named `output_counting.mp4`._

## 👨‍💻 Author

**Lê Tuấn Khanh**

- GitHub: [@khanhx419](https://github.com/khanhx419)
