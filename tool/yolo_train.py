# -----------------------------------------------------------
# YOLOv5 Training Script using Ultralytics
# written by Jeongmin Kim (jm.kim@dankook.ac.kr)
# -----------------------------------------------------------
import argparse
from ultralytics import YOLO
from config import cfg, update_config  # Config 로드 함수
import models  # YOLO 모델 생성 함수 포함

def main():
    # ArgumentParser로 학습 설정 받기
    parser = argparse.ArgumentParser(description='Train YOLOv5 model for X-ray foot detection')
    parser.add_argument('--cfg', type=str, required=True, help='Path to config file')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    args = parser.parse_args()

    # Config 파일 불러오기
    update_config(cfg, args)  # Config 업데이트
    cfg.freeze()

    # 모델 생성 (Ultralytics YOLO)
    model = models.create('YOLOV5', model_name='yolov5s', pretrained=True, num_classes=1)

    # 1. 학습
    print("Starting training...")
    model.train(
        data=cfg.DATA.DATASET,  # 데이터셋 경로 (data.yaml)
        epochs=args.epochs,    # 학습 에포크
        imgsz=cfg.DATA.IMG_SIZE,  # 입력 이미지 크기
        batch=args.batch_size, # 배치 크기
        device=0               # GPU 설정
    )

    # 2. 검증
    print("Starting validation...")
    metrics = model.val(
        data=cfg.DATA.DATASET,  # 검증 데이터셋
        batch=args.batch_size, # 배치 크기
        device=0               # GPU 설정
    )
    print(f"Validation Metrics: {metrics}")

    # 3. 추론 (선택적)
    print("Running inference...")
    results = model.predict(
        source='path/to/val/images',  # 검증 이미지 경로
        save=True                    # 결과 저장
    )
    results.show()


if __name__ == '__main__':
    main()
