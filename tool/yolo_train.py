# -----------------------------------------------------------
# 
# written by Jeongmin Kim (jm.kim@dankook.ac.kr)
#
# -----------------------------------------------------------
import torch
import argparse
from torch.utils.data import DataLoader
from yolov5 import YOLOv5  # YOLOv5 라이브러리 import -> yolo설치
from data import COCOAnnotationDataset  # COCOAnnotationDataset import
from utils import load_config  # config 로드 함수 # 내 횐경에 맞게 수정

def train(cfg, train_loader, model, optimizer, device):
    model.train()
    running_loss = 0.0
    for epoch in range(cfg.TRAIN.EPOCHS):
        for batch_idx, batch in enumerate(train_loader):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            bboxes = batch['bbox'].to(device)  # Bounding box 정보도 가져옵니다.

            optimizer.zero_grad()
            outputs = model(images)

            # 손실 함수 (YOLOv5 모델에 맞는 loss 함수 사용)
            loss = model.compute_loss(outputs, bboxes, labels)  # YOLOv5에 맞는 손실 함수 사용
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if batch_idx % 100 == 0:
                print(f"Epoch [{epoch}/{cfg.TRAIN.EPOCHS}], Step [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}")

        # 모델 저장
        torch.save(model.state_dict(), f"{cfg.TRAIN.MODEL_DIR}/model_epoch_{epoch+1}.pth")
        print(f"Epoch {epoch+1} completed with loss: {running_loss / len(train_loader)}")

def main():
    # ArgumentParser로 학습 설정 받기
    parser = argparse.ArgumentParser(description='Train YOLOv5 model for X-ray foot detection')
    parser.add_argument('--cfg', type=str, required=True, help='Path to config file')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    args = parser.parse_args()

    # Config 파일 불러오기
    cfg = load_config(args.cfg)  # load_config는 config 파일을 로드하는 함수입니다.

    # 모델, 손실 함수 및 옵티마이저 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = YOLOv5.load(cfg.TRAIN.MODEL_PATH).to(device)  # YOLOv5 모델 로드
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # 데이터 로더 준비
    dataset = COCOAnnotationDataset(cfg, augment=True)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # 학습 시작
    train(cfg, train_loader, model, optimizer, device)

if __name__ == '__main__':
    main()
