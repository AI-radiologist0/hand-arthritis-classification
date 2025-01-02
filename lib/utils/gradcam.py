# -----------------------------------------------------------
# 
# written by Jeongmin Kim (jm.kim@dankook.ac.kr)
#
# -----------------------------------------------------------
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from .vis import denormalize_image
from gradcam import GradCAM
from gradcam.utils import visualize_cam
import os
import math

def visualize_gradcam_batch(batch, model, target_layer, class_names, output_dir="vis_img"):
    """
    Grad-CAM을 배치 단위로 시각화하고 저장.
    Args:
        batch (dict): DataLoader에서 가져온 배치 데이터 (image, label 포함).
        model (torch.nn.Module): 학습된 모델.
        target_layer (torch.nn.Module): Grad-CAM 대상 레이어.
        class_names (list): 클래스 이름 리스트.
        output_dir (str): 시각화 결과를 저장할 디렉토리.
    """
    images = batch["image"].to("cuda")  # 배치 이미지 (batch_size, C, H, W)
    labels = batch["label"].to("cuda")  # 원핫 레이블 (batch_size, num_classes)
    patient_ids = batch.get("patient_id", None)  # Optional patient IDs

    model.eval()
    with torch.no_grad():
        outputs = model(images)  # 모델 예측 (batch_size, num_classes)
        predictions = torch.argmax(outputs, dim=1)  # 예측 클래스 인덱스
        true_labels = torch.argmax(labels, dim=1)  # 실제 클래스 인덱스

    # Grad-CAM 초기화

    # 결과 저장 폴더 생성
    os.makedirs(output_dir, exist_ok=True)

    batch_size = images.size(0)
    grid_size = math.ceil(math.sqrt(batch_size))  # 정사각형 그리드 크기 계산
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size * 4, grid_size * 4))
    axes = axes.flatten()

    for i in range(batch_size):
        cam = GradCAM.from_config(model_type='vgg', arch=model.model, layer_name=target_layer)
        # 개별 이미지 처리
        cam_input_image = images[i].unsqueeze(0)
        grayscale_cam, _ = cam(cam_input_image)
        # 원본 이미지 복구
        input_image = denormalize_image(images[i].to('cpu'))  # 정규화 복구

        # Grad-CAM 오버레이 생성
        img_tensor = torch.tensor(input_image).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
        heatmap, result = visualize_cam(grayscale_cam, img_tensor)

        # 그리드에 시각화
        axes[i].imshow(result.permute(1, 2, 0).cpu().numpy())
        title = (
            f"ID: {patient_ids[i] if patient_ids else f'image_{i}'}\n"
            f"GT: {class_names[true_labels[i].item()]}\n"
            f"Pred: {class_names[predictions[i].item()]}"
        )
        axes[i].set_title(title, fontsize=8)
        axes[i].axis("off")

    # 남은 그리드 숨기기
    for j in range(batch_size, len(axes)):
        axes[j].axis("off")

    # 저장
    save_path = os.path.join(output_dir, "gradcam_batch.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)

    print(f"Grad-CAM visualization saved to {save_path}")