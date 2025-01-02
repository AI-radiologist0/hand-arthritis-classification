# -----------------------------------------------------------
# 
# written by Jeongmin Kim (jm.kim@dankook.ac.kr)
#
# -----------------------------------------------------------
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import torch
# from gradcam import GradCAM

def denormalize_image(image_tensor):
    """
    이미지 텐서를 denormalize하여 원래 색상으로 복원합니다.
    Args:
        image_tensor (torch.Tensor): (C, H, W)의 정규화된 이미지 텐서.

    Returns:
        np.ndarray: 복원된 이미지 (H, W, C).
    """
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = image_tensor.permute(1, 2, 0).numpy()  # (C, H, W) -> (H, W, C)
    image = std * image + mean  # Denormalize
    image = np.clip(image, 0, 1)  # Clip values to [0, 1]
    return image

def visualize_batch_with_gt_pred(batch, model, idx, class_names, output_dir="vis_img"):
    """
    배치 데이터를 시각화하고 GT(Ground Truth)와 Pred(Prediction)를 각각 표시하며 결과를 저장합니다.
    Args:
        batch (dict): DataLoader에서 가져온 배치 데이터 (image, label 등 포함).
        model (torch.nn.Module): 학습된 모델.
        class_names (list): 클래스 이름 리스트.
        output_dir (str): 시각화 결과를 저장할 폴더 경로.
    """
    patient_ids = batch.get("patient_id", None)
    images = batch["image"].to('cuda')  # (batch_size, C, H, W)
    labels = batch["label"].to('cuda')  # 원핫 벡터 형태 (batch_size, num_classes)

    model.eval()
    with torch.no_grad():
        outputs = model(images)  # 예측 결과 (batch_size, num_classes)
        predictions = torch.argmax(outputs, dim=1)  # 예측 레이블 (정수)
        true_labels = torch.argmax(labels, dim=1)  # 원핫 레이블 -> 정수 변환

    batch_size = images.size(0)
    fig, axes = plt.subplots(1, batch_size, figsize=(15, 5))
    # 그리드 크기 계산
    grid_size = math.ceil(math.sqrt(batch_size))  # 정사각형에 가까운 그리드 크기
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size * 4, grid_size * 4))

    axes = axes.flatten()

    images = images.to('cpu')
    for i in range(batch_size):
        ax = axes[i]
        # 이미지 복원 및 표시
        image = denormalize_image(images[i])
        ax.imshow(image)
        ax.axis("off")

        # 레이블 추가
        true_label = class_names[true_labels[i].item()]  # Ground Truth
        predicted_label = class_names[predictions[i].item()]  # Predicted Label
        patient_id = patient_ids[i] if patient_ids is not None else "Unknown"

        # 타이틀 색상: 정답 여부에 따라 다르게 표시
        title_color = "green" if true_label == predicted_label else "red"
        title = f"ID: {patient_id}\nGT: {true_label}\nPred: {predicted_label}"
        ax.set_title(title, fontsize=10, color=title_color)

        ax.text(
            0.5, -0.15, title,  # x, y 위치
            fontsize=8,
            color=title_color,
            ha="center",
            va="center",
            transform=ax.transAxes
        )


    # 남은 빈칸 숨기기
    for j in range(batch_size, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()

    # 저장
    filename = f"batch_visualization.png"
    save_visualization(fig, output_dir, filename)

    # 시각화 닫기
    plt.close(fig)

def save_visualization(fig, output_dir, filename):
    """
    시각화 결과를 지정된 폴더에 저장합니다.
    Args:
        fig (matplotlib.figure.Figure): 저장할 Matplotlib Figure.
        output_dir (str): 저장할 폴더 경로.
        filename (str): 저장할 파일 이름.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    save_path = os.path.join(output_dir, filename)
    fig.savefig(save_path)
    print(f"Visualization saved to {save_path}")

