# -----------------------------------------------------------
# 
# written by Jeongmin Kim (jm.kim@dankook.ac.kr)
#
# -----------------------------------------------------------
import torch
import torch.nn as nn

class YOLOv5(nn.Module):
    def __init__(self, model_name='yolov5s', pretrained=True, num_classes=80):
        """
        YOLOv5 모델 클래스

        Args:
            model_name (str): 사용할 YOLOv5 모델의 이름 (e.g., 'yolov5s', 'yolov5m', 'yolov5l', 'yolov5x')
            pretrained (bool): 사전 학습된 가중치 사용 여부
            num_classes (int): 출력 클래스 수
        """
        super(YOLOv5, self).__init__()

        # torch.hub를 통해 YOLOv5 모델 로드
        self.model = torch.hub.load('ultralytics/yolov5', model_name, pretrained=pretrained)

        # 클래스 수에 맞게 출력 레이어 수정
        self.model.model[-1].nc = num_classes  # number of classes
        self.model.model[-1].bias = nn.Parameter(torch.zeros(num_classes))  # initialize bias
        self.model.names = [str(i) for i in range(num_classes)]  # update class names

    def forward(self, x):
        """
        Forward pass

        Args:
            x (torch.Tensor): 입력 이미지 텐서

        Returns:
            결과 텐서
        """
        return self.model(x)

    def detect(self, images):
        """
        이미지에서 객체를 탐지합니다.

        Args:
            images (torch.Tensor): 입력 이미지 텐서

        Returns:
            list: 탐지된 객체들의 좌표 및 클래스 정보
        """
        results = self.model(images)
        return [x[:, :4].cpu().detach().numpy() for x in results.xyxy]  # bbox 좌표 반환

    def load_pretrained(self, checkpoint_path):
        """
        저장된 체크포인트에서 모델 가중치를 로드합니다.

        Args:
            checkpoint_path (str): 체크포인트 경로
        """
        checkpoint = torch.load(checkpoint_path)
        self.load_state_dict(checkpoint['model_state_dict'])

    def save_checkpoint(self, checkpoint_path):
        """
        현재 모델의 가중치를 체크포인트로 저장합니다.

        Args:
            checkpoint_path (str): 저장할 경로
        """
        torch.save({
            'model_state_dict': self.state_dict(),
        }, checkpoint_path)
