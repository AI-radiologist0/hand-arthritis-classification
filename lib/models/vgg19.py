# -----------------------------------------------------------
# 
# written by Jeongmin Kim (jm.kim@dankook.ac.kr)
#
# -----------------------------------------------------------
import torch
import torch.nn as nn
from torchvision import models


class VGG19(nn.Module):
    def __init__(self, cfg, is_train):
        super(VGG19, self).__init__()
        self.is_train = is_train
        self.num_classes = cfg.MODEL.NUM_CLASSES
        self.device = torch.device(cfg.DEVICE)

        # VGG19 모델 로드
        self.model = models.vgg19(pretrained=True)

        # 마지막 fully connected layer 수정
        self.model.classifier[6] = nn.Linear(4096, self.num_classes)

        # 훈련 모드일 때만 dropout을 활성화
        if self.is_train:
            self.model.train()
        else:
            self.model.eval()

    def forward(self, x):
        return self.model(x.to(self.device))

    def get_classifier(self):
        return self.model.classifier[6]  # 마지막 fully connected layer

    def load_pretrained(self, checkpoint_path):
        """훈련된 모델 가중치 로드"""
        checkpoint = torch.load(checkpoint_path)
        self.load_state_dict(checkpoint['model_state_dict'])

    def save_checkpoint(self, checkpoint_path):
        """모델 가중치를 저장"""
        torch.save({
            'model_state_dict': self.state_dict(),
        }, checkpoint_path)
