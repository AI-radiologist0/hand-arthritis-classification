# -----------------------------------------------------------
# 
# written by Jeongmin Kim (jm.kim@dankook.ac.kr)
#
# -----------------------------------------------------------
import os
import time

import torch
import torch.nn as nn
from utils import AverageMeter

class Validator:
    def __init__(self):
        pass


class Trainer:
    def __init__(self, cfg, model, output_dir, writer_dict):
        self.device = cfg.DEVICE
        self.model = model
        self.output_dir = output_dir  # cfg로 설정
        self.print_freq = cfg.PRINT_FREQ


    def train(self, epoch, data_loader, optimizer):
        # logger
        batch_time = AverageMeter()
        data_time = AverageMeter()
        loss = AverageMeter()

        criterion = nn.CrossEntropyLoss()
        self.model.train()

        end = time.time()

        for i, inputs, labels in enumerate(data_loader):
            data_time.update(time.time() - end)
            # output (B, num_classes) -> softmax, digit형태로
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            optimizer.zero_grad()

            outputs = self.model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            loss.update(loss.item(), inputs.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            # logger -----)
            if i % self.print_freq == 0:
                msg = 'Epoch: [{0}] [{1}/{2}] \t ' \
                      f'Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s) \t' \
                      f'Data Time: {data_time.val:.3f}s ({data_time.avg:.3f}s) \t' \
                      f'Loss: {loss.val:.4f} ({loss.avg:.4f})'
                print(msg)

    def validate(self, epoch, val_loader):
        batch_time = AverageMeter()
        losses = AverageMeter()

        criterion = nn.CrossEntropyLoss()
        self.model.eval()

        correct = 0
        total = 0

        with torch.no_grad():
            end = time.time()
            for i, (inputs, labels) in enumerate(val_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # 모델의 순전파 및 손실 계산
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)

                # 손실 업데이트
                losses.update(loss.item(), inputs.size(0))

                # 정확도 계산
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

                batch_time.update(time.time() - end)
                end = time.time()

        # 정확도 계산
        accuracy = 100.0 * correct / total
        print(f'Validation Accuracy: {accuracy:.2f}% \t Loss: {losses.avg:.4f}')
        return accuracy