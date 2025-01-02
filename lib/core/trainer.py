# -----------------------------------------------------------
# 
# written by Jeongmin Kim (jm.kim@dankook.ac.kr)
#
# -----------------------------------------------------------
import logging
import os
import time

import torch
import torch.nn as nn
from utils import AverageMeter
from collections import Counter

class Validator:
    def __init__(self):
        pass


class Trainer:
    def __init__(self, cfg, model, output_dir, writer_dict):
        self.device = 'cuda' if cfg.DEVICE == "GPU" else 'cpu'
        self.model = model
        self.output_dir = output_dir  # cfg로 설정
        self.print_freq = cfg.PRINT_FREQ
        self.writer_dict = writer_dict
        self.val_loss = None
        self.val_accuracy = None


    def train(self, epoch, data_loader, optimizer, scheduler):
        # logger
        logger = logging.getLogger("Training")
        batch_time = AverageMeter()
        data_time = AverageMeter()
        loss_meter = AverageMeter()

        criterion = nn.CrossEntropyLoss()
        self.model.train()

        end = time.time()

        for i, input_infos in enumerate(data_loader):

            data_time.update(time.time() - end)
            # output (B, num_classes) -> softmax, digit형태로
            inputs, labels = input_infos['image'].to(self.device), input_infos['label'].to(self.device)
            labels = torch.argmax(labels, dim=1)
          

            outputs = self.model(inputs)

            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            loss_meter.update(loss.item(), inputs.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            # logger -----)
            if i % self.print_freq == 0:
                msg = f'Epoch: [{epoch}] [{i}/{len(data_loader)}] \t ' \
                      f'Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s) \t' \
                      f'Data Time: {data_time.val:.3f}s ({data_time.avg:.3f}s) \t' \
                      f'Loss: {loss_meter.val:.4f} ({loss_meter.avg:.4f})'
                logger.info(msg)

                logger.info(f"Current learning rate: {optimizer.param_groups[0]['lr']}")

            writer = self.writer_dict['writer']
            global_steps = self.writer_dict['train_global_steps']
            writer.add_scalar('train_loss', loss_meter.val, global_steps)
            # writer.add_scalar("train_acc", ) 
            self.writer_dict['train_global_steps'] = global_steps + 1
        scheduler.step()
        # if self.val_loss is not None:
        #     logger.info(f"scheduler make learning rate decrease {self.val_loss}")
        #     scheduler.step(self.val_loss)
        # if self.val_accuracy is not None:
        #     logger.info(f"scheduler make learning rate decrease {self.val_accuracy}")
        #     scheduler.step(self.val_accuracy)
        
            

    def validate(self, epoch, val_loader):
        batch_time = AverageMeter()
        losses = AverageMeter()
        logger = logging.getLogger("Validation")

        criterion = nn.CrossEntropyLoss()
        self.model.eval()

        correct = 0
        total = 0

        with torch.no_grad():
            end = time.time()
            for i, input_infos in enumerate(val_loader):              
                
                inputs, labels = input_infos['image'].to(self.device), input_infos['label'].to(self.device)
                labels = torch.argmax(labels, dim=1)
                # 모델의 순전파 및 손실 계산
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)

                # 손실 업데이트
                losses.update(loss.item(), inputs.size(0))

                # 정확도 계산
                # preds = torch.softmax(outputs, dim=1)
                preds = torch.argmax(outputs, dim=1)
                # preds_one_hot = torch.zeros_like(preds).scatter_(1, preds.argmax(dim=1, keepdim=True),1)

                # _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

                batch_time.update(time.time() - end)
                end = time.time()

            
        # 정확도 계산
        accuracy = 100.0 * correct / total
        if self.writer_dict:
            writer = self.writer_dict['writer']
            global_steps = self.writer_dict['valid_global_steps']
            writer.add_scalar('valid_loss', losses.avg, global_steps)
            writer.add_scalar('valid_acc', accuracy, global_steps)
            self.writer_dict['valid_global_steps'] = global_steps + 1

        self.val_loss = losses.avg
        self.val_accuracy = accuracy
        logger.info(f'Validation Accuracy: {accuracy:.2f}% \t Loss: {losses.avg:.4f}')
        return accuracy, losses.avg