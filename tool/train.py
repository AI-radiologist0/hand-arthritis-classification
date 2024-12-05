# -----------------------------------------------------------
# 
# written by Jeongmin Kim (jm.kim@dankook.ac.kr)
#
# -----------------------------------------------------------
import argparse
import yaml

import torch
from torch import optim
from config import cfg
from config import update_config
import models
from core.trainer import Trainer
from utils.tools import split_dataset

def parser_args():
    parser = argparse.ArgumentParser(description="hand-arthritis-classification")
    parser.add_argument('--dfg', dest='config', help='experiment config file', default='experiments/train.yaml', type=str)
    return parser.parse_args()

def main():
    args = parser_args()
    update_config(cfg, args)
    cfg.freeze()

    # create logger
    # logger -----

    # get device from cfg
    device = torch.device(cfg.DEVICE)

    # output / log file path
    final_output_dir = 'path/'

    # model create
    model = models.create(cfg.MODEL.NAME, cfg, is_train=True)
    model = model.to(device)

    # optimizer
    # optimizer = get_optimizer(cfg, model.parameters(), awl.parameters())
    optimizer = optim.Adam(model.parameters(), lr=cfg.TRAIN.LR)

    # Data Loading code
    # 데이터 로딩 및 분할
    train_loader, val_loader, test_loader = split_dataset(
        data_dir=cfg.DATASET.PATH,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU,
        num_workers=cfg.WORKERS
    )

    # Trainier
    trainer = Trainer(cfg, model, output_dir=final_output_dir, writer_dict=None)


    best_perf = -1
    last_epoch = -1

    begin_epoch = cfg.TRAIN.BEGIN_EPOCH
    end_epoch = cfg.TRAIN.END_EPOCH

    # train per epoch
    for epoch in range(begin_epoch, end_epoch):
        trainer.train(epoch,train_loader, optimizer)

        val_perf = trainer.validate(epoch, val_loader)
        if val_perf > best_perf:
            best_perf = val_perf
            print(f"New best performance: {best_perf:.4f}, saving model...")
            torch.save({'epoch': epoch + 1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_perf': best_perf}, best_model_path)