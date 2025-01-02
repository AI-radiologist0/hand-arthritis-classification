# -----------------------------------------------------------
# 
# written by Jeongmin Kim (jm.kim@dankook.ac.kr)
#
# -----------------------------------------------------------
import argparse
import os
import sys
import yaml
import logging
from datetime import datetime



sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import _init_path
import torch
torch.cuda.empty_cache()
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from config import cfg
from config import update_config
import models
from core.trainer import Trainer
from core.kfold_handler import run_kfold_training
from utils.tools import split_dataset_parallel, balance_dataset, set_seed, EarlyStopping, BestModelSaver
from data import MedicalImageDataset
from data.dataloader import create_test_loader

def parser_args():
    parser = argparse.ArgumentParser(description="hand-arthritis-classification")
    parser.add_argument('--resume', help='path to checkpoint to resume training', type=str, default=None)
    parser.add_argument('--cfg', help='experiment config file', default='experiments/ra_hand_classifier_RA_Normal_Kfold.yaml', type=str)
    parser.add_argument('--seed', help='random seed for reproducibility', type=int, default=42)

    return parser.parse_args()


def create_logger(output_dir):
    """Create a logger for logging training process."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # Set the log level explicitly
    handler = logging.FileHandler(os.path.join(output_dir, 'training.log'))
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)

    return logger


def main():
    args = parser_args()
    set_seed(args.seed)
    update_config(cfg, args)
    cfg.freeze()

    begin_epoch = cfg.TRAIN.BEGIN_EPOCH
    end_epoch = cfg.TRAIN.END_EPOCH

    # Get current timestamp and config name for unique output
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    cfg_name = os.path.splitext(os.path.basename(cfg.DATASET.JSON))[0]
    # Get list of labels to classify from config file. 
    label_list = cfg.DATASET.INCLUDE_CLASSES
    str_label_list = '_'.join(label_list)
    # get device from cfg
    device = torch.device('cuda' if cfg.DEVICE == 'GPU' else 'cpu')
    # output / log file path
    final_output_dir = os.path.join('output', f"{cfg_name}_{timestamp}_epoch_{begin_epoch}_{end_epoch}_classifier_{str_label_list}_kfold_usage_{cfg.KFOLD.USE_KFOLD}_{cfg.KFOLD.P}")
    

    # Output / log file path
    ckpt_save_dir = os.path.join(final_output_dir, 'ckpt')
    if not os.path.exists(ckpt_save_dir):
        os.makedirs(ckpt_save_dir)
    best_model_path = os.path.join(ckpt_save_dir, 'best_model.pth.tar')
    final_model_path = os.path.join(ckpt_save_dir, 'final_model.pth.tar')

    logger = create_logger(final_output_dir)
    logger.info(f"Using configuration: {cfg}")

    # Initialize TensorBoard
    writer_dict = {'writer': SummaryWriter(log_dir=os.path.join(final_output_dir, 'tensorboard')),
                   "train_global_steps": 0,
                   "valid_global_steps": 0}

    # model create
    model = models.create(cfg.MODEL.NAME, cfg, is_train=True)
    model = model.to(device)

    # optimizer
    # optimizer = get_optimizer(cfg, model.parameters(), awl.parameters())
    optimizer = optim.Adam(model.parameters(), lr=cfg.TRAIN.LR, weight_decay=1e-4)

    # Scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # Data Loading code
    dataset = MedicalImageDataset(cfg, augment=cfg.DATASET.AUGMENT, include_classes=label_list)
    dataset = balance_dataset(dataset)
    dataset.show_class_count()
    
    test_loader, train_val_idx, train_val_set = create_test_loader(
    dataset=dataset,
    test_ratio=cfg.TEST.TEST_SET_RATIO,
    batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU,
    num_workers=cfg.WORKERS,
    seed=args.seed
    )
    trainer = Trainer(cfg, model, output_dir=final_output_dir, writer_dict=writer_dict)
    
    
    # Kfold phase, training phase
    if cfg.KFOLD.USE_KFOLD:
        # Run K-Fold training
        run_kfold_training(
            cfg=cfg,
            dataset=dataset,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            num_folds=cfg.KFOLD.KFOLD_SIZE,
            seed=42,
            p = cfg.KFOLD.P,
            final_output_dir=final_output_dir,
            output_dir=final_output_dir,
            train_val_indices=train_val_idx,
            test_loader=test_loader,
            writer_dict=writer_dict,
        )
    else:
        # dataloader = DataLoader(dataset)
        # 데이터 로딩 및 분할
        train_loader, val_loader = split_dataset_parallel(
            dataset=dataset,
            train_val_idx=train_val_idx,
            train_ratio=0.7,
            val_ratio=0.3,
            batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU,
            num_workers=cfg.WORKERS,
        )

        # Trainier
        best_model_saver = BestModelSaver(save_path=f"{final_output_dir}/bestmodel.pth.tar")
        # train per epoch
        for epoch in range(begin_epoch, end_epoch):
            trainer.train(epoch, train_loader, optimizer, scheduler)
            # scheduler.step()
            val_perf, val_loss = trainer.validate(epoch, val_loader)
            best_model_saver.update(val_loss, val_perf, model, epoch)

        best_model_saver.save_final_model(model)

        test_perf, _ = trainer.validate(0, test_loader)
        logger.info(f"Test performance: {test_perf:.4f}")


if __name__ == "__main__":

    main()
