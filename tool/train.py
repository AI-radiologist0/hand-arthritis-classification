import argparse
import os
import sys
import yaml
import logging
from datetime import datetime
import wandb

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import _init_path
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from config import cfg, update_config
import models
from core.trainer import Trainer
from core.kfold_handler import run_kfold_training
from utils.tools import split_dataset_parallel, balance_dataset, set_seed, EarlyStopping, BestModelSaver, log_misclassified_images
from data import MedicalImageDataset
from data.dataloader import create_test_loader

def parser_args():
    parser = argparse.ArgumentParser(description="hand-arthritis-classification")
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume training')
    parser.add_argument('--cfg', type=str, default='experiments/ra_hand_classifier_RA_Normal_Kfold.yaml', help='Experiment config file')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    return parser.parse_args()

def create_logger(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
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
    
    # Initialize wandb
    wandb.init(project="hand-arthritis-classification", config=dict(cfg))
    
    begin_epoch = cfg.TRAIN.BEGIN_EPOCH
    end_epoch = cfg.TRAIN.END_EPOCH
    
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    cfg_name = os.path.splitext(os.path.basename(cfg.DATASET.JSON))[0]
    label_list = cfg.DATASET.INCLUDE_CLASSES
    str_label_list = '_'.join(label_list)
    device = torch.device('cuda' if cfg.DEVICE == 'GPU' else 'cpu')
    
    final_output_dir = os.path.join('output', f"{cfg_name}_{timestamp}_classifier_{str_label_list}")
    ckpt_save_dir = os.path.join(final_output_dir, 'ckpt')
    os.makedirs(ckpt_save_dir, exist_ok=True)
    
    best_model_path = os.path.join(ckpt_save_dir, 'best_model.pth.tar')
    final_model_path = os.path.join(ckpt_save_dir, 'final_model.pth.tar')
    
    logger = create_logger(final_output_dir)
    logger.info(f"Using configuration: {cfg}")
    
    writer_dict = {'writer': SummaryWriter(log_dir=os.path.join(final_output_dir, 'tensorboard')),
                   "train_global_steps": 0,
                   "valid_global_steps": 0}
    
    model = models.create(cfg.MODEL.NAME, cfg, is_train=True).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.TRAIN.LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    dataset = MedicalImageDataset(cfg, augment=cfg.DATASET.AUGMENT, include_classes=label_list)
    dataset = balance_dataset(dataset)
    dataset.show_class_count()
    
    test_loader, train_val_idx, train_val_set = create_test_loader(
        dataset, cfg.TEST.TEST_SET_RATIO, cfg.TRAIN.BATCH_SIZE_PER_GPU, cfg.WORKERS, args.seed
    )
    
    trainer = Trainer(cfg, model, output_dir=final_output_dir, writer_dict=writer_dict)
    
    if cfg.KFOLD.USE_KFOLD:
        run_kfold_training(cfg, dataset, model, optimizer, scheduler, cfg.KFOLD.KFOLD_SIZE, 42, cfg.KFOLD.P,
                           final_output_dir, final_output_dir, train_val_idx, test_loader, writer_dict)
    else:
        train_loader, val_loader = split_dataset_parallel(dataset, train_val_idx, 0.7, 0.3, cfg.TRAIN.BATCH_SIZE_PER_GPU, cfg.WORKERS)
        best_model_saver = BestModelSaver(save_path=best_model_path)
        
        for epoch in range(begin_epoch, end_epoch):
            train_loss = trainer.train(epoch, train_loader, optimizer, scheduler)
            val_perf, val_loss, precision, recall, f1 = trainer.validate_with_metric(epoch, val_loader)
            log_misclassified_images(cfg, trainer, val_loader, epoch)
            
            # Log to wandb
            wandb.log({"train_loss": train_loss, "val_loss": val_loss, "val_performance": val_perf,
                       "precision": precision, "recall": recall, "f1_score": f1, "epoch": epoch}, step=epoch)
            
            best_model_saver.update(val_loss, val_perf, model, epoch)
        
        best_model_saver.save_final_model(model, save_path=final_model_path)
        test_perf, _ = trainer.validate_with_metric(0, test_loader)
        
        logger.info(f"Test performance: {test_perf:.4f}")
        
        # Save best model to wandb
        wandb.save(best_model_path)
        
if __name__ == "__main__":
    main()
