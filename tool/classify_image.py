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
import random
import numpy as np
from torch.utils.data import DataLoader
from config import cfg
from config import update_config
import models
from core.trainer import Trainer
from utils.tools import split_dataset_parallel, balance_dataset_parallel
from utils.vis import visualize_batch_with_gt_pred
from utils.gradcam import visualize_gradcam_batch
from data import MedicalImageDataset

def set_seed(seed):
    """Set the random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parser_args():
    parser = argparse.ArgumentParser(description="hand-arthritis-classification")
    parser.add_argument('--ckpt', help='path to checkpoint to resume training', type=str, default='output/RA_hand_merge_cau_2024-12-14_22-35-43_epoch_0_250_classifier_ra_normal/ckpt/best_model.pth.tar')
    parser.add_argument('--cfg', help='experiment config file', default='experiments/ra_hand_classifier_RA_Normal.yaml', type=str)
    parser.add_argument('--seed', help='random seed for reproducibility', type=int, default=42)
    return parser.parse_args()

def create_logger(output_dir):
    """Create a logger for logging training process."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # Set the log level explicitly
    handler = logging.FileHandler(os.path.join(output_dir, 'validation_test.log'))
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)

    return logger

def validate_and_test():
    args = parser_args()
    set_seed(args.seed)
    update_config(cfg, args)
    cfg.freeze()

    # Get current timestamp and config name for unique output
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    cfg_name = os.path.splitext(os.path.basename(cfg.DATASET.JSON))[0]

    # get device from cfg
    device = torch.device('cuda' if cfg.DEVICE == 'GPU' else 'cpu')

    # output / log file path
    final_output_dir = os.path.join('output', f"{cfg_name}_{timestamp}_validate_test")
    logger = create_logger(final_output_dir)
    logger.info(f"Using configuration: {cfg}")

    # model create
    model = models.create(cfg.MODEL.NAME, cfg, is_train=False)
    model = model.to(device)

    # Data Loading code
    dataset = MedicalImageDataset(cfg, augment=False, include_classes=cfg.DATASET.INCLUDE_CLASSES)
    dataset = balance_dataset_parallel(dataset)

    train_loader, val_loader, test_loader = split_dataset_parallel(
        dataset=dataset,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU,
        num_workers=cfg.WORKERS,
        logger=logger
    )

    # Load checkpoint
    if args.ckpt:
        if os.path.isfile(args.ckpt):
            logger.info(f"Loading checkpoint from {args.ckpt}")
            checkpoint = torch.load(args.ckpt)
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Checkpoint loaded successfully.")
        else:
            logger.error(f"Checkpoint {args.ckpt} not found.")
            return

    # Trainer for validation and testing
    trainer = Trainer(cfg, model, output_dir=final_output_dir, writer_dict=None)

    # Validate
    logger.info("Starting validation...")
    val_perf = trainer.validate(0, val_loader)
    logger.info(f"Validation performance: {val_perf:.4f}")

    # Test
    logger.info("Starting testing...")
    test_perf = trainer.validate(0, test_loader)
    logger.info(f"Test performance: {test_perf:.4f}")

    # visualization
    for i, batch in enumerate(val_loader):
        visualize_batch_with_gt_pred(batch, model, i, class_names=cfg.DATASET.INCLUDE_CLASSES)
        visualize_gradcam_batch(batch, model, target_layer="features_49", class_names=cfg.DATASET.INCLUDE_CLASSES, output_dir='./gradcam')
        break

if __name__ == "__main__":
    validate_and_test()
