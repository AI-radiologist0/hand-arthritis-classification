# -----------------------------------------------------------
# 
# written by Jeongmin Kim (jm.kim@dankook.ac.kr)
#
# -----------------------------------------------------------
import os
import sys
import argparse
import json
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import _init_path

import torch
from torch.utils.data import DataLoader
from utils.tools import split_dataset_parallel, balance_dataset, set_seed
from data import MedicalImageDataset
from config import cfg
from config import update_config
from data.dataloader import create_test_loader


def parser_args():
    parser = argparse.ArgumentParser(description="hand-arthritis-classification")
    parser.add_argument('--resume', help='path to checkpoint to resume training', type=str, default=None)
    parser.add_argument('--cfg', help='experiment config file', default='experiments/ra_hand_classifier_RA_Normal.yaml',
                        type=str)
    parser.add_argument('--seed', help='random seed for reproducibility', type=int, default=42)

    return parser.parse_args()


if __name__ == "__main__":

    args = parser_args()
    set_seed(args.seed)
    update_config(cfg, args)
    cfg.freeze()

    label_list = cfg.DATASET.INCLUDE_CLASSES

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

    train_loader, val_loader = split_dataset_parallel(
        dataset=dataset,
        train_val_idx=train_val_idx,
        train_ratio=0.7,
        val_ratio=0.3,
        batch_size=1,
        num_workers=cfg.WORKERS,
    )

    json_file_path = cfg.DATASET.JSON

    with open(json_file_path, "r") as json_file:
        json_data = json.load(json_file)

    file_pid_to_idx = {entry['patient_id']: idx for idx, entry in enumerate(json_data["data"])}

    original_data = json_data['data']
    correct = 0
    # total = len(train_val_idx)
    total = len(train_loader)

    for i, item in enumerate(train_loader):
        label = dataset.idx_to_label[np.argmax(item['label'].numpy())]
        patient_id = item['patient_id'][0]

        if original_data[file_pid_to_idx[patient_id]]['class'] == label:
            correct += 1
        else:
            print(f"org : {original_data[file_pid_to_idx[patient_id]]['class']} / tmp : {label}")

    # for idx in train_val_idx:
    #     label = dataset.idx_to_label[np.argmax(dataset[idx]['label'])]
    #     patient_id = dataset[idx]['patient_id']
    #
    #     if original_data[file_pid_to_idx[patient_id]]['class'] == label:
    #
    #         correct += 1
    #     else:
    #         print(f"org : {original_data[file_pid_to_idx[patient_id]]['class']} / tmp : {label}")

    print(correct / total * 100)
