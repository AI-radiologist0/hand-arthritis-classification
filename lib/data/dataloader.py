# -----------------------------------------------------------
# 
# written by Jeongmin Kim (jm.kim@dankook.ac.kr)
#
# -----------------------------------------------------------
import os
import argparse
import logging
import json
import random
import torch
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms, datasets
from collections import Counter
from config import cfg, update_config
from utils.tools import set_seed

def get_basic_transforms(mean, std):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=30),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return transform

logger = logging.getLogger(__name__)

def get_augmentation_transforms(mean, std):
    """
    데이터 증강 옵션을 포함한 Transform 생성
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=30),
        # transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),

    ])

def get_data_loaders(data_path, batch_size, split_ratio):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=30),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = datasets.ImageFolder(data_path, transform=transform)

    train_size = int(split_ratio['train'] * len(dataset))
    val_size = int(split_ratio['validation'] * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_set, val_set, test_set = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

class MedicalImageDataset(Dataset):
    def __init__(self, cfg, transform=get_basic_transforms, augment=False, include_classes=None, num_workers=5):
        """
        Args:
            cfg: Configuration object containing dataset path.
            transform (callable, optional): A function/transform to apply to the images.
            include_classes (list, optional): A list of classes to include in the dataset.
        """
        json_file = cfg.DATASET.JSON
        with open(json_file, 'r') as f:
            data = json.load(f)  # Load the JSON file

        self.mean, self.std = cfg.DATASET.MEAN, cfg.DATASET.STD
        logger.info(f"Dataset mean {self.mean}, std {self.std}")

        self.data = []
        self.transform = transform(mean=self.mean, std=self.std)
        self.augment = augment
        self.include_classes = include_classes
        self.label_to_idx = {label: idx for idx, label in enumerate(include_classes)}
        self.idx_to_label = {v: k for k, v in self.label_to_idx.items()}

        for record in data["data"]:
            # Ensure file_path is valid
            file_path = record.get("file_path")
            if not file_path or (isinstance(file_path, list) and len(file_path) == 0):
                logger.info(f"Skipping record with invalid file_path: {record}")
                continue

            classes = [cls.strip() for cls in record["class"].split(',')]
            for cls in classes:
                if include_classes and cls in include_classes:
                    new_record = record.copy()
                    new_record["class"] = cls
                    self.data.append(new_record)

        logger.info(f"Total samples after filtering: {len(self.data)}")

        self.db_rec = self._initialize_db_rec(data['data'], num_workers=num_workers)
        
        self.class_counts = self.summary()
        self.num_classes = len(include_classes)

        self.show_class_count()
    
    def _initialize_db_rec(self, data, num_workers=5):
        """
        Filter and initialize the db_rec with original and augmented data.

        Args:
            data (list): List of dataset records from the JSON.

        Returns:
            list: A list of dictionaries containing patient_id, image, and label.
        """
        db_rec = []

        def process_record(record):
            """
            Process a single record to generate original and augmented images.
            """
            processed_records = []
            file_path = record.get("file_path")
            if not file_path or (isinstance(file_path, list) and len(file_path) == 0):
                return processed_records  # Skip invalid records

            classes = [cls.strip() for cls in record["class"].split(',')]
            for cls in classes:
                if self.include_classes and cls in self.include_classes:
                    label_idx = self.label_to_idx[cls]

                    # Load original image
                    if os.path.exists(file_path):
                        original_image = Image.open(file_path).convert("RGB")
                        if self.transform:
                            processed_original_image = self.transform(original_image)

                        # Append original data
                        processed_records.append({
                            'patient_id': record["patient_id"],
                            'image': processed_original_image,
                            'label': label_idx
                        })

                        # If augment, generate multiple augmented versions
                        if self.augment:
                            augment_transform = get_augmentation_transforms(mean=self.mean, std=self.std)
                            augmented_images = [augment_transform(original_image) for _ in range(3)]  # Generate 5 augmented images
                            for aug_img in augmented_images:
                                processed_records.append({
                                    'patient_id': record["patient_id"],
                                    'image': aug_img,
                                    'label': label_idx
                                })
                        else:
                            transform = get_basic_transforms(self.mean, self.std)
                            
            return processed_records

        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit tasks
            futures = {executor.submit(process_record, record): record for record in data}

            # Process results with tqdm
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing records"):
                result = future.result()
                if result:
                    db_rec.extend(result)

        return db_rec

    def show_class_count(self):
        logger.info("Class distribution:")
        for class_name, count in self.class_counts.items():
            logger.info(f"{class_name}: {count}")
    def __len__(self):
        """
        Returns the size of the dataset.
        """
        return len(self.db_rec)
    
    def __getitem__(self, idx):
        """
        Fetch a sample and its label for the given index.
        """
        """
        Fetch a sample and its label from db_rec using the given index.
        """
        record = self.db_rec[idx]
        label_idx = record['label']
        one_hot_label = np.zeros(self.num_classes, dtype=np.int64)
        one_hot_label[label_idx] = 1
        return {
            'patient_id': record['patient_id'],
            'image': record['image'],
            'label': one_hot_label
        }
        
    def summary(self):
        """
        Summarize the dataset, including the total number of samples and the number of samples for each class.
        """
        total_samples = len(self.db_rec)
        class_counts = Counter(record['label'] for record in self.db_rec)

        print(f"Total number of samples: {total_samples}")
        print("Number of samples per class:")
        for class_idx, count in class_counts.items():
            class_name = self.idx_to_label[class_idx]
            print(f"  {class_name}: {count}")
        
        return class_counts

    def calculate_mean_std(self):
        """
        Calculate the mean and standard deviation of all images in the dataset before augmentation.

        Returns:
            tuple: Mean and standard deviation of the dataset images.
        """
        all_pixels = []

        for record in self.data:
            file_path = record["file_path"]
            if os.path.exists(file_path):
                image = Image.open(file_path).convert("RGB")  # Load the image
                image = self.transform(image) if self.transform else transforms.ToTensor()(
                    image)  # Apply base transform
                all_pixels.append(image.numpy().flatten())

        # Concatenate all pixel data and calculate statistics
        all_pixels = np.concatenate(all_pixels)
        mean = np.mean(all_pixels)
        std = np.std(all_pixels)

        print(f"Mean: {mean:.4f}, Std: {std:.4f}")
        return mean, std
    
def create_test_loader(dataset, test_ratio=0.15, batch_size=8, num_workers=4, seed=42):
    """
    Create a test_loader by splitting the dataset.

    Args:
        dataset: The complete dataset.
        test_ratio (float): Ratio of the dataset to use for testing.
        batch_size (int): Batch size for the DataLoader.
        num_workers (int): Number of workers for the DataLoader.
        seed (int): Random seed for reproducibility.

    Returns:
        test_loader (DataLoader or None): DataLoader for the test set (None if test_ratio=0).
        train_val_idx (list): Indices of the remaining train/validation dataset.
        train_val_set (Subset): Subset for train/validation data.
    """
    set_seed(seed)

    # Split dataset indices
    indices = list(range(len(dataset)))

    if test_ratio > 0:
        train_val_idx, test_idx = train_test_split(indices, test_size=test_ratio, random_state=seed)
        test_set = Subset(dataset, test_idx)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    else:
        train_val_idx = indices  # Use all data for train/validation
        test_loader = None

    # Train and Validation subset
    train_val_set = Subset(dataset, train_val_idx)

    return test_loader, train_val_idx, train_val_set



if __name__ == "__main__":
    def parser_args():
        parser = argparse.ArgumentParser(description="hand-arthritis-classification")
        parser.add_argument('--cfg', help='experiment config file', default='experiments/ra_hand_classifier_RA_Normal_Kfold.yaml',
                            type=str)
        return parser.parse_args()

    arg = parser_args()
    update_config(cfg, arg)
    include_classes = cfg.DATASET.INCLUDE_CLASSES
    dataset = MedicalImageDataset(cfg, augment=False, include_classes=include_classes)
    mean, std = dataset.calculate_mean_std()
    print(mean, std)
    # dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=2)
    # _, train_val_indices, _ = create_test_loader(dataset)
    #
    # labels = [dataset.db_rec[idx]["label"] for idx in train_val_indices]
    # print(labels[:10])
    #
    # for batch in dataloader:
    #     print("Patient IDs:", batch["patient_id"])
    #     print("Images:", batch["image"].shape)  # (batch_size, 3, 224, 224)
    #     print("Labels:", batch["label"])
    #     break
    #
    # dataset.summary()
    
class OversampledDataset(Dataset):
    def __init__(self, base_dataset, target_count=None):
        """
        기존 Dataset 또는 Subset에서 Oversampling을 적용한 Dataset.

        Args:
            base_dataset (Dataset or Subset): 기존 Dataset 또는 Subset.
            target_count (int, optional): 클래스별 목표 데이터 수. None이면 최대 클래스 수로 설정.
        """
        self.base_dataset = base_dataset
        self.target_count = target_count

        # Subset인지 확인하여 처리
        if isinstance(base_dataset, Subset):
            self.original_dataset = base_dataset.dataset  # 원래 전체 데이터셋
            self.indices = base_dataset.indices  # Subset의 인덱스
            self.labels = [self.original_dataset.data[i]['class'] for i in self.indices]
        else:
            self.original_dataset = base_dataset
            self.indices = list(range(len(base_dataset)))  # 전체 인덱스
            self.labels = [record['class'] for record in base_dataset.data]

        self.class_counts = Counter(self.labels)
        self.max_class_count = max(self.class_counts.values())
        self.target_count = target_count or self.max_class_count

        # Oversampling된 데이터 인덱스 생성
        self.oversampled_indices = self._create_oversampled_indices()

    def _create_oversampled_indices(self):
        """
        클래스별 목표 데이터 수에 맞게 Oversampling된 인덱스를 생성합니다.

        Returns:
            list: Oversampled 데이터 인덱스 리스트.
        """
        oversampled_indices = []

        for label in self.class_counts:
            # 해당 클래스의 원래 인덱스 가져오기
            class_indices = [self.indices[i] for i, lbl in enumerate(self.labels) if lbl == label]

            # 목표 데이터 수에 맞게 인덱스 중복 추가
            oversampled_indices.extend(class_indices)
            additional_indices = random.choices(class_indices, k=self.target_count - len(class_indices))
            oversampled_indices.extend(additional_indices)

        # 전체 인덱스 섞기
        random.shuffle(oversampled_indices)
        return oversampled_indices

    def __len__(self):
        """
        Oversampling된 데이터셋의 크기.
        """
        return len(self.oversampled_indices)

    def __getitem__(self, idx):
        """
        Oversampling된 인덱스를 사용해 원래 데이터셋의 샘플을 반환.
        """
        base_idx = self.oversampled_indices[idx]
        return self.original_dataset[base_idx]