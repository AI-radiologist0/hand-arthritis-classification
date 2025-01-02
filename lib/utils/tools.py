from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import torch
import os, re
import logging
import numpy as np
import random

from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split

""" Check Device and Path for saving and loading """
logger = logging.getLogger(__name__)

def check_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir


def find_latest_ckpt(folder):
    """ find latest checkpoint """
    files = []
    for fname in os.listdir(folder):
        s = re.findall(r'\d+', fname)
        if len(s) == 1:
            files.append((int(s[0]), fname))
    if files:
        file = max(files)[1]
        file_name = os.path.splitext(file)[0]
        previous_iter = int(file_name.split("_")[1])
        return file, previous_iter
    else:
        return None, 0


""" Training Tool for model """


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def apply_gradients(loss, optim):
    optim.zero_grad()
    loss.backward()
    optim.step()


def infinite_iterator(loader):
    while True:
        for batch in loader:
            yield batch


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def cross_entroy_loss(logit, label):
    loss = torch.nn.CrossEntropyLoss()(logit, label)
    return loss


def accuracy(outputs, label):
    """ if you want to make custom accuracy for your model, you need to implement this function."""
    y = torch.argmax(outputs, dim=1)
    return (y.eq(label).sum())


def reduce_loss(tmp):
    """ will implement reduce_loss func """
    loss = tmp
    return loss


# def reduce_loss_dict(loss_dict):
#     world_size = get_world_size()
#
#     if world_size < 2:
#         return loss_dict
#
#     with torch.no_grad():
#         keys = []
#         losses = []
#
#         for k in sorted(loss_dict.keys()):
#             keys.append(k)
#             losses.append(loss_dict[k])
#
#         losses = torch.stack(losses, 0)
#         dist.reduce(losses, dst=0)
#
#         if dist.get_rank() == 0:
#             losses /= world_size
#
#         reduced_losses = {k: v.mean().item() for k, v in zip(keys, losses)}
#
#     return reduced_losses

""" Tool to set for model by loading config files """


def read_config(config_path):
    """ read config file """
    file = open(config_path, 'r')
    lines = file.read().split('\n')
    lines = [x for x in lines if x and not x.startswith('#')]
    lines = [x.rstrip().lstrip() for x in lines]

    return lines


def parse_model_config(config_path):
    """ Parse your model of configuration files and set module defines"""
    lines = read_config(config_path)
    module_configs = []

    for line in lines:
        if line.startswith('['):
            layer_name = line[1:-1].rstrip()
            if layer_name == "net":
                continue
            module_configs.append({})
            module_configs[-1]['type'] = layer_name

            if module_configs[-1]['type'] == 'convolutional':
                module_configs[-1]['batch_normalize'] = 0
        else:
            if layer_name == "net":
                continue
            key, value = line.split("=")
            value = value.strip()
            if value.startswith('['):
                module_configs[-1][key.rstrip()] = list(map(int, value[1:-1].rstrip().split(',')))
            else:
                module_configs[-1][key.rstrip()] = value.strip()

    return module_configs


""" If you want visualize_inference to fit your model, you need to implement below func."""


def visualize_inference(img, label, batch_size):
    """ Visualize Image Batch"""
    fig, axes = plt.subplots(1, batch_size, figsize=(10, 10))
    for i in range(batch_size):
        axes[i].imshow(np.squeeze(img[i]), cmap="gray")
        axes[i].set_title(f"predicted: {label[i]}")
        axes[i].axis('off')
    plt.show()


def visualize_feature_map(model, image):
    model.network.eval()

    # transformer
    # transform = mnist_transform()

    feature_map = None

    # input_tensor
    input_tensor = image.to(check_device())

    def hook(module, input, output):
        nonlocal feature_map
        feature_map = output.detach().cpu()

    target_layer = model.network.layers[6]
    hook_handle = target_layer.register_forward_hook(hook)

    with torch.no_grad():
        model.network(input_tensor)

    hook_handle.remove()

    plt.figure(figsize=(12, 8))
    for i in range(feature_map.size(1)):
        plt.subplot(4, 8, i + 1)
        plt.imshow(feature_map[0, i], cmap='viridis')
        plt.axis('off')
    plt.show()


def show_img(img):
    """ Display an img"""
    img = np.array(img, dtype=np.uint8)
    img = Image.fromarray(img)
    img.show()


def data_transform(img_size):
    transform_list = [
        transforms.Resize(size=[img_size, img_size]),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),  # [0, 255] -> [0, 1]
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),  # [0, 1] -> [-1, 1]
    ]
    return transforms.Compose(transform_list)


def mnist_transform():
    transforms_list = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ]
    return transforms.Compose(transforms_list)


def create_sampler(subset, dataset):
    """
    주어진 Subset에 대해 WeightedRandomSampler를 생성합니다.

    Args:
        subset (Subset): 데이터 Subset
        dataset (Dataset): 전체 Dataset

    Returns:
        WeightedRandomSampler: 클래스 균등성을 위한 샘플러
    """
    # Subset의 클래스 분포 계산
    class_counts = dataset.class_counts
    total_samples = sum(class_counts.values())

    # 클래스별 가중치 계산 (최대 가중치 제한)
    max_weight = 10.0
    class_weights = {cls: min(total_samples / count, max_weight) for cls, count in class_counts.items()}

    # 샘플별 가중치 생성
    sample_weights = []
    for idx in subset.indices:
        label = torch.argmax(torch.from_numpy(dataset[idx]['label'])).item()
        sample_weights.append(class_weights[dataset.idx_to_label[label]])

    # WeightedRandomSampler 생성
    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.float),
        num_samples=len(subset),
        replacement=True
    )
    return sampler

def create_sampler_parallel(subset, dataset, num_workers=4):
    """
    병렬 처리를 사용하여 WeightedRandomSampler를 생성합니다.

    Args:
        subset (Subset): 데이터 Subset.
        dataset (Dataset): 전체 Dataset.
        num_workers (int): 병렬 처리에 사용할 워커 수.

    Returns:
        WeightedRandomSampler: 클래스 균등성을 위한 샘플러.
    """
    # 클래스별 개수 및 총 샘플 수 계산
    class_counts = dataset.class_counts  # {class_label: count}
    total_samples = sum(class_counts.values())

    # 클래스별 가중치 계산
    max_weight = 10.0
    class_weights = {cls: min(total_samples / count, max_weight) for cls, count in class_counts.items()}
    print(f"Class Weights: {class_weights}")

    def compute_sample_weight(idx):
        # 샘플의 라벨 추출
        label = dataset[idx]['label']  # 인덱스를 통해 데이터셋에서 라벨 추출
        if isinstance(label, np.ndarray):
            if label.size == 1:  # 배열 크기가 1인 경우
                label = label.item()
            else:  # 배열이 다차원인 경우 (예: 원-핫 인코딩)
                label = np.argmax(label)
        elif isinstance(label, torch.Tensor):
            label = label.argmax().item()  # Tensor -> 정수 값으로 변환
        return class_weights[label]

    # 병렬로 샘플별 가중치 생성
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        sample_weights = list(executor.map(compute_sample_weight, subset.indices))

    # WeightedRandomSampler 생성
    sample_weights = torch.tensor(sample_weights, dtype=torch.float)
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(subset),
        replacement=True
    )
    return sampler
def show_class_distribution(subset, dataset, num_workers=4):
    """
    Subset 내 클래스 분포를 출력합니다.

    Args:
        subset (Subset): 데이터 Subset
        dataset (Dataset): 전체 Dataset
        num_workers : workers
    """

    def get_label(idx):
        """
        인덱스에 해당하는 데이터의 라벨을 반환합니다.
        """
        return dataset.idx_to_label[torch.argmax(torch.from_numpy(dataset[idx]["label"])).item()]

    # 병렬로 라벨 추출
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        labels = list(executor.map(get_label, subset.indices))

    # 클래스 분포 계산
    class_distribution = Counter(labels)
    logger.info(f"Class Distribution in Subset: {class_distribution}")
    return class_distribution


def check_loader_balance(loader, dataset, num_batches=5):
    """
    데이터 로더에서 각 배치의 클래스 분포를 확인합니다.

    Args:
        loader: DataLoader 객체.
        dataset: 전체 데이터셋 객체.
        num_batches: 확인할 배치 수 (기본값: 5).
    """
    for batch_idx, batch_data in enumerate(loader):
        if batch_idx >= num_batches:
            break

        # 데이터 로더의 반환 구조에 맞게 수정
        inputs = batch_data['image']
        labels = batch_data['label']

        # 정수형 라벨로 변환 (필요할 경우)
        if len(labels.shape) > 1:  # 원-핫 벡터라면 변환
            labels = torch.argmax(labels, dim=1)

        # 클래스 이름 매핑
        label_names = [dataset.idx_to_label[label.item()] for label in labels]
        batch_distribution = Counter(label_names)

        print(f"Batch {batch_idx + 1} Class Distribution: {batch_distribution}")

def split_dataset(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, batch_size=8, num_workers=4):
    """
    데이터셋을 train/val/test 비율로 분할하고, 각각의 DataLoader를 반환합니다.

    Args:
        dataset (Dataset): 전체 데이터셋.
        train_ratio (float): 훈련 데이터 비율 (0~1).
        val_ratio (float): 검증 데이터 비율 (0~1).
        test_ratio (float): 테스트 데이터 비율 (0~1).
        batch_size (int): 각 DataLoader의 배치 크기.
        num_workers (int): DataLoader에서 사용할 워커 수.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    assert train_ratio + val_ratio + test_ratio == 1.0, "train/val/test 비율의 합은 1이어야 합니다."

    # 전체 인덱스 생성 및 섞기
    indices = list(range(len(dataset)))
    train_idx, temp_idx = train_test_split(indices, test_size=(val_ratio + test_ratio), random_state=42)
    val_idx, test_idx = train_test_split(temp_idx, test_size=(test_ratio / (val_ratio + test_ratio)), random_state=42)

    # Subset으로 데이터셋 분리
    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)
    test_set = Subset(dataset, test_idx)

    # 각 Subset의 클래스 분포 출력
    logger.info("Train Set Distribution:")
    show_class_distribution(train_set, dataset)
    logger.info("Validation Set Distribution:")
    show_class_distribution(val_set, dataset)
    logger.info("Test Set Distribution:")
    show_class_distribution(test_set, dataset)

    # Sampler 생성
    train_sampler = create_sampler(train_set, dataset)
    val_sampler = create_sampler(val_set, dataset)
    test_sampler = create_sampler(test_set, dataset)

    # DataLoader 생성
    train_loader = DataLoader(train_set, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, sampler=val_sampler, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, sampler=test_sampler, num_workers=num_workers)

    return train_loader, val_loader, test_loader

def split_dataset_parallel(dataset, train_val_idx, train_ratio=0.7, val_ratio=0.3, batch_size=8, num_workers=4, logger=None):
    """
    병렬 처리를 사용하여 Train/Validation 데이터를 분할합니다.

    Args:
        dataset (Dataset): 전체 데이터셋.
        train_val_idx (list): Train/Validation에 사용할 데이터 인덱스.
        train_ratio (float): 훈련 데이터 비율 (0~1).
        val_ratio (float): 검증 데이터 비율 (0~1).
        batch_size (int): 각 DataLoader의 배치 크기.
        num_workers (int): DataLoader에서 사용할 워커 수.
        logger (Logger): 로깅 객체.

    Returns:
        tuple: (train_loader, val_loader)
    """
    assert train_ratio + val_ratio == 1.0, "train/val 비율의 합은 1이어야 합니다."

    labels = [dataset.db_rec[idx]["label"] for idx in train_val_idx]
    # Train/Validation 인덱스 분할
    train_idx, val_idx = train_test_split(train_val_idx, test_size=val_ratio, random_state=42, stratify=labels)

    # Subset으로 데이터셋 분리
    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)

    # 각 Subset의 클래스 분포 출력
    if logger:
        logger.info("Train Set Distribution:")
        show_class_distribution(train_set, dataset)
        logger.info("Validation Set Distribution:")
        show_class_distribution(val_set, dataset)

    # Sampler 병렬 생성
    train_sampler = create_sampler_parallel(train_set, dataset, num_workers=num_workers)
    val_sampler = create_sampler_parallel(val_set, dataset, num_workers=num_workers)

    # DataLoader 생성
    train_loader = DataLoader(train_set, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, sampler=val_sampler, num_workers=num_workers)

    return train_loader, val_loader


def balance_dataset(dataset, min_ratio=1, max_ratio=1.05):
    """
    Balance the dataset by ensuring all classes have data within a certain ratio of the smallest class.

    Args:
        dataset (MedicalImageDataset): The dataset to balance.
        min_ratio (float): Minimum scaling factor for class balancing.
        max_ratio (float): Maximum scaling factor for class balancing.

    Returns:
        list: A balanced list of dataset indices.
    """
    class_counts = Counter([record['label'] for record in dataset.db_rec])
    min_class_count = min(class_counts.values())

    balanced_data = []
    for class_name, count in class_counts.items():
        class_data = [record for record in dataset.db_rec if record['label'] == class_name]
        scale_factor = random.uniform(min_ratio, max_ratio) if count > min_class_count else 1
        target_count = int(min_class_count * scale_factor)
        balanced_data.extend(random.choices(class_data, k=target_count))
    random.shuffle(balanced_data)
    dataset.db_rec = balanced_data
    
    dataset.class_counts = Counter([record['label'] for record in dataset.db_rec])
    return dataset



def balance_dataset_parallel(dataset, min_ratio=1.1, max_ratio=1.2, num_workers=4):
    """
    병렬 처리를 사용하여 데이터셋을 균형 잡힌 형태로 조정합니다.

    Args:
        dataset (MedicalImageDataset): 조정할 데이터셋.
        min_ratio (float): 최소 스케일링 비율.
        max_ratio (float): 최대 스케일링 비율.
        num_workers (int): 병렬 처리에 사용할 워커 수.

    Returns:
        dataset: 균형 잡힌 데이터셋.
    """
    class_counts = Counter([record['label'] for record in dataset.db_rec])
    min_class_count = min(class_counts.values())

    def process_class(class_name):
        class_data = [record for record in dataset.db_rec if record['label'] == class_name]
        scale_factor = random.uniform(min_ratio, max_ratio) if len(class_data) > min_class_count else 1
        target_count = int(min_class_count * scale_factor)
        return random.choices(class_data, k=target_count)

    balanced_data = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(process_class, class_counts.keys()))
        for res in results:
            balanced_data.extend(res)
    random.shuffle(balanced_data)
    dataset.data = balanced_data
    dataset.class_counts = Counter([record['label'] for record in dataset.db_rec])
    return dataset


def set_seed(seed):
    """Set the random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def oversample_data(dataset, labels, target_count):
    """
    소수 클래스 샘플을 중복하여 데이터 크기를 확장합니다.

    Args:
        dataset: 전체 데이터셋.
        labels: 각 데이터의 레이블.
        target_count: 클래스별 목표 데이터 수.

    Returns:
        oversampled_dataset, oversampled_labels
    """
    class_counts = Counter(labels)
    oversampled_data = []
    oversampled_labels = []

    for label in set(labels):
        # 해당 클래스의 데이터 추출
        class_data = [dataset[i] for i in range(len(labels)) if labels[i] == label]
        oversampled_data.extend(class_data)

        # 중복 추가
        oversampled_labels.extend([label] * len(class_data))
        while len(oversampled_labels) < target_count:
            oversampled_data.extend(random.choices(class_data, k=target_count - len(oversampled_labels)))
            oversampled_labels.extend([label] * (target_count - len(oversampled_labels)))

    return oversampled_data, oversampled_labels


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.01, warmup_epochs=50, save_path="best_model.pth"):
        """
        Args:
            patience (int): Number of epochs to wait before stopping if no improvement.
            min_delta (float): Minimum change in the monitored metric to qualify as improvement.
            save_path (str): Path to save the best model.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.save_path = save_path
        self.best_score = None
        self.warmup_epochs = warmup_epochs
        self.counter = 0
        self.early_stop = False
        self.best_epoch = 0

    def __call__(self, current_score, model, epoch):
        """
        Args:
            current_score (float): The current validation metric (e.g., validation loss).
            model (torch.nn.Module): The model to save when validation improves.
            epoch (int): The current epoch number.
        """
        if epoch < self.warmup_epochs:
            if self.best_score is None:
                self.best_score = current_score
                self.best_epoch = epoch
            else:
                if self.best_score >= current_score:
                    self.best_score = current_score
                    self.best_epoch = epoch
                    self.save_checkpoint(model)
            return False
        if self.best_score is None:
            self.best_score = current_score
            self.best_epoch = epoch
            self.save_checkpoint(model)
        elif current_score < self.best_score - self.min_delta:
            self.counter = 0
            self.best_score = current_score
            self.best_epoch = epoch
            self.save_checkpoint(model)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop

    def save_checkpoint(self, model):
        """Saves the model when validation improves."""
        torch.save(model.state_dict(), self.save_path)
        logger.info(f"Model saved at epoch {self.best_epoch + 1} with score: {self.best_score:.4f}")

class BestModelSaver:
    def __init__(self, save_path="bestmodel_EarlyStop.pth.tar"):
        """
        Args:
            save_path (str): Path to save the best model.
        """
        self.save_path = save_path
        self.best_loss = None
        self.best_epoch = 0

    def update(self, current_loss, current_accuracy, model, epoch):
        """
        Updates the best model if the current loss is lower than the previous best.

        Args:
            current_loss (float): Current validation loss.
            current_accuracy (float): Current validation accuracy.
            model (torch.nn.Module): Model to save when validation improves.
            epoch (int): Current epoch number.
        """
        if self.best_loss is None or current_loss < self.best_loss:
            self.best_loss = current_loss
            self.best_epoch = epoch
            self.save_checkpoint(model, current_loss, current_accuracy)

    def save_checkpoint(self, model, loss, accuracy):
        """Saves the model when validation improves."""
        torch.save({
            "model_state_dict": model.state_dict(),
            "loss": loss,
            "accuracy": accuracy,
            "epoch": self.best_epoch
        }, self.save_path)
        logger.info(
            f"Best model saved at epoch {self.best_epoch + 1} with loss: {loss:.4f} and accuracy: {accuracy:.4f}")

    def save_final_model(self, model, save_path="final_model.pth.tar"):
        """Saves the final model at the end of training."""
        torch.save(model.state_dict(), save_path)
        logger.info(f"Final model saved at {save_path}")