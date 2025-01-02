import json
import os
import logging
import torch
from sklearn.model_selection import KFold, StratifiedKFold, LeavePOut
from torch.utils.data import Subset, DataLoader
from statistics import mean, stdev
from .trainer import Trainer
from data.dataloader import OversampledDataset
from utils.tools import EarlyStopping, show_class_distribution, check_loader_balance, create_sampler_parallel, BestModelSaver
from torch.utils.tensorboard import SummaryWriter


def compute_summary(val_accuracies, test_accuracies=None):
    summary = {
        "mean_val_acc": mean(val_accuracies),
        "std_val_acc": stdev(val_accuracies)
    }
    if test_accuracies:
        summary.update({
            "mean_test_acc": mean(test_accuracies),
            "std_test_acc": stdev(test_accuracies)
        })
    return summary

def run_lpo_training(cfg, dataset, model, p=2, seed=42, final_output_dir=None, output_ddir=None, writer_dict=None):
    """
    Perform Leave-p-out (LPO) training.

    Args:
        cfg: Configuration object containing training settings.
        dataset: Dataset object to be used for training and validation.
        model: PyTorch model to train.
        p (int): Number of samples to leave out for validation.
        seed (int): Random seed for reproducibility.
        final_output_dir: Directory to save final outputs.
        output_ddir: Directory to save intermediate outputs.
        writer_dict: Dictionary for managing logging.

    Returns:
        model: Trained PyTorch model.
    """
    torch.manual_seed(seed)

    # Initialize LeavePOut splitter
    lpo = LeavePOut(p)
    dataset_size = len(dataset)
    indices = list(range(dataset_size))

    # Iterate over LPO splits
    for train_indices, val_indices in lpo.split(indices):
        train_subset = Subset(dataset, train_indices)
        val_subset = Subset(dataset, val_indices)

        train_loader = DataLoader(train_subset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=False)

        # Initialize optimizer, loss function, etc.
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.TRAIN.LR)
        criterion = torch.nn.CrossEntropyLoss()

        # Training loop
        model.train()
        for epoch in range(cfg.TRAIN.EPOCHS):
            total_loss = 0.0

            for batch in train_loader:
                inputs, labels = batch
                inputs, labels = inputs.to(cfg.DEVICE), labels.to(cfg.DEVICE)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            print(f"Epoch {epoch + 1}/{cfg.TRAIN.EPOCHS}, Loss: {total_loss / len(train_loader)}")

        # Validation loop
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                inputs, labels = batch
                inputs, labels = inputs.to(cfg.DEVICE), labels.to(cfg.DEVICE)

                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        print(f"Validation Accuracy for split: {accuracy:.4f}")

        # Save model checkpoint if needed
        if final_output_dir:
            checkpoint_path = f"{final_output_dir}/model_split_{train_indices[0]}.pth"
            torch.save(model.state_dict(), checkpoint_path)

    print("LPO training complete.")
    return model



def run_kfold_training(cfg, dataset, model, optimizer, scheduler, num_folds=5, p = None, 
                       seed=42, final_output_dir=None, output_dir="./kfold_results",
                       train_val_indices=None, test_loader=None, writer_dict=None):
    """
    Perform K-Fold cross-validation training and evaluation using samplers.

    Args:
        cfg: Configuration file.
        dataset: Full dataset.
        model: PyTorch model.
        optimizer: Optimizer.
        scheduler: Learning rate scheduler.
        num_folds: Number of K-Folds (default: 5).
        seed: Random seed for reproducibility (default: 42).
        output_dir: Directory to save results.
        train_val_indices: Indices for training and validation.
        test_loader: Fixed DataLoader for testing.

    Returns:
        None
    """
    logger = logging.getLogger("K_FOLD TRAINING")

    assert train_val_indices is not None , "train_val_indices와 고정된 test_loader가 필요합니다."

    torch.manual_seed(seed)

    # Prepare results storage
    fold_results = []
    labels = [dataset.db_rec[idx]["label"] for idx in train_val_indices]
    num_samples = len(labels)
    print(num_samples)

    # kfold = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
    if p > 0:
        logger.info(f"num_samples // p : {num_samples // p } kfold")
        skfold = StratifiedKFold(n_splits=num_samples // p, shuffle=True, random_state=seed)
        num_folds = num_samples // p
    else:
        logger.info("original kfold")
        skfold = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)




    # for fold, (train_indices, val_indices) in enumerate(kfold.split(train_val_indices)):
    for fold, (train_indices, val_indices) in enumerate(skfold.split(train_val_indices, labels)):
        assert len(set(train_indices).intersection(set(val_indices))) == 0, "Train and validation indices overlap!"
        logger.info(f"Starting Fold {fold + 1}/{num_folds}...")
        fold_output_dir = f"{output_dir}/fold_{fold + 1}"
        os.makedirs(fold_output_dir, exist_ok=True)

        # Train/Validation splits
        train_set = Subset(dataset, train_indices)
        val_set = Subset(dataset, val_indices)
        show_class_distribution(train_set, dataset, 4)
        show_class_distribution(val_set, dataset, 4)
        # train_labels = [labels[i] for i in train_indices]

        # OverSampling
        # oversampled_train_set = OversampledDataset(train_set, target_count=240)
        # show_class_distribution(oversampled_train_set, dataset, 4)

        # 샘플러 생성
        train_sampler = create_sampler_parallel(train_set, dataset)
        val_sampler = create_sampler_parallel(val_set, dataset)

        # DataLoaders with samplers
        # train_loader = DataLoader(
        #     train_set, batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU, num_workers=cfg.WORKERS
        # )
        # sampler
        train_loader = DataLoader(
            train_set, batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU, num_workers=cfg.WORKERS, sampler=train_sampler, pin_memory=True,
        )
        val_loader = DataLoader(
            val_set, batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU, num_workers=cfg.WORKERS, sampler=val_sampler, pin_memory=True,
        )

        check_loader_balance(train_loader, dataset, 32)
        check_loader_balance(val_loader, dataset, 16)

        best_val_acc = -1
        best_pert = float('inf')
        patience = 10
        counter = 0
        best_model_path = f"{fold_output_dir}/best_model.pth.tar"
        final_model_path = f"{fold_output_dir}/final_model.pth.tar"

        # Optimizer와 Scheduler 초기화
        # optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        # early_stopping = EarlyStopping(patience=10, min_delta=0.1, warmup_epochs=20)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
        #                                                        mode='min',
        #                                                        factor=0.5,
        #                                                        patience=cfg.TRAIN.PATIENCE,
        #                                                        verbose=True)
        model.initialize_model()
        model = model.to("cuda")

        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.TRAIN.LR, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        best_model_saver = BestModelSaver(save_path=os.path.join(final_output_dir, 'ckpt', f"bestmodel_EarlyStop_{fold}_{num_folds}.pth.tar"))


        writer_dict = {'writer': SummaryWriter(log_dir=os.path.join(final_output_dir, 'tensorboard', 'kfold', f'kfold_{fold}')),
                       "train_global_steps": 0,
                       "valid_global_steps": 0
                       }

        trainer = Trainer(cfg, model, output_dir, writer_dict=writer_dict)

        # Train and validate
        for epoch in range(cfg.TRAIN.BEGIN_EPOCH, cfg.TRAIN.END_EPOCH):
            trainer.train(epoch, train_loader, optimizer, scheduler)
            # scheduler.step()
            val_acc, val_loss = trainer.validate(epoch, val_loader)

            # if val_acc > best_val_acc:
            #     best_val_acc = val_acc
            #     torch.save(model.state_dict(), best_model_path)
            #     logger.info(f"Fold {fold + 1}, Epoch {epoch + 1}: New best validation accuracy: {best_val_acc:.4f}")
            best_model_saver.update(val_loss, val_acc, model, epoch)
            # if early_stopping(val_loss, model, epoch):
            #     logger.info(f"Early stopping triggered at epoch {epoch + 1}")
            
            
            
        # save final_model_ckpt
        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_perf': val_loss}, final_model_path)    
            
        fold_results.append({
            "fold": fold+1,
            "best_val_acc" : best_val_acc
        })
        
        if test_loader is not None:
            # Test with the best model for this fold
            model.load_state_dict(torch.load(best_model_path))
            test_acc, _ = trainer.validate(0, test_loader)
            logger.info(f"Fold {fold + 1}: Test accuracy: {test_acc:.4f}")

            # Save results for this fold
            fold_results[-1].update({'test_acc': test_acc})

    # Summary statistics
    val_accuracies = [result["best_val_acc"] for result in fold_results]
    if test_loader:
        test_accuracies = [result["test_acc"] for result in fold_results]
        summary = {
            "mean_val_acc": mean(val_accuracies),
            "std_val_acc": stdev(val_accuracies),
            "mean_test_acc": mean(test_accuracies),
            "std_test_acc": stdev(test_accuracies)
        }
    else:
        summary = {
            "mean_val_acc": mean(val_accuracies),
            "std_val_acc": stdev(val_accuracies),
        }
    logger.info("K-Fold Results Summary:")
    logger.info(summary)

    # Save results to JSON
    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, "kfold_results.json")
    with open(results_path, "w") as f:
        json.dump({"folds": fold_results, "summary": summary}, f, indent=4)

    print(f"Results saved to {results_path}")

