import torch
import numpy as np
import random
from torch.utils.data import Subset, DataLoader
from sklearn.model_selection import KFold
from torch.optim.lr_scheduler import StepLR
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def cross_validate(model_class, criterion, optimizer_class, param_grid, train_dataset,epochs=5, num_folds=3):
    kf = KFold(n_splits=num_folds)
    best_params = None
    best_loss = float('inf')
    
    for param in tqdm(param_grid, desc='Parameter Grid'):
        avg_val_loss = 0
        for train_idx, val_idx in kf.split(train_dataset):
            train_subset = Subset(train_dataset, train_idx)
            val_subset = Subset(train_dataset, val_idx)
            train_loader = DataLoader(train_subset, batch_size=128, shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=128, shuffle=False)

            model = model_class()  # Ensure model is instantiated correctly
            optimizer = optimizer_class(model.parameters(), **param)
            scheduler = StepLR(optimizer, step_size=1, gamma=0.7)

            for epoch in tqdm(range(epochs), desc=f'Cross-Validation for params {param}', leave=False):
                model.train()
                for inputs, labels in train_loader:
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                scheduler.step()

            val_loss = model.evaluate(val_loader)
            avg_val_loss += val_loss

        avg_val_loss /= num_folds
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_params = param
    
    return best_params

def print_best_params(optimizer_name, best_params):
    print(f"Best parameters for {optimizer_name}: {best_params}")


def plot_confusion_matrix(cm, classes, title='Confusion Matrix', save_path=None):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    if save_path:
        plt.savefig(save_path)
    plt.show()

def generate_confusion_matrices(trained_models, test_loader, classes, experiment_path):
    for name, model in trained_models.items():
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        cm = confusion_matrix(all_labels, all_preds)
        save_path = os.path.join(experiment_path, f'confusion_matrix_{name}.png')
        plot_confusion_matrix(cm, classes, title=f'Confusion Matrix for {name}', save_path=save_path)

