import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import os
import numpy as np
import seaborn as sns

class TrainingManager:
    def __init__(self, model_class, criterion, optimizers, train_loader, val_loader, test_loader, num_epochs=10, calc_val_loss=True, calc_train_all_loss=True):
        self.model_class = model_class
        self.criterion = criterion
        self.optimizers = optimizers
        self.num_epochs = num_epochs
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.calc_val_loss = calc_val_loss
        self.calc_train_all_loss = calc_train_all_loss
        self.loss_values = {name: {'train': [], 'val': [], 'train_all': []} for name in self.optimizers.keys()}
        self.metrics = {name: [] for name in self.optimizers.keys()}
        self.models = {name: None for name in self.optimizers.keys()}
        self.colors = sns.color_palette("colorblind", len(self.optimizers))
        self.optimizer_colors = {name: color for name, color in zip(self.optimizers.keys(), self.colors)}
    
    def train(self):
        for name, optimizer_class in self.optimizers.items():
            print(f"\nTraining with optimizer: {name}")
            model = self.model_class()  # Reinitialize the model
            optimizer = optimizer_class(model.parameters())  # Create a new optimizer instance
            self.models[name] = model
            
            for epoch in range(self.num_epochs):
                model.train()
                running_loss = 0.0
                for inputs, labels in self.train_loader:
                    # Forward pass
                    outputs = model(inputs)
                    loss = self.criterion(outputs, labels)
                    self.loss_values[name]['train'].append(loss.item())
                    
                    # Zero gradients, perform a backward pass, and update the weights
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                
                # Calculate validation loss if required
                if self.calc_val_loss:
                    model.eval()
                    val_loss = 0.0
                    with torch.no_grad():
                        for inputs, labels in self.val_loader:
                            outputs = model(inputs)
                            loss = self.criterion(outputs, labels)
                            val_loss += loss.item()
                    val_loss /= len(self.val_loader)
                    self.loss_values[name]['val'].append(val_loss)

                if self.calc_train_all_loss:
                    model.eval()
                    train_loss = 0.0
                    with torch.no_grad():
                        for inputs, labels in self.train_loader:
                            outputs = model(inputs)
                            loss = self.criterion(outputs, labels)
                            train_loss += loss.item()
                    train_loss /= len(self.train_loader)
                    self.loss_values[name]['train_all'].append(train_loss)

                if self.calc_val_loss:
                    print(f'Epoch [{epoch + 1}/{self.num_epochs}], Train Loss: {running_loss/len(self.train_loader):.4f}, Val Loss: {val_loss:.4f}')
                else:
                    print(f'Epoch [{epoch + 1}/{self.num_epochs}], Train Loss: {running_loss/len(self.train_loader):.4f}')
    
    def plot_loss(self, save_path):
        plt.figure(figsize=(20, 10))
        for name, values in self.loss_values.items():
            linewidth = 5 if name == 'AdaSecant' else 2
            plt.plot(range(len(values['train'])), np.log(values['train']), label=f'{name} Training log(Loss)', linewidth=linewidth, color=self.optimizer_colors[name])
        plt.xlabel('Iterations', fontsize=16)
        plt.ylabel('log(Loss)', fontsize=16)
        plt.title('log(Loss) over Iterations', fontsize=20)
        plt.legend(fontsize=14)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.savefig(save_path+ '/train_loss.png')
        plt.show()

    def plot_train_val_loss(self, save_path):
        plt.figure(figsize=(20, 10))
        for name, values in self.loss_values.items():
            linewidth = 5 if name == 'AdaSecant' else 2
            if self.calc_train_all_loss:
                plt.plot(range(len(values['train_all'])), np.log(values['train_all']), label=f'{name} Train log(Loss)', linewidth=linewidth, linestyle="dashed", color=self.optimizer_colors[name])
            if self.calc_val_loss:
                plt.plot(range(len(values['val'])), np.log(values['val']), label=f'{name} Validation log(Loss)', linewidth=linewidth, color=self.optimizer_colors[name])
        plt.xlabel('Epochs', fontsize=16)
        plt.ylabel('log(Loss)', fontsize=16)
        plt.title('Training and Validation log(Loss) over Iterations', fontsize=20)
        plt.legend(fontsize=14)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.savefig(save_path +  '/train_val_loss.png')
        plt.show()
    
    def calculate_metrics(self, save_path):
        metrics = {}
        for name, model in self.models.items():
            metric = model.evaluate(self.test_loader)
            metrics[name] = metric
            print(f'Metric of the model with {name} optimizer on the test set: {metric:.2f}')
        with open(save_path + '/metrics.txt', 'w') as f:
            for name, metric in metrics.items():
                f.write(f'{name}: {metric:.2f}\n')
        return metrics
