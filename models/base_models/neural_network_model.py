"""Neural Network Model for Tennis Prediction.

Implements deep neural network with:
- Multi-layer perceptron architecture
- Dropout regularization
- Batch normalization
- Early stopping
- Learning rate scheduling
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import logging
from datetime import datetime

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .base_model import BaseModel
from config import get_config


class TennisNeuralNetwork(nn.Module):
    """Neural network architecture for tennis prediction."""
    
    def __init__(self, input_size: int, hidden_layers: List[int] = [128, 64, 32], 
                 dropout_rate: float = 0.3):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        
        # Build network layers
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_layers:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, 2))  # Binary classification
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)


class NeuralNetworkModel(BaseModel):
    """Deep Neural Network Model for Tennis Prediction."""
    
    def __init__(self, use_scaling: bool = True):
        super().__init__()
        
        self.model_name = "NeuralNetwork"
        self.config = get_config()
        self.logger = logging.getLogger(f"model.{self.model_name}")
        
        # Configuration
        self.use_scaling = use_scaling
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if TORCH_AVAILABLE else None
        
        # Model components
        self.model = None
        self.scaler = StandardScaler() if use_scaling else None
        self.training_history = []
        
        # Training parameters
        self.hidden_layers = [128, 64, 32]
        self.learning_rate = 0.001
        self.epochs = 1000
        self.batch_size = 32
        self.dropout_rate = 0.3
        self.early_stopping_patience = 50
        
        if not TORCH_AVAILABLE:
            self.logger.warning("PyTorch not available, using fallback implementation")
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'NeuralNetworkModel':
        """Train the neural network."""
        
        self.logger.info(f"Training NeuralNetwork on {len(X)} samples")
        start_time = datetime.now()
        
        if not TORCH_AVAILABLE:
            # Fallback to simple logistic regression
            from sklearn.linear_model import LogisticRegression
            self.model = LogisticRegression(random_state=42)
            
            if self.use_scaling:
                X_scaled = self.scaler.fit_transform(X)
                self.model.fit(X_scaled, y)
            else:
                self.model.fit(X, y)
            
            train_accuracy = accuracy_score(y, self.predict(X))
            
        else:
            # Full PyTorch implementation
            train_accuracy = self._train_pytorch_model(X, y)
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        self.training_metrics = {
            'accuracy': train_accuracy,
            'training_time_seconds': training_time,
            'epochs_trained': len(self.training_history),
            'device_used': str(self.device) if self.device else 'cpu'
        }
        
        self.is_trained = True
        self.logger.info(f"NeuralNetwork training completed - Accuracy: {train_accuracy:.3f}")
        
        return self
    
    def _train_pytorch_model(self, X: pd.DataFrame, y: pd.Series) -> float:
        """Train PyTorch neural network."""
        
        # Prepare data
        if self.use_scaling:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = X.values
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        y_tensor = torch.LongTensor(y.values).to(self.device)
        
        # Train/validation split
        X_train, X_val, y_train, y_val = train_test_split(
            X_tensor, y_tensor, test_size=0.2, random_state=42, stratify=y_tensor.cpu()
        )
        
        # Create data loaders
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size)
        
        # Initialize model
        self.model = TennisNeuralNetwork(
            input_size=X.shape[1],
            hidden_layers=self.hidden_layers,
            dropout_rate=self.dropout_rate
        ).to(self.device)
        
        # Loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20)
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
                    
                    _, predicted = torch.max(outputs.data, 1)
                    total += batch_y.size(0)
                    correct += (predicted == batch_y).sum().item()
            
            val_accuracy = correct / total
            scheduler.step(val_loss)
            
            # Record training history
            self.training_history.append({
                'epoch': epoch,
                'train_loss': train_loss / len(train_loader),
                'val_loss': val_loss / len(val_loader),
                'val_accuracy': val_accuracy
            })
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= self.early_stopping_patience:
                self.logger.info(f"Early stopping at epoch {epoch}")
                break
        
        # Calculate final training accuracy
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs, 1)
            train_accuracy = (predicted == y_tensor).float().mean().item()
        
        return train_accuracy
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        if not TORCH_AVAILABLE:
            # Fallback prediction
            if self.use_scaling:
                X_scaled = self.scaler.transform(X)
                return self.model.predict(X_scaled)
            else:
                return self.model.predict(X)
        
        # PyTorch prediction
        if self.use_scaling:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X.values
        
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs, 1)
            
        return predicted.cpu().numpy()
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        if not TORCH_AVAILABLE:
            # Fallback prediction
            if self.use_scaling:
                X_scaled = self.scaler.transform(X)
                return self.model.predict_proba(X_scaled)
            else:
                return self.model.predict_proba(X)
        
        # PyTorch prediction
        if self.use_scaling:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X.values
        
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            
        return probabilities.cpu().numpy()