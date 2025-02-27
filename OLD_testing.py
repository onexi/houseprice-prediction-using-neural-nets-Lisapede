import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error
from torch.utils.data import TensorDataset, DataLoader

# -----------------------------
# 1. Load and Preprocess the Data
# -----------------------------
data = pd.read_csv('train.csv')

# Handle missing values more effectively
data.fillna(data.median(numeric_only=True), inplace=True)  # Use median instead of 0

# One-hot encode categorical features
data = pd.get_dummies(data, drop_first=True)

# -----------------------------
# 2. Feature Selection
# -----------------------------
corr_matrix = data.corr()
target_corr = corr_matrix['SalePrice'].drop('SalePrice').abs().sort_values(ascending=False)

# Select the top 10 most correlated features instead of just 4
top_features = target_corr.head(10).index
print("Selected top features:", list(top_features))

# Define input features and target variable
X = data[top_features].values
y = np.log1p(data['SalePrice'].values.reshape(-1, 1))  # Log transform target

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Normalize target variable
target_scaler = MinMaxScaler()
y_scaled = target_scaler.fit_transform(y)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Create DataLoaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)  # Increase batch size

# -----------------------------
# 3. Define Optimized Neural Network Model
# -----------------------------
class HousePriceModel(nn.Module):
    def __init__(self, input_dim):
        super(HousePriceModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.fc4 = nn.Linear(32, 1)
        self.leaky_relu = nn.LeakyReLU(0.1)  # Better than ReLU
        self.dropout = nn.Dropout(0.3)  # Increase dropout for regularization

    def forward(self, x):
        x = self.leaky_relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.leaky_relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.leaky_relu(self.bn3(self.fc3(x)))
        x = self.fc4(x)
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = HousePriceModel(input_dim=X_train.shape[1]).to(device)
print(model)

# -----------------------------
# 4. Loss Function and Optimizer
# -----------------------------
def hybrid_loss(y_pred, y_true):
    mse_loss = nn.MSELoss()(y_pred, y_true)
    mae_loss = nn.L1Loss()(y_pred, y_true)
    return mse_loss + mae_loss  # Combining MSE & MAE for stability

criterion = hybrid_loss
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)  # Lower learning rate
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=10)  # Reduce LR when stuck

# -----------------------------
# 5. Train the Model with Early Stopping
# -----------------------------
num_epochs = 1000
early_stopping_threshold = 20
best_loss = float('inf')
patience = 0

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * batch_X.size(0)

    epoch_loss = running_loss / len(train_dataset)
    
    # Early stopping condition
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        patience = 0
    else:
        patience += 1

    scheduler.step(epoch_loss)

    if patience >= early_stopping_threshold:
        print(f"Early stopping at epoch {epoch+1}")
        break

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

# -----------------------------
# 6. Evaluate the Model
# -----------------------------
model.eval()
with torch.no_grad():
    predictions = model(X_test_tensor.to(device))
    test_loss = criterion(predictions, y_test_tensor.to(device)).item()
    print("Test Mean Squared Error:", test_loss)

# Convert back from log scale
predictions_np = target_scaler.inverse_transform(predictions.cpu().numpy())

mse = mean_squared_error(np.expm1(y_test), predictions_np)  # Reverse log transform
print("Test MSE (scikit-learn):", mse)

# -----------------------------
# 7. Prepare Test Data & Make Predictions
# -----------------------------
test_df = pd.read_csv('test.csv')
test_df.fillna(test_df.median(numeric_only=True), inplace=True)
test_df = pd.get_dummies(test_df, drop_first=True)

# Ensure test data has the same columns as training data
missing_cols = set(data.columns) - set(test_df.columns)
for col in missing_cols:
    test_df[col] = 0
test_df = test_df[top_features]  # Use only selected features

# Standardize test data
X_test_scaled = scaler.transform(test_df.values)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)

# Make predictions
model.eval()
with torch.no_grad():
    predictions = model(X_test_tensor)
    predictions_np = target_scaler.inverse_transform(predictions.cpu().numpy())

# Save predictions
predictions_df = pd.DataFrame({'Id': test_df.index, 'SalePrice': predictions_np.flatten()})
predictions_df.to_csv('predictions.csv', index=False)
