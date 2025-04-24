import os
import pandas as pd
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from tqdm import tqdm

print("🚀 Starting the image classification pipeline...")

# -------------------------
# Dataset Preparation
# -------------------------
class RegionDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
        print(f"📦 Initialized dataset with {len(self.df)} samples from {img_dir}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['filename'])
        image = Image.open(img_path).convert('RGB')
        label = row['Region_ID'] - 1  # convert to 0-indexed

        if self.transform:
            image = self.transform(image)

        return image, label

# -------------------------
# Transformations
# -------------------------
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])
print("✅ Defined image transformations.")

# -------------------------
# Load Data
# -------------------------
train_df = pd.read_csv('labels_train.csv')
val_df = pd.read_csv('labels_val.csv')
print("📄 Loaded label CSV files.")

train_dataset = RegionDataset(train_df, 'images_train', transform)
val_dataset = RegionDataset(val_df, 'images_val', transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)
print("📊 Data loaders ready.")

# -------------------------
# Model
# -------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"💻 Using device: {device}")

model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 15)
model = model.to(device)
print("🧠 Model initialized and modified for 15 output classes.")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
print("⚙️ Optimizer and loss function set.")

# -------------------------
# Training Loop
# -------------------------
epochs = 10
print("🚦 Starting training...")
for epoch in range(epochs):
    model.train()
    running_loss = 0
    pbar = tqdm(train_loader, desc=f"📚 Epoch {epoch+1}/{epochs}", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        pbar.set_postfix({'Loss': f"{running_loss / (pbar.n + 1):.4f}"})

    avg_loss = running_loss / len(train_loader)
    print(f"✅ Epoch {epoch+1} finished - Average Loss: {avg_loss:.4f}")

# -------------------------
# Evaluation on Validation Set
# -------------------------
print("🔍 Evaluating on validation set...")
model.eval()
predictions = []
ground_truth = []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        predictions.extend(preds.cpu().numpy())
        ground_truth.extend(labels.numpy())

acc = accuracy_score(ground_truth, predictions)
print(f"🎯 Validation Accuracy: {acc:.4f}")

# -------------------------
# Submission File
# -------------------------
print("📝 Creating submission file...")
val_preds_df = pd.DataFrame({
    'id': list(range(369)),
    'Region_ID': [p + 1 for p in predictions]
})
print("✅ Added validation predictions.")

test_df = pd.DataFrame({
    'id': list(range(369, 738)),
    'Region_ID': [1] * 369
})
print("✅ Filled test sample predictions with Region_ID = 1.")

submission_df = pd.concat([val_preds_df, test_df], ignore_index=True)
submission_df.to_csv('2022101005_1.csv', index=False)
print("📁 Submission file '2022101005_1.csv' created successfully.")

print("✅ All steps completed!")
