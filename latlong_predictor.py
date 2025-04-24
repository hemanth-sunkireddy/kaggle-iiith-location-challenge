# ⛰️ Colab and Environment Setup
from google.colab import drive
drive.mount('/content/drive')

import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

# 📁 Define paths inside your Google Drive
root_dir = '/content/drive/MyDrive/SMAI_Project'
train_csv = os.path.join(root_dir, 'labels_train.csv')
val_csv = os.path.join(root_dir, 'labels_val.csv')
train_img_dir = os.path.join(root_dir, 'images_train')
val_img_dir = os.path.join(root_dir, 'images_val')

# 🧾 Read the CSVs
train_df = pd.read_csv(train_csv)
val_df = pd.read_csv(val_csv)

# ❌ Remove anomaly IDs from val
anomaly_ids = [95, 145, 146, 158, 159, 160, 161]
val_df_cleaned = val_df[~val_df.index.isin(anomaly_ids)].reset_index(drop=True)

# 🖼️ Custom Dataset
class GeoDataset(Dataset):
    def __init__(self, df, img_dir, transform=None, task='both'):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
        self.task = task

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['filename'])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        lat = int(row['Latitude'])
        lon = int(row['Longitude'])
        return image, torch.tensor([lat, lon], dtype=torch.float)

# 📦 Transforms
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# 🔄 Dataloaders
train_dataset = GeoDataset(train_df, train_img_dir, transform)
val_dataset = GeoDataset(val_df_cleaned, val_img_dir, transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# 🧠 Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)

# ⚙️ Loss & Optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# 🏋️‍♂️ Training
epochs = 10
for epoch in range(epochs):
    model.train()
    running_loss = 0
    for images, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    print(f"📚 Epoch {epoch+1} Loss: {running_loss / len(train_loader):.4f}")

# 📈 Validation
model.eval()
val_preds = []
with torch.no_grad():
    for images, _ in val_loader:
        images = images.to(device)
        outputs = model(images)
        preds = outputs.round().cpu().numpy().astype(int)
        val_preds.extend(preds)

# 📤 Submission file
print("📝 Generating submission...")
val_ids = val_df_cleaned.index.tolist()
val_submission = pd.DataFrame({
    'id': val_ids,
    'Latitude': [lat for lat, lon in val_preds],
    'Longitude': [lon for lat, lon in val_preds]
})

# Add 0,0 for test samples
test_ids = list(range(369, 738))
test_submission = pd.DataFrame({
    'id': test_ids,
    'Latitude': [0] * len(test_ids),
    'Longitude': [0] * len(test_ids)
})

# Combine
submission_df = pd.concat([val_submission, test_submission], ignore_index=True)
submission_df = submission_df.sort_values(by='id').reset_index(drop=True)
submission_df.to_csv('2022101005_1.csv', index=False)
print("✅ Submission file '2022101005_1.csv' created!")
