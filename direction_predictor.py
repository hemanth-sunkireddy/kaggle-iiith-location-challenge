import os
import pandas as pd
import numpy as np
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error

from google.colab import drive
drive.mount('/content/drive')
print("📂 Google Drive mounted.")

# Set base directory
base_dir = '/content/drive/MyDrive/SMAI_Project'
print(f"📁 Base directory set to: {base_dir}")

# -------------------------
# Dataset
# -------------------------
class AngleDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
        print(f"📦 Dataset initialized with {len(self.df)} samples from {img_dir}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['filename'])
        image = Image.open(img_path).convert('RGB')
        angle = row['angle']

        if self.transform:
            image = self.transform(image)

        return image, angle

# -------------------------
# Transforms
# -------------------------
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])
print("✅ Transformations defined.")

# -------------------------
# Load Data
# -------------------------
train_df = pd.read_csv(os.path.join(base_dir, 'angles_train.csv'))
val_df = pd.read_csv(os.path.join(base_dir, 'angles_val.csv'))

train_dataset = AngleDataset(train_df, os.path.join(base_dir, 'images_train'), transform)
val_dataset = AngleDataset(val_df, os.path.join(base_dir, 'images_val'), transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)
print("📊 Data loaded.")

# -------------------------
# Model Setup
# -------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"💻 Using device: {device}")

model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 1)  # Regression output
model = model.to(device)
print("🧠 Model initialized for regression.")

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
print("⚙️ Loss and optimizer set.")

# -------------------------
# Training
# -------------------------
epochs = 10
print("🚦 Starting training...")

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, angles in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
        images = images.to(device)
        angles = angles.float().unsqueeze(1).to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, angles)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"📚 Epoch {epoch+1} - Loss: {running_loss / len(train_loader):.4f}")

# -------------------------
# Evaluation
# -------------------------
model.eval()
preds = []
gts = []

with torch.no_grad():
    for images, angles in tqdm(val_loader, desc="🔍 Evaluating"):
        images = images.to(device)
        angles = angles.numpy()
        outputs = model(images).cpu().numpy().flatten()
        preds.extend(outputs)
        gts.extend(angles)

# Compute MAAE
def angular_error(true, pred):
    return np.mean(np.minimum(np.abs(np.array(true) - np.array(pred)),
                              360 - np.abs(np.array(true) - np.array(pred))))

maae = angular_error(gts, preds)
print(f"🎯 Validation MAAE: {maae:.2f} degrees")

# -------------------------
# Submission
# -------------------------
print("📝 Creating submission file...")

val_ids = list(range(369))
val_angles = [round(max(0, min(360, p))) for p in preds]  # Clamp to [0, 360]
val_submission = pd.DataFrame({
    'id': val_ids,
    'angle': val_angles
})
print("✅ Validation predictions added.")

test_submission = pd.DataFrame({
    'id': list(range(369, 738)),
    'angle': [0] * 369
})
print("✅ Test samples filled with angle = 0.")

final_submission = pd.concat([val_submission, test_submission], ignore_index=True)
submission_path = os.path.join(base_dir, '2022101005_1.csv')
final_submission.to_csv(submission_path, index=False)
print(f"📁 Submission file saved to {submission_path}")
print("✅ All done!")
