# -------------------- Imports --------------------
import os, zipfile
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch.amp import autocast, GradScaler
import numpy as np

# -------------------- Config --------------------
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 32
NUM_EPOCHS = 45
LR = 1e-4
WEIGHT_DECAY = 1e-4
FORBIDDEN_IDS = [95, 145, 146, 158, 159, 160, 161]

# -------------------- Paths --------------------
zip_path = '/content/dataset.zip'
extract_dir = '/content/data'

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

train_csv = os.path.join(extract_dir, 'labels_train.csv')
val_csv = os.path.join(extract_dir, 'labels_val.csv')
train_img_dir = os.path.join(extract_dir, 'images_train', 'images_train')
val_img_dir = os.path.join(extract_dir, 'images_val')

# -------------------- Data --------------------
df_train = pd.read_csv(train_csv)
df_val = pd.read_csv(val_csv)

lat_mean = df_train['latitude'].mean()
lat_std = df_train['latitude'].std()
long_mean = df_train['longitude'].mean()
long_std = df_train['longitude'].std()

def normalize_latlong(lat, long):
    return (lat - lat_mean) / lat_std, (long - long_mean) / long_std

def denormalize_latlong(lat, long):
    return lat * lat_std + lat_mean, long * long_std + long_mean

# -------------------- Dataset --------------------
class LocationDataset(Dataset):
    def __init__(self, df, img_dir, transform):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(os.path.join(self.img_dir, row['filename'])).convert('RGB')
        image = self.transform(image)

        region_id = int(row['Region_ID']) - 1
        lat, long = normalize_latlong(row['latitude'], row['longitude'])

        return image, torch.tensor(region_id), torch.tensor([lat, long], dtype=torch.float)

    def __len__(self):
        return len(self.df)

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
])

train_dataset = LocationDataset(df_train, train_img_dir, transform)
val_dataset = LocationDataset(df_val, val_img_dir, transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
]))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# -------------------- FiLM --------------------
class FiLM(nn.Module):
    def __init__(self, embedding_dim, feature_dim):
        super().__init__()
        self.gamma = nn.Linear(embedding_dim, feature_dim)
        self.beta = nn.Linear(embedding_dim, feature_dim)

    def forward(self, x, embedding):
        gamma = self.gamma(embedding).unsqueeze(2).unsqueeze(3)
        beta = self.beta(embedding).unsqueeze(2).unsqueeze(3)
        return gamma * x + beta

# -------------------- Model --------------------
class CombinedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.convnext_tiny(pretrained=True, stochastic_depth_prob=0.2)
        self.backbone.classifier = nn.Identity()
        self.region_embed = nn.Embedding(15, 128)

        self.film = FiLM(embedding_dim=128, feature_dim=768)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(768 + 128, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )

    def forward(self, x, region_ids):
        region_feat = self.region_embed(region_ids)
        feat = self.backbone.features(x)
        feat = self.film(feat, region_feat)

        pooled = self.pool(feat).view(feat.size(0), -1)
        fused = torch.cat([pooled, region_feat], dim=1)
        return self.fc(fused)

# -------------------- Metric --------------------
def compute_average_mse(preds, targets, lat_mean, lat_std, long_mean, long_std):
    preds_denorm = preds.clone()
    preds_denorm[:, 0] = preds[:, 0] * lat_std + lat_mean
    preds_denorm[:, 1] = preds[:, 1] * long_std + long_mean

    targets_denorm = targets.clone()
    targets_denorm[:, 0] = targets[:, 0] * lat_std + lat_mean
    targets_denorm[:, 1] = targets[:, 1] * long_std + long_mean

    mse_lat = nn.functional.mse_loss(preds_denorm[:, 0], targets_denorm[:, 0])
    mse_long = nn.functional.mse_loss(preds_denorm[:, 1], targets_denorm[:, 1])

    return 0.5 * (mse_lat + mse_long)

# -------------------- Training --------------------
def train(model, train_loader, val_loader):
    model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    criterion = nn.HuberLoss(delta=5.0)
    scaler = GradScaler()

    best_val_loss = float('inf')
    best_model_state = None  # Store best model state in memory
    patience_counter = 0
    patience = 10

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0
        for img, region_ids, target in tqdm(train_loader):
            img, region_ids, target = img.to(DEVICE), region_ids.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            with autocast(device_type='cuda'):
                output = model(img, region_ids)
                loss = criterion(output, target)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        
        # Validation Phase
        model.eval()
        val_loss = 0
        avg_mse_total = 0
        with torch.no_grad():
            for img, region_ids, target in val_loader:
                img, region_ids, target = img.to(DEVICE), region_ids.to(DEVICE), target.to(DEVICE)
                output = model(img, region_ids)
                loss = criterion(output, target)
                val_loss += loss.item()

                avg_mse_total += compute_average_mse(output.cpu(), target.cpu(), lat_mean, lat_std, long_mean, long_std).item()

        val_loss /= len(val_loader)
        avg_mse_total /= len(val_loader)

        print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Real Avg MSE: {avg_mse_total:.2f}")

        # Scheduler step
        scheduler.step(val_loss)

        # Early Stopping & Best Model Check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()  # Save to memory, not disk
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    # Save the best model state after training is complete
    if best_model_state is not None:
        torch.save(best_model_state, 'best_model_latlong.pt')
        print("Best model saved after training.")

    return model


# -------------------- Train --------------------
model = CombinedModel()
model = train(model, train_loader, val_loader)

# -------------------- Predict --------------------
model.load_state_dict(torch.load('best_model_latlong.pt'))
model.eval()

preds = []
with torch.no_grad():
    for img, region_ids, _ in DataLoader(val_dataset, batch_size=32):
        img = img.to(DEVICE)
        region_ids = region_ids.to(DEVICE)
        out = model(img, region_ids).cpu().numpy()
        out[:, 0], out[:, 1] = denormalize_latlong(out[:, 0], out[:, 1])
        preds.extend(np.round(out).astype(int).tolist())

# -------------------- Save Final CSV --------------------
submission = []
val_ids = list(range(370))
pred_map = {i: pred for i, pred in zip(val_ids, preds) if i not in FORBIDDEN_IDS}

for i in range(738):
    if i in FORBIDDEN_IDS:
        continue
    if i in pred_map:
        submission.append([i, pred_map[i][0], pred_map[i][1]])
    else:
        submission.append([i, 0, 0])

submission_df = pd.DataFrame(submission, columns=['id', 'Latitude', 'Longitude'])
submission_df.to_csv('latlong_predictions.csv', index=False)
print("Saved latlong_predictions.csv with 732 rows.")
