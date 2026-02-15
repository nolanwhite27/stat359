print("\n========== Loading Dataset ==========")
from datasets import load_dataset

dataset = load_dataset('financial_phrasebank', 'sentences_50agree', trust_remote_code=True)
print("Dataset loaded. Example:", dataset['train'][:5])

import random
import numpy as np
import pandas as pd
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import gensim.downloader as api

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# define random seed
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

df = dataset["train"].to_pandas()

print(df.head())
print(df["label"].value_counts())

# create test dataset
train_val_df, test_df = train_test_split(
    df,
    test_size=0.15,
    stratify=df["label"],
    random_state=42
)

# create train and validation sets from the remaining data
train_df, val_df = train_test_split(
    train_val_df,
    test_size=0.15,
    stratify=train_val_df["label"],
    random_state=42
)

def print_distribution(name, data):
    print(f"\n{name} distribution:")
    print(data["label"].value_counts(normalize=True))

print_distribution("Train", train_df)
print_distribution("Validation", val_df)
print_distribution("Test", test_df)

y_train = train_df["label"].to_numpy()
num_classes = 3

counts = np.bincount(y_train, minlength=num_classes)
N = counts.sum()

class_weights = N / (num_classes * counts)
class_weights = torch.tensor(class_weights, dtype=torch.float32)

print(f"Train counts: {counts}")
print(f"Class weights: {class_weights}")

print("Loading FastText Vectors...")
ft = api.load("fasttext-wiki-news-subwords-300")
print("FastText loaded. Vector size:", ft.vector_size)

TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:\.\d+)?")

def tokenize(text: str):
    return TOKEN_RE.findall(text.lower())

def mean_pool_fasttext(tokens, ft_vectors, dim=300):
    if len(tokens) == 0: # addresses edge case of a sentence w/ 0 tokens
        return np.zeros(dim, dtype=np.float32)

    vecs = []
    for tok in tokens:
        try:
            vecs.append(ft_vectors[tok])
        except KeyError:
            continue
    
    if len(vecs) == 0:
        return np.zeros(dim, dtype=np.float32)

    return np.mean(vecs, axis=0).astype(np.float32)

class NumpyDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)         
        self.y = torch.from_numpy(y)          

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    

def build_features(df, ft):
    X = np.vstack([
        mean_pool_fasttext(tokenize(sent), ft)
        for sent in df["sentence"]
    ])
    y = df["label"].values.astype(np.int64)
    return X, y

X_train, y_train = build_features(train_df, ft)
X_val, y_val = build_features(val_df, ft)
X_test, y_test = build_features(test_df, ft)


train_loader = DataLoader(NumpyDataset(X_train, y_train), batch_size=64, shuffle=True)
val_loader = DataLoader(NumpyDataset(X_val, y_val), batch_size=256, shuffle=False)
test_loader = DataLoader(NumpyDataset(X_test, y_test), batch_size=256, shuffle=False)

# Set up the MLP model
class MLPClassifier(nn.Module):
    def __init__(self, input_dim=300, hidden_dims=(256, 128), dropout=0.2, num_classes=3):
        super().__init__()

        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = h

        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

device = torch.device("mps")
model = MLPClassifier(input_dim=300, hidden_dims=(256, 128), dropout=0.2, num_classes=3).to(device)

criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))

# define a few helper functions to help set up the training loop
def accuracy_from_logits(logits, y):
    preds = torch.argmax(logits, dim=1)
    return (preds == y).float().mean().item()

@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_y = []

    for X, y in dataloader:
        X = X.to(device)
        y = y.to(device)

        logits = model(X)
        loss = criterion(logits, y)
        total_loss += loss.item() * y.size(0)

        preds = torch.argmax(logits, dim=1)
        all_preds.append(preds.detach().cpu().numpy())
        all_y.append(y.detach().cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_y = np.concatenate(all_y)

    avg_loss = total_loss / len(dataloader.dataset)
    acc = (all_preds == all_y).mean()
    macro_f1 = f1_score(all_y, all_preds, average="macro")

    return avg_loss, acc, macro_f1

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    all_preds = []
    all_y = []

    for X, y in dataloader:
        X = X.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * y.size(0)

        preds = torch.argmax(logits, dim=1)
        all_preds.append(preds.detach().cpu().numpy())
        all_y.append(y.detach().cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_y = np.concatenate(all_y)

    avg_loss = total_loss / len(dataloader.dataset)
    acc = (all_preds == all_y).mean()
    macro_f1 = f1_score(all_y, all_preds, average="macro")

    return avg_loss, acc, macro_f1

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))

#define training loop
def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    device,
    num_epochs=30,
    save_path="best_mlp_fasttext.pt",
):
    best_val_f1 = -1.0

    history = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "train_macro_f1": [],
        "val_loss": [],
        "val_acc": [],
        "val_macro_f1": [],
    }

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc, train_f1 = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_acc, val_f1 = evaluate(
            model, val_loader, criterion, device
        )

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["train_macro_f1"].append(train_f1)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_macro_f1"].append(val_f1)

        print(
            f"Epoch {epoch:02d} | "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} f1 {train_f1:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f} f1 {val_f1:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), save_path)
            print(f"Saved new best model (val macro-F1 = {best_val_f1:.4f}) -> {save_path}")

    history_df = pd.DataFrame(history)
    return best_val_f1, history_df

best_val_f1, history_df = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    criterion=criterion,
    device=device,
    num_epochs=30,
    save_path="best_mlp_fasttext.pt",
)

history_df.to_csv("mlp_fasttext_history.csv", index=False)
print(history_df.tail())

# plot metrics and save plots
def plot_metric(history_df, train_col, val_col, ylabel, filename):
    plt.figure()
    plt.plot(history_df["epoch"], history_df[train_col], label="train")
    plt.plot(history_df["epoch"], history_df[val_col], label="val")
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} vs Epochs")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()

plot_metric(history_df, "train_loss", "val_loss", "Loss", "loss_vs_epochs.png")
plot_metric(history_df, "train_acc", "val_acc", "Accuracy", "accuracy_vs_epochs.png")
plot_metric(history_df, "train_macro_f1", "val_macro_f1", "Macro F1", "macro_f1_vs_epochs.png")

@torch.no_grad()
def evaluate_test_macro_f1(model_class, model_path, test_loader, device):
    # Recreate model and load best weights
    model = model_class().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_preds = []
    all_labels = []

    for X, y in test_loader:
        X = X.to(device)
        y = y.to(device)

        logits = model(X)
        preds = torch.argmax(logits, dim=1)

        all_preds.append(preds.cpu().numpy())
        all_labels.append(y.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    return macro_f1


test_macro_f1 = evaluate_test_macro_f1(
    model_class=MLPClassifier,
    model_path="best_mlp_fasttext.pt",
    test_loader=test_loader,
    device=device
)

print("Test Macro-F1:", test_macro_f1)
if test_macro_f1 > 0.65:
    print("MLP passes 0.65 threshold.")
else:
    print("MLP does not pass 0.65 threshold.")

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

@torch.no_grad()
def get_preds_and_labels(model, dataloader, device):
    model.eval()
    preds_all, y_all = [], []
    for X, y in dataloader:
        X = X.to(device)
        logits = model(X)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        preds_all.append(preds)
        y_all.append(y.cpu().numpy())
    return np.concatenate(preds_all), np.concatenate(y_all)

def save_confusion_matrix(model_class, model_path, test_loader, device,
                          filename="mlp_confusion_matrix.png"):
    model = model_class().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    preds, y = get_preds_and_labels(model, test_loader, device)

    cm = confusion_matrix(y, preds, labels=[0, 1, 2])
    disp = ConfusionMatrixDisplay(cm, display_labels=["neg", "neu", "pos"])

    plt.figure()
    disp.plot(values_format="d")
    plt.title("MLP Confusion Matrix (Test)")
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()
    print(f"Saved confusion matrix to {filename}")

save_confusion_matrix(
    model_class=MLPClassifier,
    model_path="best_mlp_fasttext.pt",
    test_loader=test_loader,
    device=device,
    filename="mlp_confusion_matrix.png"
)
