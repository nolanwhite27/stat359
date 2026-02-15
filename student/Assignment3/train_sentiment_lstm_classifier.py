import random
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

import gensim.downloader as api
from torch.utils.data import Dataset, DataLoader

# device (cuda > mps > cpu)
device = "mps"
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)
print("Using device:", device)

dataset = load_dataset(
    "financial_phrasebank",
    "sentences_50agree",
    trust_remote_code=True
)

df = dataset["train"].to_pandas()

train_val_df, test_df = train_test_split(
    df,
    test_size=0.15,
    stratify=df["label"],
    random_state=42
)

train_df, val_df = train_test_split(
    train_val_df,
    test_size=0.15,
    stratify=train_val_df["label"],
    random_state=42
)

print("Sizes:", len(train_df), len(val_df), len(test_df))

print("Loading FastText vectors...")
ft = api.load("fasttext-wiki-news-subwords-300")
print("Loaded FastText. Dim:", ft.vector_size)

MAX_LEN = 32
TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:\.\d+)?")

def tokenize(text: str):
    return TOKEN_RE.findall(text.lower())

def sentence_to_matrix(sentence: str, ft, max_len=MAX_LEN):
    
    tokens = tokenize(sentence)[:max_len]
    dim = ft.vector_size
    mat = np.zeros((max_len, dim), dtype=np.float32)

    for i, tok in enumerate(tokens):
        try:
            mat[i] = ft[tok]
        except KeyError:
            pass  # leave as zeros
    return mat

def build_sequence_features(df, ft, max_len=MAX_LEN):
    X = np.stack([sentence_to_matrix(s, ft, max_len=max_len) for s in df["sentence"].tolist()])
    y = df["label"].to_numpy(dtype=np.int64)
    return X, y

X_train, y_train = build_sequence_features(train_df, ft)
X_val, y_val     = build_sequence_features(val_df, ft)
X_test, y_test   = build_sequence_features(test_df, ft)

print("X_train:", X_train.shape, "y_train:", y_train.shape)
print("X_val:  ", X_val.shape,   "y_val:  ", y_val.shape)
print("X_test: ", X_test.shape,  "y_test: ", y_test.shape)

num_classes = 3
counts = np.bincount(y_train, minlength=num_classes)
N = counts.sum()

class_weights = N / (num_classes * counts)
class_weights = torch.tensor(class_weights, dtype=torch.float32)

print("Train counts:", counts)
print("Class weights:", class_weights)

class SeqDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)  # (N, 32, 300) float32
        self.y = torch.from_numpy(y)  # (N,) int64

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(SeqDataset(X_train, y_train), batch_size=64, shuffle=True)
val_loader   = DataLoader(SeqDataset(X_val, y_val),     batch_size=256, shuffle=False)
test_loader  = DataLoader(SeqDataset(X_test, y_test),   batch_size=256, shuffle=False)

class LSTMClassifier(nn.Module):
    def __init__(self, input_dim=300, hidden_dim=128, num_layers=1, dropout=0.2, num_classes=3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,  # input: (B, 32, 300)
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x: (B, 32, 300)
        _, (h_n, _) = self.lstm(x)  # h_n: (num_layers, B, hidden_dim)
        h_last = h_n[-1]            # (B, hidden_dim)
        h_last = self.dropout(h_last)
        return self.fc(h_last)      # logits: (B, 3)
    
model = LSTMClassifier(input_dim=300, hidden_dim=128, num_layers=1, dropout=0.2, num_classes=3).to(device)
print(model)

import numpy as np
import torch
from sklearn.metrics import f1_score

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    all_preds, all_y = [], []

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


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_y = [], []

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

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    device,
    num_epochs=30,
    save_path="best_lstm_fasttext.pt",
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

    return best_val_f1, pd.DataFrame(history)

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

best_val_f1, history_df = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    criterion=criterion,
    device=device,
    num_epochs=30,  # required
    save_path="best_lstm_fasttext.pt",
)

history_df.to_csv("lstm_fasttext_history.csv", index=False)

plot_metric(history_df, "train_loss", "val_loss", "Loss", "lstm_loss_vs_epochs.png")
plot_metric(history_df, "train_acc", "val_acc", "Accuracy", "lstm_accuracy_vs_epochs.png")
plot_metric(history_df, "train_macro_f1", "val_macro_f1", "Macro F1", "lstm_macro_f1_vs_epochs.png")

@torch.no_grad()
def evaluate_test_macro_f1(model_class, model_kwargs, model_path, test_loader, device):
    model = model_class(**model_kwargs).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_preds, all_y = [], []
    for X, y in test_loader:
        X = X.to(device)
        logits = model(X)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.append(preds)
        all_y.append(y.numpy())

    all_preds = np.concatenate(all_preds)
    all_y = np.concatenate(all_y)
    return f1_score(all_y, all_preds, average="macro")

model_kwargs = dict(input_dim=300, hidden_dim=128, num_layers=1, dropout=0.2, num_classes=3)

test_macro_f1 = evaluate_test_macro_f1(
    model_class=LSTMClassifier,
    model_kwargs=model_kwargs,
    model_path="best_lstm_fasttext.pt",
    test_loader=test_loader,
    device=device,
)

print("Test macro-F1:", test_macro_f1)
if test_macro_f1 > 0.7:
    print("LSTM passes 0.70 threshold.")
else: 
    print("LSTM does not pass 0.70 threshold.")

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
        y_all.append(y.numpy())
    return np.concatenate(preds_all), np.concatenate(y_all)

def save_confusion_matrix(model_class, model_kwargs, model_path, test_loader, device,
                          filename="lstm_confusion_matrix.png"):
    model = model_class(**model_kwargs).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    preds, y = get_preds_and_labels(model, test_loader, device)

    cm = confusion_matrix(y, preds, labels=[0, 1, 2])
    disp = ConfusionMatrixDisplay(cm, display_labels=["neg", "neu", "pos"])

    plt.figure()
    disp.plot(values_format="d")
    plt.title("LSTM Confusion Matrix (Test)")
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()

save_confusion_matrix(
    model_class=LSTMClassifier,
    model_kwargs=model_kwargs,
    model_path="best_lstm_fasttext.pt",
    test_loader=test_loader,
    device=device,
    filename="lstm_confusion_matrix.png"
)