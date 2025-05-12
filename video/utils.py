import torch
from sklearn.metrics import accuracy_score, confusion_matrix

def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for videos, labels in loader:
            videos = videos.to(device).repeat(1,3,1,1,1)
            labels = labels.to(device)
            outputs = model(videos)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    acc = accuracy_score(all_labels, all_preds)
    cm  = confusion_matrix(all_labels, all_preds)
    return acc, cm

class EarlyStopper:
    def __init__(self, patience=5, delta=1e-4):
        self.patience = patience
        self.delta    = delta
        self.best    = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, score):
        if self.best is None or score > self.best + self.delta:
            self.best = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
