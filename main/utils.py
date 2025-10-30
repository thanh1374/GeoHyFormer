import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    y_true = []
    y_pred = []
    for data in loader:
        data = data.to(device)
        out = model(data)  # shape: (B, C) or (B,) or (B,1)
        # binary vs multiclass handling
        if out.dim() == 1 or (out.dim() == 2 and out.size(-1) == 1):
            logits = out.view(-1)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).long()
        else:
            preds = out.argmax(dim=-1)

        y_pred.extend(preds.cpu().tolist())
        y_true.extend(data.y.view(-1).cpu().tolist())

    if len(y_true) == 0:
        metrics = {
            'acc': 0.0,
            'f1_macro': 0.0,
            'f1_micro': 0.0,
            'precision_macro': 0.0,
            'recall_macro': 0.0,
            'classification_report': '',
            'confusion_matrix': None
        }
        return metrics

    metrics = {
        'acc': accuracy_score(y_true, y_pred),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_micro': f1_score(y_true, y_pred, average='micro', zero_division=0),
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'classification_report': classification_report(y_true, y_pred, zero_division=0, digits=4),
        'confusion_matrix': confusion_matrix(y_true, y_pred)
    }
    return metrics


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total = 0
    y_true = []
    y_pred = []

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)

        if out.dim() == 1 or (out.dim() == 2 and out.size(-1) == 1):
            logits = out.view(-1)
            # if using BCEWithLogitsLoss, criterion expects float targets
            if isinstance(criterion, torch.nn.BCEWithLogitsLoss):
                loss = criterion(logits, data.y.view(-1).float())
            else:
            # fallback: treat logits and integer labels (works if criterion expects long)
                loss = criterion(logits, data.y.view(-1).long())
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).long()
        else:
            loss = criterion(out, data.y.view(-1))
            preds = out.argmax(dim=-1)

        loss.backward()
        optimizer.step()

        batch_n = data.y.view(-1).size(0)
        total_loss += loss.item() * batch_n
        total += batch_n

        y_pred.extend(preds.cpu().tolist())
        y_true.extend(data.y.view(-1).cpu().tolist())

    avg_loss = total_loss / total if total > 0 else 0.0

    if len(y_true) == 0:
        metrics = {
            'acc': 0.0,
            'f1_macro': 0.0,
            'f1_micro': 0.0,
            'precision_macro': 0.0,
            'recall_macro': 0.0,
            'classification_report': '',
            'confusion_matrix': None
        }
    else:
        metrics = {
            'acc': accuracy_score(y_true, y_pred),
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_micro': f1_score(y_true, y_pred, average='micro', zero_division=0),
            'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'classification_report': classification_report(y_true, y_pred, zero_division=0, digits=4),
            'confusion_matrix': confusion_matrix(y_true, y_pred)
        }

    return avg_loss, metrics
