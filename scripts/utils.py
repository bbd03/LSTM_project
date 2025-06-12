import torch
import torch.nn.functional as F
from pathlib import Path
from torchmetrics.classification import MulticlassRecall, MulticlassPrecision, MulticlassF1Score
from typing import Optional, Callable, List, Dict
from scripts.dataloader import Dataloader
from collections import defaultdict
import logging

class Metrics:
    def __init__(self, num_classes, device='cpu'):
        self.recall = MulticlassRecall(num_classes=num_classes, average=None).to(device)
        self.precision = MulticlassPrecision(num_classes=num_classes, average=None).to(device)
        self.f1 = MulticlassF1Score(num_classes=num_classes, average='macro').to(device)

    def reset(self):
        self.recall.reset()
        self.precision.reset()
        self.f1.reset()

    def update(self, logits, labels):
        self.recall.update(logits, labels)
        self.precision.update(logits, labels)
        self.f1.update(logits, labels)

    def compute(self):
        return {
            'per_class_recall':    self.recall.compute().cpu().tolist(),
            'per_class_precision': self.precision.compute().cpu().tolist(),
            'macro_f1':            self.f1.compute().item()
        }

def save_model_weights(
    model: torch.nn.Module,
    path: Path
):
    path = Path(path)

    if path.is_dir():
        path = path / "model_weights.pt"
    else:
        if path.suffix.lower() != ".pt":
            path = path.with_suffix(path.suffix + ".pt" if path.suffix else ".pt")

    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), path)
    print(f"[utils] Model weights saved to {path}")

def evaluate_model(
    model: torch.nn.Module,
    dataloader: Dataloader,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    metrics: Metrics,
    num_classes: int,
    device: torch.device,
    logger: Optional[logging.Logger] = None
) -> Dict[str, float]:

    model.eval()
    m = metrics
    val_loss = 0.0
    all_preds, all_labels = [], []

    m.reset()
    with torch.no_grad():
        for x, lengths, y in dataloader:
            x, lengths, y = x.to(device), lengths.to(device), y.to(device)
            logits = model(x, lengths)
            val_loss += criterion(logits, y).item()

            m.update(logits, y)
            all_preds.append(logits.argmax(dim=1).cpu())
            all_labels.append(y.cpu())

    preds = torch.cat(all_preds)
    labs = torch.cat(all_labels)

    stats = m.compute()
    stats['val_loss'] = val_loss / len(dataloader)
    stats['accuracy'] = (preds == labs).float().mean().item()

    # per-class accuracy
    cnt = defaultdict(lambda: {'c': 0, 't': 0})
    for p, t in zip(preds.tolist(), labs.tolist()):
        cnt[t]['t'] += 1
        if p == t:
            cnt[t]['c'] += 1

    stats['per_class_accuracy'] = [
        (cnt[i]['c'] / cnt[i]['t']) if cnt[i]['t']>0 else 0.0
        for i in range(num_classes)
    ]

    if logger:
        logger.info(
            f"[Eval] loss={stats['val_loss']:.4f}  "
            f"acc={stats['accuracy']:.4f}  f1={stats['macro_f1']:.4f}"
        )
        for i in range(num_classes):
            logger.info(
                f"  cls {i:2d} | Rcl={stats['per_class_recall'][i]:.3f}  "
                f"Prc={stats['per_class_precision'][i]:.3f}  "
                f"Acc={stats['per_class_accuracy'][i]:.3f}"
            )

    return stats


def compute_class_weights(labels, num_classes, min_weight=None, max_weight=None):

    if not torch.is_tensor(labels):
        labels = torch.tensor(labels, dtype=torch.long)

    counts = torch.bincount(labels, minlength=num_classes).float()
    # avoid division by zero
    counts = counts.clamp(min=1.0)

    weights = 1.0 / counts
    if min_weight is not None:
        weights = weights.clamp(min=min_weight)
    if max_weight is not None:
        weights = weights.clamp(max=max_weight)

    return weights


def clip_gradients(
    model: torch.nn.Module,
    max_norm=1.0,
    norm_type=2.0
) -> float:
    """
    Clips all gradients in `model` so that the global norm <= max_norm.
    Returns the (pre-clipped) total norm for logging if you need it.
    """
    total_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), max_norm, norm_type
    )
    return total_norm



def log_gradients(model, logger, prefix="", batch_idx=None):
    """
    Logs the L2 norm of each parameter’s gradient.
    """
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm(2).item()
            logger.info(f"{prefix}{name:30s} | grad-norm = {grad_norm:.4f}")
    if batch_idx is not None:
        logger.info(f"{prefix} after batch {batch_idx}")


if __name__ == "__main__":
    m = Metrics(27, )

