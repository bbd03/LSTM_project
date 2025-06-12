import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.data import WeightedRandomSampler
from scripts.dataset import MovieDataset, collate_batch
from scripts.model import LSTMClassifier
from scripts.config import Config
from scripts.utils import Metrics, evaluate_model
from scripts.utils import clip_gradients, log_gradients
from scripts.utils import compute_class_weights
from collections import defaultdict

# Setting up logging
logger = logging.getLogger(__name__)  # module-specific logger

def train_model(
        full_ds,       
        vocab,           
        batch_size,          
        num_epochs,
        test_num,             
        learning_rate,           
        emb_dim,                 
        hidden_size,             
        num_classes,             
        pad_idx,                 
        max_len=None,
        use_class_weights=False,
        max_class_weight=None,
        min_class_weight=None,
        use_clip_grad=False,
        grad_clip_norm=None, 
        use_sampler=False,           
        device=None,             
        log_interval=100         
    ):

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
     
    full_ds = MovieDataset(full_ds, vocab, max_len=max_len)

    # splitting
    N = len(full_ds)
    val_size   = int(test_num * N)  
    train_size = N - val_size  

    train_ds, val_ds = random_split(full_ds, [train_size, val_size])

    class_weights = None # set the class weights

    if use_class_weights:
        # Compute per-class weights and optionally clamp them
        class_weights = compute_class_weights(
            full_ds.labels,
            num_classes,
            min_weight=min_class_weight,
            max_weight=max_class_weight
        ).to(device)

    sampler = None
    if use_class_weights and use_sampler:
        # Extract labels *for the training subset only*
        train_labels = torch.tensor(
            [full_ds.labels[i] for i in train_ds.indices],
            dtype=torch.long,
            device=device
        )
        # Build per-example weights from class_weights
        example_weights = class_weights[train_labels]  # shape: [train_size]
        sampler = WeightedRandomSampler(
            weights=example_weights,
            num_samples=len(example_weights),
            replacement=True
        )

    train_dl = DataLoader(
        dataset=train_ds,
        batch_size=batch_size,
        shuffle=False if sampler is not None else True,
        sampler=sampler,
        collate_fn=collate_batch,
        num_workers=4,
        pin_memory=True
    )

    val_dl = DataLoader(
        dataset=val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_batch,
        num_workers=4,
        pin_memory=True
    )

    model = LSTMClassifier(
        vocab_size=len(vocab),
        embed_mat=vocab.embed_mat,
        hidden_size=hidden_size,
        num_classes=num_classes,
        pad_idx=pad_idx,
        bidirectional=True
    ).to(device)

 

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # setting the sheduler
    # A) validation-aware: Reduce LR when val-metric plateaus

    # setting up the metrics

    metrics = Metrics(num_classes, device)

    # Training loop
    best_f1 = 0.0
    logger.info("🚀 Starting training")

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0

        for batch_idx, (x_pad, lengths, labels) in enumerate(train_dl, start=1):
            x_pad, labels = x_pad.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(x_pad, lengths)  # [B, num_classes]
            loss = criterion(logits, labels)
            loss.backward()

            if use_clip_grad:
                norm = clip_gradients(model, max_norm=grad_clip_norm)

            optimizer.step()

            running_loss += loss.item()
            
            batch_acc = (logits.argmax(dim=1) == labels).float().mean().item()
            if batch_idx % log_interval == 0:
                lr = optimizer.param_groups[0]["lr"]
                logger.info(
                    f"[Train] b={batch_idx:4d}  loss={loss.item():.4f}  "
                    f"acc={batch_acc:.3f}  lr={lr:.2e}"
                )
                log_gradients(model, logger, "grads>>>", batch_idx)

        avg_train_loss = running_loss / len(train_dl)
        logger.info(
            f"✅ Epoch {epoch:2d} — train_loss={avg_train_loss:.4f}"
        )

        stats = evaluate_model(
            model=model,
            dataloader=val_dl,
            criterion=criterion,
            metrics=metrics,
            num_classes=num_classes,
            device=device,
            logger=logger
        )

        logger.info(
            f"🔎 Epoch {epoch:2d} — val_loss={stats['val_loss']:.4f}  "
            f"val_acc={stats['accuracy']:.4f}  val_f1={stats['macro_f1']:.4f}"
        )

    logger.info("🏁 Training complete")

    return model

