"""
Domain Adaptation Training Utilities
Handles dual-domain training with DANN, MMD, and Orthogonal losses
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
from itertools import cycle

from ..losses import mmd_loss, class_conditional_mmd_loss, orthogonal_loss
from .early_stopping import EarlyStopping


def get_rampup_coefficient(epoch, rampup_epochs=5):
    """
    Compute ramp-up coefficient for gradual increase of lambda values

    Args:
        epoch: Current epoch (0-indexed)
        rampup_epochs: Number of epochs for ramp-up

    Returns:
        coefficient: Value between 0 and 1
    """
    if epoch >= rampup_epochs:
        return 1.0
    else:
        return float(epoch) / float(rampup_epochs)


def compute_domain_loss(domain_pred, domain_label, label_smoothing=0.0):
    """
    Compute binary cross-entropy loss for domain prediction with optional label smoothing

    Args:
        domain_pred: Domain predictions [batch_size, 1] or [1]
        domain_label: Target domain labels (0 or 1)
        label_smoothing: Label smoothing factor (0.0 to 0.5)

    Returns:
        loss: BCE loss (scalar)
    """
    # Apply label smoothing if specified
    if label_smoothing > 0:
        if domain_label == 0:
            target = torch.full_like(domain_pred, label_smoothing)
        else:
            target = torch.full_like(domain_pred, 1.0 - label_smoothing)
    else:
        target = torch.full_like(domain_pred, float(domain_label))

    loss = F.binary_cross_entropy_with_logits(domain_pred, target)
    return loss


def train_one_epoch_domain_adaptation(
    model, source_loader, target_loader, optimizer, criterion,
    device, epoch, num_epochs, config, fold=None, target_loader_unlabeled=None
):
    """
    Train for one epoch with domain adaptation (supports supervised, unsupervised, semi-supervised)

    Args:
        model: DomainAdaptationModel
        source_loader: DataLoader for source domain (QLD1) - always labeled
        target_loader: DataLoader for target domain (QLD2) - labeled or unlabeled based on DA_MODE
        optimizer: Optimizer
        criterion: Classification loss criterion
        device: Device
        epoch: Current epoch
        num_epochs: Total number of epochs
        config: Config object with hyperparameters
        fold: Fold number (optional)
        target_loader_unlabeled: DataLoader for unlabeled target samples (semi-supervised only)

    Returns:
        metrics: Dictionary with training metrics
    """
    model.train()

    # Determine DA mode
    da_mode = config.DA_MODE if hasattr(config, 'DA_MODE') else 'supervised'

    # Initialize metric trackers
    epoch_loss_total = 0.0
    epoch_loss_cls_source = 0.0
    epoch_loss_cls_target = 0.0
    epoch_loss_adv = 0.0
    epoch_loss_mmd = 0.0
    epoch_loss_orth = 0.0

    source_preds = []
    source_labels = []
    target_preds = []
    target_labels = []

    # Compute ramp-up coefficients for this epoch
    rampup_coef = get_rampup_coefficient(epoch, config.RAMPUP_EPOCHS)

    lambda_adv = config.LAMBDA_ADV * rampup_coef if config.RAMPUP_LAMBDA_ADV else config.LAMBDA_ADV
    lambda_mmd = config.LAMBDA_MMD * rampup_coef if config.RAMPUP_LAMBDA_MMD else config.LAMBDA_MMD
    lambda_orth = config.LAMBDA_ORTH  # Constant
    grl_coeff = config.GRL_COEFF * rampup_coef if config.RAMPUP_GRL_COEFF else config.GRL_COEFF

    # Update GRL lambda
    model.set_grl_lambda(grl_coeff)

    # Cycle the smaller loader to match the larger one
    source_iter = iter(source_loader)
    target_iter = iter(target_loader)

    # For semi-supervised, also iterate over unlabeled target data
    if da_mode == 'semi_supervised' and target_loader_unlabeled is not None:
        target_unlabeled_iter = iter(target_loader_unlabeled)
        num_batches = max(len(source_loader), len(target_loader), len(target_loader_unlabeled))
    else:
        target_unlabeled_iter = None
        num_batches = max(len(source_loader), len(target_loader))

    fold_str = f" (Fold {fold})" if fold is not None else ""
    train_pbar = tqdm(range(num_batches),
                     desc=f'Epoch {epoch+1}/{num_epochs}{fold_str} - DA Training ({da_mode.upper()})',
                     leave=False, unit='batch')

    for batch_idx in train_pbar:
        # Get batches from both domains (cycle if needed)
        try:
            source_batch = next(source_iter)
        except StopIteration:
            source_iter = iter(source_loader)
            source_batch = next(source_iter)

        try:
            target_batch = next(target_iter)
        except StopIteration:
            target_iter = iter(target_loader)
            target_batch = next(target_iter)

        # Unpack batches
        source_bags, source_labels_batch, _, _ = source_batch
        target_bags, target_labels_batch, _, _ = target_batch

        source_bags = source_bags.to(device)
        target_bags = target_bags.to(device)
        source_labels_batch = source_labels_batch.to(device).long()

        # Handle target labels based on DA mode
        if da_mode == 'unsupervised':
            # Target is unlabeled (labels are -1), don't use for classification
            target_has_labels = False
        elif da_mode == 'semi_supervised':
            # Target has some labels, others are -1
            target_labels_batch = target_labels_batch.to(device).long()
            target_has_labels = (target_labels_batch >= 0).any().item()
        else:  # supervised
            target_labels_batch = target_labels_batch.to(device).long()
            target_has_labels = True

        # For semi-supervised: also get unlabeled target batch
        if da_mode == 'semi_supervised' and target_unlabeled_iter is not None:
            try:
                target_unlabeled_batch = next(target_unlabeled_iter)
            except StopIteration:
                target_unlabeled_iter = iter(target_loader_unlabeled)
                target_unlabeled_batch = next(target_unlabeled_iter)
            target_unlabeled_bags, _, _, _ = target_unlabeled_batch
            target_unlabeled_bags = target_unlabeled_bags.to(device)
        else:
            target_unlabeled_bags = None

        optimizer.zero_grad()

        # === Forward pass for SOURCE domain ===
        source_batch_logits = []
        source_batch_features = []
        source_domain_preds = []

        for i in range(source_bags.shape[0]):
            bag = source_bags[i]
            logits, features, domain_pred = model.forward_with_domain(bag)
            source_batch_logits.append(logits)
            source_batch_features.append(features)
            source_domain_preds.append(domain_pred)

        source_batch_logits = torch.stack(source_batch_logits)
        source_batch_features = torch.stack(source_batch_features)
        source_domain_preds = torch.stack(source_domain_preds)

        # === Forward pass for TARGET domain (labeled or unlabeled) ===
        target_batch_logits = []
        target_batch_features = []
        target_domain_preds = []

        for i in range(target_bags.shape[0]):
            bag = target_bags[i]
            logits, features, domain_pred = model.forward_with_domain(bag)
            target_batch_logits.append(logits)
            target_batch_features.append(features)
            target_domain_preds.append(domain_pred)

        target_batch_logits = torch.stack(target_batch_logits)
        target_batch_features = torch.stack(target_batch_features)
        target_domain_preds = torch.stack(target_domain_preds)

        # === Forward pass for UNLABELED target (semi-supervised only) ===
        if target_unlabeled_bags is not None:
            target_unlabeled_features = []
            target_unlabeled_domain_preds = []

            for i in range(target_unlabeled_bags.shape[0]):
                bag = target_unlabeled_bags[i]
                _, features, domain_pred = model.forward_with_domain(bag)
                target_unlabeled_features.append(features)
                target_unlabeled_domain_preds.append(domain_pred)

            target_unlabeled_features = torch.stack(target_unlabeled_features)
            target_unlabeled_domain_preds = torch.stack(target_unlabeled_domain_preds)
        else:
            target_unlabeled_features = None
            target_unlabeled_domain_preds = None

        # === Compute losses ===

        # 1. Classification losses (depends on DA mode)
        loss_cls_source = criterion(source_batch_logits, source_labels_batch)

        if da_mode == 'unsupervised':
            # No target classification loss
            loss_cls_target = torch.tensor(0.0, device=device)
            loss_cls = loss_cls_source
        elif da_mode == 'semi_supervised' and target_has_labels:
            # Only compute loss on labeled target samples
            labeled_mask = target_labels_batch >= 0
            if labeled_mask.sum() > 0:
                loss_cls_target = criterion(
                    target_batch_logits[labeled_mask],
                    target_labels_batch[labeled_mask]
                )
            else:
                loss_cls_target = torch.tensor(0.0, device=device)
            loss_cls = loss_cls_source + loss_cls_target
        else:  # supervised or semi_supervised without labels in this batch
            if target_has_labels:
                loss_cls_target = criterion(target_batch_logits, target_labels_batch)
                loss_cls = loss_cls_source + loss_cls_target
            else:
                loss_cls_target = torch.tensor(0.0, device=device)
                loss_cls = loss_cls_source

        # 2. Adversarial domain confusion loss (DANN) - uses ALL target data (labeled + unlabeled)
        # Domain labels: 0 for source (QLD1), 1 for target (QLD2)
        loss_adv_source = compute_domain_loss(
            source_domain_preds, 0, config.DOMAIN_LABEL_SMOOTHING
        )
        loss_adv_target = compute_domain_loss(
            target_domain_preds, 1, config.DOMAIN_LABEL_SMOOTHING
        )

        # Include unlabeled target in adversarial loss for semi-supervised
        if target_unlabeled_domain_preds is not None:
            loss_adv_target_unlabeled = compute_domain_loss(
                target_unlabeled_domain_preds, 1, config.DOMAIN_LABEL_SMOOTHING
            )
            loss_adv = loss_adv_source + loss_adv_target + loss_adv_target_unlabeled
        else:
            loss_adv = loss_adv_source + loss_adv_target

        # 3. MMD loss - mode-dependent
        # For unsupervised/semi-supervised: use non-class-conditional MMD
        # For supervised: can use class-conditional if enabled
        if da_mode == 'supervised' and config.USE_CLASS_COND_MMD:
            # Class-conditional MMD (requires both domains to be labeled)
            loss_mmd = class_conditional_mmd_loss(
                source_batch_features, target_batch_features,
                source_labels_batch, target_labels_batch,
                num_classes=config.NUM_CLASSES,
                bandwidths=config.MMD_BANDWIDTHS
            )
        elif da_mode == 'semi_supervised':
            # For semi-supervised: combine labeled and unlabeled target features
            if target_unlabeled_features is not None:
                all_target_features = torch.cat([target_batch_features, target_unlabeled_features], dim=0)
            else:
                all_target_features = target_batch_features

            # Use non-class-conditional MMD for semi-supervised (can't split unlabeled by class)
            loss_mmd = mmd_loss(
                source_batch_features, all_target_features,
                bandwidths=config.MMD_BANDWIDTHS
            )
        else:  # unsupervised
            # Non-class-conditional MMD (target is unlabeled)
            loss_mmd = mmd_loss(
                source_batch_features, target_batch_features,
                bandwidths=config.MMD_BANDWIDTHS
            )

        # 4. Orthogonal regularization (weight matrices)
        W_cls = model.get_classifier_weights()
        W_dom = model.get_discriminator_weights()
        loss_orth = orthogonal_loss(W_cls, W_dom)

        # Total loss
        loss_total = (loss_cls +
                     lambda_adv * loss_adv +
                     lambda_mmd * loss_mmd +
                     lambda_orth * loss_orth)

        # Backward pass
        loss_total.backward()

        # Gradient clipping if enabled
        if config.USE_GRADIENT_CLIPPING:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.GRADIENT_CLIP_MAX_NORM
            )

        optimizer.step()

        # Track metrics (handle both tensor and float cases)
        epoch_loss_total += loss_total.item()
        epoch_loss_cls_source += loss_cls_source.item()
        epoch_loss_cls_target += loss_cls_target.item()
        epoch_loss_adv += loss_adv.item()
        epoch_loss_mmd += loss_mmd.item() if torch.is_tensor(loss_mmd) else loss_mmd
        epoch_loss_orth += loss_orth.item() if torch.is_tensor(loss_orth) else loss_orth

        # Track predictions (only for labeled samples)
        source_preds.extend(torch.argmax(source_batch_logits, dim=1).cpu().numpy())
        source_labels.extend(source_labels_batch.cpu().numpy())

        # For target: only track if labeled
        if da_mode != 'unsupervised':
            if da_mode == 'semi_supervised':
                # Only track labeled target samples
                labeled_mask = target_labels_batch >= 0
                if labeled_mask.sum() > 0:
                    target_preds.extend(torch.argmax(target_batch_logits[labeled_mask], dim=1).cpu().numpy())
                    target_labels.extend(target_labels_batch[labeled_mask].cpu().numpy())
            else:  # supervised
                target_preds.extend(torch.argmax(target_batch_logits, dim=1).cpu().numpy())
                target_labels.extend(target_labels_batch.cpu().numpy())

        # Update progress bar
        mmd_val = loss_mmd.item() if torch.is_tensor(loss_mmd) else loss_mmd
        train_pbar.set_postfix({
            'Loss': f'{loss_total.item():.4f}',
            'Cls': f'{loss_cls.item():.4f}',
            'Adv': f'{loss_adv.item():.4f}',
            'MMD': f'{mmd_val:.4f}',
            'λ_adv': f'{lambda_adv:.3f}'
        })

    # Compute average losses
    avg_loss_total = epoch_loss_total / num_batches
    avg_loss_cls_source = epoch_loss_cls_source / num_batches
    avg_loss_cls_target = epoch_loss_cls_target / num_batches
    avg_loss_adv = epoch_loss_adv / num_batches
    avg_loss_mmd = epoch_loss_mmd / num_batches
    avg_loss_orth = epoch_loss_orth / num_batches

    # Compute bag-level accuracies
    source_bag_acc = accuracy_score(source_labels, source_preds)

    # Target accuracy only if we have labeled target samples
    if da_mode == 'unsupervised' or len(target_labels) == 0:
        target_bag_acc = 0.0  # No labeled target data
    else:
        target_bag_acc = accuracy_score(target_labels, target_preds)

    metrics = {
        'loss_total': avg_loss_total,
        'loss_cls_source': avg_loss_cls_source,
        'loss_cls_target': avg_loss_cls_target,
        'loss_adv': avg_loss_adv,
        'loss_mmd': avg_loss_mmd,
        'loss_orth': avg_loss_orth,
        'source_bag_acc': source_bag_acc,
        'target_bag_acc': target_bag_acc,
        'lambda_adv': lambda_adv,
        'lambda_mmd': lambda_mmd,
        'grl_coeff': grl_coeff,
        'da_mode': da_mode
    }

    return metrics


def validate_domain_adaptation(model, source_loader, target_loader, device,
                               epoch, num_epochs, fold=None, criterion=None, config=None):
    """
    Validate on both source and target domains (pile-level)
    Handles unlabeled target data for unsupervised/semi-supervised modes

    Args:
        model: DomainAdaptationModel
        source_loader: Source domain validation loader
        target_loader: Target domain validation loader (may be unlabeled)
        device: Device
        epoch: Current epoch
        num_epochs: Total epochs
        fold: Fold number (optional)
        criterion: Loss criterion (optional)
        config: Config object (optional, for DA mode detection)

    Returns:
        metrics: Dictionary with validation metrics for both domains
    """
    from .pooling import aggregate_pile_predictions

    model.eval()

    # Determine if target is labeled
    da_mode = config.DA_MODE if config and hasattr(config, 'DA_MODE') else 'supervised'
    target_is_labeled = (da_mode == 'supervised')

    # Process both domains
    domains = [
        ('source', source_loader, True),  # Source is always labeled
        ('target', target_loader, target_is_labeled)
    ]

    results = {}

    for domain_name, loader, is_labeled in domains:
        pile_predictions = {}
        val_loss = 0.0
        num_batches = 0

        fold_str = f" (Fold {fold})" if fold is not None else ""
        val_pbar = tqdm(loader,
                       desc=f'Epoch {epoch+1}/{num_epochs}{fold_str} - Val ({domain_name})',
                       leave=False, unit='bag')

        with torch.no_grad():
            for bags, labels, pile_ids, image_paths in val_pbar:
                bags = bags.to(device)

                # Process each bag
                batch_logits = []
                for i in range(bags.shape[0]):
                    bag = bags[i]
                    pile_id = pile_ids[i]
                    label = labels[i].item() if is_labeled else -1

                    # Get prediction
                    logits, _ = model(bag)
                    batch_logits.append(logits)
                    pred_probs = torch.softmax(logits, dim=0)

                    # Store for pile-level aggregation
                    if pile_id not in pile_predictions:
                        pile_predictions[pile_id] = {'preds': [], 'label': label}
                    pile_predictions[pile_id]['preds'].append(pred_probs.cpu().numpy())

                # Compute validation loss if criterion provided and data is labeled
                if criterion is not None and is_labeled:
                    labels = labels.to(device).long()
                    # Filter out unlabeled samples (label == -1) for semi-supervised
                    labeled_mask = labels >= 0
                    if labeled_mask.sum() > 0:
                        batch_logits_tensor = torch.stack(batch_logits)
                        loss = criterion(batch_logits_tensor[labeled_mask], labels[labeled_mask])
                        val_loss += loss.item()
                        num_batches += 1

        # Aggregate to pile level
        pile_true_labels = []
        pile_pred_labels = []

        for pile_id, data in pile_predictions.items():
            bag_probs = torch.tensor(np.array(data['preds']), dtype=torch.float32).to(device)
            agg_probs, _ = aggregate_pile_predictions(bag_probs, method='mean')

            if isinstance(agg_probs, torch.Tensor):
                pred_class = torch.argmax(agg_probs).item()
            else:
                pred_class = np.argmax(agg_probs)

            pile_pred_labels.append(pred_class)
            pile_true_labels.append(data['label'])

        # Compute metrics only if labeled
        if is_labeled:
            # Filter out unlabeled piles
            labeled_indices = [i for i, label in enumerate(pile_true_labels) if label >= 0]
            if len(labeled_indices) > 0:
                filtered_true = [pile_true_labels[i] for i in labeled_indices]
                filtered_pred = [pile_pred_labels[i] for i in labeled_indices]
                pile_acc = accuracy_score(filtered_true, filtered_pred)
                pile_f1 = f1_score(filtered_true, filtered_pred, average='weighted')
            else:
                pile_acc = 0.0
                pile_f1 = 0.0
        else:
            # No ground truth for unlabeled data
            pile_acc = 0.0
            pile_f1 = 0.0

        avg_val_loss = val_loss / num_batches if num_batches > 0 else 0.0

        results[domain_name] = {
            'pile_acc': pile_acc,
            'pile_f1': pile_f1,
            'val_loss': avg_val_loss,
            'pile_preds': pile_pred_labels,
            'pile_labels': pile_true_labels
        }

    return results


def train_model_domain_adaptation(
    model, source_train_loader, source_val_loader,
    target_train_loader, target_val_loader,
    num_epochs, learning_rate, config,
    class_weights=None, fold=None, resume_state=None,
    target_train_loader_unlabeled=None
):
    """
    Train model with domain adaptation (supports supervised, unsupervised, semi-supervised)

    Args:
        model: DomainAdaptationModel
        source_train_loader: Source domain training loader (always labeled)
        source_val_loader: Source domain validation loader (always labeled)
        target_train_loader: Target domain training loader (labeled/unlabeled based on DA_MODE)
        target_val_loader: Target domain validation loader (labeled/unlabeled based on DA_MODE)
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        config: Config object
        class_weights: Class weights for loss
        fold: Fold number (optional)
        resume_state: Resume training state (optional)
        target_train_loader_unlabeled: Unlabeled target loader (semi-supervised only)

    Returns:
        Training history
    """
    logger = logging.getLogger(__name__)
    device = next(model.parameters()).device

    # Get DA mode
    da_mode = config.DA_MODE if hasattr(config, 'DA_MODE') else 'supervised'
    print(f"\nDomain Adaptation Mode: {da_mode.upper()}")

    # Print device info
    if device.type == 'cuda':
        print(f"\nGPU Training: {torch.cuda.get_device_name(0)}")
        if logger.hasHandlers():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # Setup optimizer
    if config.USE_ADAMW:
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,
                                     weight_decay=config.WEIGHT_DECAY)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                    weight_decay=config.WEIGHT_DECAY)

    # Setup scheduler
    scheduler = None
    if config.USE_LR_SCHEDULER and config.LR_SCHEDULER_TYPE == 'reduce_on_plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=config.LR_SCHEDULER_MODE,
            factor=config.LR_SCHEDULER_FACTOR,
            patience=config.LR_SCHEDULER_PATIENCE,
            min_lr=config.LR_SCHEDULER_MIN_LR
        )

    # Setup loss
    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    # Training history
    history = {
        'train_loss': [],
        'source_val_acc': [],
        'source_val_f1': [],
        'target_val_acc': [],
        'target_val_f1': []
    }

    best_metric = 0.0
    early_stopping = None
    if config.USE_EARLY_STOPPING:
        early_stopping = EarlyStopping(
            patience=config.EARLY_STOPPING_PATIENCE,
            min_delta=config.EARLY_STOPPING_MIN_DELTA,
            verbose=True
        )

    # Training loop
    fold_str = f" (Fold {fold})" if fold is not None else ""
    epoch_pbar = tqdm(range(num_epochs),
                     desc=f'DA Training{fold_str}',
                     unit='epoch')

    for epoch in epoch_pbar:
        # Train
        train_metrics = train_one_epoch_domain_adaptation(
            model, source_train_loader, target_train_loader,
            optimizer, criterion, device, epoch, num_epochs, config, fold,
            target_loader_unlabeled=target_train_loader_unlabeled
        )

        # Validate
        val_metrics = validate_domain_adaptation(
            model, source_val_loader, target_val_loader,
            device, epoch, num_epochs, fold, criterion, config=config
        )

        # Update history
        history['train_loss'].append(train_metrics['loss_total'])
        history['source_val_acc'].append(val_metrics['source']['pile_acc'])
        history['source_val_f1'].append(val_metrics['source']['pile_f1'])
        history['target_val_acc'].append(val_metrics['target']['pile_acc'])
        history['target_val_f1'].append(val_metrics['target']['pile_f1'])

        # Update scheduler
        if scheduler is not None:
            scheduler.step(val_metrics['target']['val_loss'])

        # Update progress bar
        epoch_pbar.set_postfix({
            'S_Acc': f"{val_metrics['source']['pile_acc']:.3f}",
            'T_Acc': f"{val_metrics['target']['pile_acc']:.3f}",
            'T_F1': f"{val_metrics['target']['pile_f1']:.3f}"
        })

        # Log
        msg = (f"Epoch {epoch+1}/{num_epochs}{fold_str} - "
               f"Loss: {train_metrics['loss_total']:.4f}, "
               f"Source F1: {val_metrics['source']['pile_f1']:.4f}, "
               f"Target F1: {val_metrics['target']['pile_f1']:.4f}")
        if logger.hasHandlers():
            logger.info(msg)

        # Save best model based on target F1
        current_metric = val_metrics['target']['pile_f1']
        if current_metric > best_metric:
            best_metric = current_metric
            model_path = config.BEST_MODEL_PATH if fold is None else f'models/best_da_model_fold{fold}.pth'

            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(model_path), exist_ok=True)

            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'best_metric': best_metric,
                'history': history
            }, model_path)
            print(f"[BEST] Model saved (Target F1: {best_metric:.4f})")

        # Early stopping
        if early_stopping is not None:
            if early_stopping(current_metric, epoch + 1):
                print(f"Early stopping at epoch {epoch+1}")
                break

    return history
