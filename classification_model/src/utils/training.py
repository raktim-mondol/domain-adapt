"""
Training utilities for bag-level training with pile-level evaluation
"""

import torch
import torch.nn as nn
import numpy as np
import logging
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
from .early_stopping import EarlyStopping


def compute_class_weights(train_df, num_classes=3, device='cpu'):
    """Compute class weights based on pile-level distribution"""
    pile_labels = train_df.groupby('pile')['BMA_label'].first().values
    pile_labels_indexed = pile_labels - 1  # Convert to 0-indexed
    
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.arange(num_classes),
        y=pile_labels_indexed
    )
    
    class_weights_tensor = torch.FloatTensor(class_weights).to(device)
    
    print("\nClass weights for handling imbalance:")
    for i, weight in enumerate(class_weights):
        class_count = np.sum(pile_labels_indexed == i)
        print(f"  Class {i}: weight={weight:.4f}, count={class_count} piles")
    
    return class_weights_tensor


def train_one_epoch(model, train_loader, optimizer, criterion, device, epoch, num_epochs, fold=None):
    """Train for one epoch (bag-level training)"""
    model.train()
    epoch_loss = 0.0
    all_preds = []
    all_labels = []
    
    fold_str = f" (Fold {fold})" if fold is not None else ""
    train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}{fold_str} - Training', 
                     leave=False, unit='bag')
    
    for batch_idx, (bags, labels, pile_ids, image_paths) in enumerate(train_pbar):
        # Move data to device
        bags = bags.to(device)  # [batch_size, num_patches, 3, H, W]
        labels = labels.to(device).long()  # [batch_size] - ensure long tensor for CrossEntropyLoss
        
        optimizer.zero_grad()
        
        # Forward pass - process each bag in batch
        batch_logits = []
        for i in range(bags.shape[0]):
            bag = bags[i]  # [num_patches, 3, H, W]
            logits, _ = model(bag)  # [num_classes]
            batch_logits.append(logits)
        
        batch_logits = torch.stack(batch_logits)  # [batch_size, num_classes]
        
        # Compute loss
        loss = criterion(batch_logits, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Track metrics
        epoch_loss += loss.item()
        preds = torch.argmax(batch_logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        # Update progress bar
        train_pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Avg Loss': f'{epoch_loss/(batch_idx+1):.4f}'
        })
    
    avg_loss = epoch_loss / len(train_loader)
    bag_acc = accuracy_score(all_labels, all_preds)
    
    return avg_loss, bag_acc


def validate_bag_level(model, val_loader, device, epoch, num_epochs, fold=None):
    """Validate on bag level (each image independently)"""
    model.eval()
    all_preds = []
    all_labels = []
    all_pile_ids = []
    
    fold_str = f" (Fold {fold})" if fold is not None else ""
    val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs}{fold_str} - Validation', 
                   leave=False, unit='bag')
    
    with torch.no_grad():
        for bags, labels, pile_ids, image_paths in val_pbar:
            bags = bags.to(device)
            labels = labels.to(device).long()
            
            # Process each bag in batch
            batch_logits = []
            for i in range(bags.shape[0]):
                bag = bags[i]  # [num_patches, 3, H, W]
                logits, _ = model(bag)  # [num_classes]
                batch_logits.append(logits)
            
            batch_logits = torch.stack(batch_logits)  # [batch_size, num_classes]
            preds = torch.argmax(batch_logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_pile_ids.extend(pile_ids)
    
    bag_acc = accuracy_score(all_labels, all_preds)
    return bag_acc, all_preds, all_labels, all_pile_ids


def validate_pile_level(model, val_loader, device, epoch, num_epochs, fold=None, criterion=None, pooling_method='mean'):
    """
    Validate on pile level by aggregating bag predictions
    Each bag gets a prediction, then we aggregate predictions per pile using specified method
    Returns validation loss if criterion is provided
    
    Args:
        pooling_method: One of ['mean', 'max', 'attention', 'majority']
    """
    from .pooling import aggregate_pile_predictions, AttentionPooling
    
    model.eval()
    pile_predictions = {}  # {pile_id: {'preds': [], 'label': int}}
    val_loss = 0.0
    num_batches = 0
    
    # Initialize attention pooling if needed
    attention_model = None
    if pooling_method == 'attention':
        # Get number of classes from config
        from configs.config import Config
        attention_model = AttentionPooling(num_classes=Config.NUM_CLASSES).to(device)
        attention_model.eval()
    
    fold_str = f" (Fold {fold})" if fold is not None else ""
    method_str = f" ({pooling_method.capitalize()} Pooling)"
    val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs}{fold_str} - Validation{method_str}', 
                   leave=False, unit='bag')
    
    with torch.no_grad():
        for bags, labels, pile_ids, image_paths in val_pbar:
            bags = bags.to(device)
            labels = labels.to(device).long()
            
            # Process each bag
            batch_logits = []
            for i in range(bags.shape[0]):
                bag = bags[i]  # [num_patches, 3, H, W]
                pile_id = pile_ids[i]
                label = labels[i].item()
                
                # Get prediction for this bag
                logits, _ = model(bag)
                batch_logits.append(logits)
                pred_probs = torch.softmax(logits, dim=0)  # [num_classes]
                
                # Store bag prediction for this pile
                if pile_id not in pile_predictions:
                    pile_predictions[pile_id] = {'preds': [], 'label': label}
                pile_predictions[pile_id]['preds'].append(pred_probs.cpu().numpy())
            
            # Compute validation loss if criterion provided
            if criterion is not None:
                batch_logits = torch.stack(batch_logits)
                loss = criterion(batch_logits, labels)
                val_loss += loss.item()
                num_batches += 1
    
    # Aggregate predictions per pile using specified method
    pile_true_labels = []
    pile_pred_labels = []
    
    for pile_id, data in pile_predictions.items():
        # Convert to tensor for consistent processing
        bag_probs = torch.tensor(np.array(data['preds']), dtype=torch.float32).to(device)
        
        # Aggregate using specified method
        agg_probs, _ = aggregate_pile_predictions(bag_probs, method=pooling_method, attention_model=attention_model)
        
        if isinstance(agg_probs, torch.Tensor):
            pred_class = torch.argmax(agg_probs).item()
        else:
            pred_class = np.argmax(agg_probs)
        
        pile_pred_labels.append(pred_class)
        pile_true_labels.append(data['label'])
    
    pile_acc = accuracy_score(pile_true_labels, pile_pred_labels)
    pile_f1 = f1_score(pile_true_labels, pile_pred_labels, average='weighted')
    
    # Calculate average validation loss
    avg_val_loss = val_loss / num_batches if num_batches > 0 and criterion is not None else None
    
    return pile_acc, pile_f1, pile_pred_labels, pile_true_labels, avg_val_loss


def train_model(model, train_loader, val_loader, num_epochs=50, learning_rate=1e-4,
                class_weights=None, fold=None, resume_state=None):
    """
    Train the BMA MIL classifier
    - Training happens at bag level (each image is a training sample)
    - Validation aggregates to pile level for final metrics
    """
    from configs.config import Config
    
    logger = logging.getLogger(__name__)
    device = next(model.parameters()).device
    
    # Print device information
    if device.type == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"\n{'='*60}")
        print(f"GPU Training Enabled")
        print(f"{'='*60}")
        print(f"GPU Device: {gpu_name}")
        print(f"Total GPU Memory: {gpu_memory:.2f} GB")
        print(f"Model is on device: {device}")
        print(f"{'='*60}\n")
        if logger.hasHandlers():
            logger.info(f"GPU Training - Device: {gpu_name}, Memory: {gpu_memory:.2f} GB")
    else:
        print(f"\n[WARNING] Training on CPU")
        if logger.hasHandlers():
            logger.info("Training on CPU")
    
    # Setup optimizer and loss
    if Config.USE_ADAMW:
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=Config.WEIGHT_DECAY)
        optimizer_name = "AdamW"
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=Config.WEIGHT_DECAY)
        optimizer_name = "Adam"
    
    print(f"Using {optimizer_name} optimizer (LR={learning_rate}, Weight Decay={Config.WEIGHT_DECAY})")
    if logger.hasHandlers():
        logger.info(f"Optimizer: {optimizer_name}, LR={learning_rate}, WD={Config.WEIGHT_DECAY}")
    
    # Setup learning rate scheduler
    scheduler = None
    if Config.USE_LR_SCHEDULER:
        if Config.LR_SCHEDULER_TYPE == 'reduce_on_plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=Config.LR_SCHEDULER_MODE,
                factor=Config.LR_SCHEDULER_FACTOR,
                patience=Config.LR_SCHEDULER_PATIENCE,
                min_lr=Config.LR_SCHEDULER_MIN_LR,
                threshold=Config.LR_SCHEDULER_THRESHOLD
            )
            print(f"Using ReduceLROnPlateau scheduler (mode={Config.LR_SCHEDULER_MODE}, patience={Config.LR_SCHEDULER_PATIENCE})")
            if logger.hasHandlers():
                logger.info(f"Scheduler: ReduceLROnPlateau, mode={Config.LR_SCHEDULER_MODE}, patience={Config.LR_SCHEDULER_PATIENCE}")
    
    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print("Using weighted CrossEntropyLoss")
        if logger.hasHandlers():
            logger.info("Using weighted CrossEntropyLoss")
    else:
        criterion = nn.CrossEntropyLoss()
        print("Using standard CrossEntropyLoss")
        if logger.hasHandlers():
            logger.info("Using standard CrossEntropyLoss")
    
    # Training history
    train_losses = []
    val_losses = []
    val_pile_accuracies = []
    val_pile_f1_scores = []
    best_val_acc = 0.0
    
    # Early stopping
    early_stopping = None
    if Config.USE_EARLY_STOPPING:
        early_stopping = EarlyStopping(
            patience=Config.EARLY_STOPPING_PATIENCE,
            min_delta=Config.EARLY_STOPPING_MIN_DELTA,
            verbose=True
        )
        print(f"Early stopping enabled: patience={Config.EARLY_STOPPING_PATIENCE}")
        if logger.hasHandlers():
            logger.info(f"Early stopping: patience={Config.EARLY_STOPPING_PATIENCE}")
    
    # Resume from checkpoint if provided
    start_epoch = 0
    if resume_state:
        if 'optimizer_state_dict' in resume_state:
            try:
                optimizer.load_state_dict(resume_state['optimizer_state_dict'])
            except Exception as e:
                print(f"[WARNING] Could not load optimizer state: {e}")
        
        if scheduler is not None and 'scheduler_state_dict' in resume_state:
            try:
                scheduler.load_state_dict(resume_state['scheduler_state_dict'])
            except Exception as e:
                print(f"[WARNING] Could not load scheduler state: {e}")
        
        train_losses = resume_state.get('train_losses', [])
        val_losses = resume_state.get('val_losses', [])
        val_pile_accuracies = resume_state.get('val_accuracies', [])
        val_pile_f1_scores = resume_state.get('val_f1_scores', [])
        best_val_acc = resume_state.get('best_val_acc', 0.0)
        start_epoch = resume_state.get('epoch', -1) + 1
        
        if start_epoch > 0:
            print(f"Resuming training from epoch {start_epoch}")
            if logger.hasHandlers():
                logger.info(f"Resuming from epoch {start_epoch}")
    
    # Training loop
    fold_str = f" (Fold {fold})" if fold is not None else ""
    epoch_pbar = tqdm(range(start_epoch, num_epochs), desc=f'Training Progress{fold_str}', 
                     unit='epoch', initial=start_epoch, total=num_epochs)
    
    for epoch in epoch_pbar:
        # Train for one epoch
        train_loss, train_bag_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch, num_epochs, fold
        )
        train_losses.append(train_loss)
        
        # Validate on pile level (with validation loss)
        pile_acc, pile_f1, _, _, val_loss = validate_pile_level(
            model, val_loader, device, epoch, num_epochs, fold, criterion=criterion
        )
        val_pile_accuracies.append(pile_acc)
        val_pile_f1_scores.append(pile_f1)
        val_losses.append(val_loss if val_loss is not None else 0.0)
        
        # Update learning rate scheduler
        old_lr = optimizer.param_groups[0]['lr']
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                # Use validation loss for ReduceLROnPlateau
                scheduler.step(val_loss if val_loss is not None else train_loss)
        
        # Get current learning rate and log if changed
        current_lr = optimizer.param_groups[0]['lr']
        if current_lr != old_lr:
            lr_msg = f'Learning rate reduced: {old_lr:.2e} -> {current_lr:.2e}'
            print(f"\n{lr_msg}")
            if logger.hasHandlers():
                logger.info(lr_msg)
        
        # Update progress bar
        epoch_pbar.set_postfix({
            'Train Loss': f'{train_loss:.4f}',
            'Val Loss': f'{val_loss:.4f}' if val_loss is not None else 'N/A',
            'Pile Acc': f'{pile_acc:.4f}',
            'Pile F1': f'{pile_f1:.4f}',
            'LR': f'{current_lr:.2e}'
        })
        
        # Logging
        msg = f'Epoch {epoch+1}/{num_epochs}{fold_str} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Pile Acc: {pile_acc:.4f}, Pile F1: {pile_f1:.4f}, LR: {current_lr:.2e}'
        if logger.hasHandlers():
            logger.info(msg)
        
        # Save best model
        if pile_acc > best_val_acc:
            best_val_acc = pile_acc
            model_path = Config.BEST_MODEL_PATH if fold is None else f'models/best_bma_mil_model_fold{fold}.pth'
            
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'best_val_acc': best_val_acc,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'val_accuracies': val_pile_accuracies,
                'val_f1_scores': val_pile_f1_scores
            }
            
            # Save scheduler state if available
            if scheduler is not None:
                checkpoint['scheduler_state_dict'] = scheduler.state_dict()
            
            torch.save(checkpoint, model_path)
            
            msg = f'[BEST] New best model saved (Pile Acc: {best_val_acc:.4f})'
            print(msg)
            if logger.hasHandlers():
                logger.info(msg)
        
        # Early stopping
        if early_stopping is not None:
            if early_stopping(pile_acc, epoch + 1):
                msg = f'Early stopping at epoch {epoch+1}. Best: {early_stopping.best_score:.4f} at epoch {early_stopping.best_epoch}'
                print(msg)
                if logger.hasHandlers():
                    logger.info(msg)
                break
    
    return train_losses, val_pile_accuracies, val_pile_f1_scores
