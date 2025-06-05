import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
from monai.metrics import ROCAUCMetric
from monai.transforms import Activations, AsDiscrete, Compose
from monai.data import decollate_batch
from sklearn.metrics import precision_recall_curve, roc_curve, auc, confusion_matrix, balanced_accuracy_score, matthews_corrcoef
from torch.utils.tensorboard import SummaryWriter


def save_feature_dict(feature_dict, pids, labels, bclcs, save_dir, logging, model_type="singlephase"):
    """Save extracted features to disk"""
    for i in range(len(pids)):
        save_data = {
            "PID": pids[i],
            "model": model_type,
            "label": labels[i].item(),
            "bclc": bclcs[i].item() if bclcs[i] is not None else None,
            "bag_feature": feature_dict["bag_feature"][i],
        }

        if model_type == "singlephase":
            save_data["patch_features"] = feature_dict["patch_features"][i]
            if "vit_patch_activations" in feature_dict:
                save_data["vit_patch_activations"] = feature_dict["vit_patch_activations"][i]
        elif model_type == "multiphase":
            save_data.update({
                "patch_features": feature_dict["fused_patch_features"][i],
                "pre_features": feature_dict["pre"][i],
                "art_features": feature_dict["art"][i],
                "pv_features": feature_dict["pv"][i],
                "delay_features": feature_dict["delay"][i],
            })
            if "vit_patch_activations" in feature_dict:
                save_data.update({
                    "pre_patch_activations": feature_dict["vit_patch_activations"]["pre"][i],
                    "art_patch_activations": feature_dict["vit_patch_activations"]["art"][i],
                    "pv_patch_activations": feature_dict["vit_patch_activations"]["pv"][i],
                    "delay_patch_activations": feature_dict["vit_patch_activations"]["delay"][i],
                })

        torch.save(save_data, os.path.join(save_dir, f"{pids[i]}_{model_type}.pt"))
    
    logging.info(f"Saved {model_type} features for {len(pids)} patients")

warmup_epochs = 10
def multiphase_warmup_lambda(epoch):
    print(f"Multiphase warmup lambda for epoch {epoch}: {min(1.0, float(epoch + 1) / warmup_epochs)}")
    return min(1.0, float(epoch + 1) / warmup_epochs)
def singlephase_warmup_lambda(epoch):
    print(f"Singlephase warmup lambda for epoch {epoch}: {min(1.0, float(epoch + 1) / warmup_epochs)}")
    return 1.0

def train_prototype_fusion_model(
    model, 
    train_mp_loader, 
    train_sp_loader, 
    val_mp_loader, 
    val_sp_loader,
    optimizer, 
    device, 
    args,
    multiphase_params
):
    """
    Train the prototype fusion model with multiphase-singlephase architecture.
    
    Args:
        model: Prototype fusion model with multiphase and singlephase components
        train_mp_loader: Training data loader for multi-phase data
        train_sp_loader: Training data loader for single-phase (PV) data
        val_mp_loader: Validation data loader for multi-phase data
        val_sp_loader: Validation data loader for single-phase (PV) data
        optimizer: Optimizer for training
        device: Computation device
        args: Training arguments
    """
    val_interval = 1
    best_metric = -1
    best_metric_epoch = -1
    
    # Metrics
    er_auc_metric = ROCAUCMetric()
    val_er_auc_metric = ROCAUCMetric()
    
    # Lists to store training metrics
    train_losses = []
    val_losses = []
    train_er_ce_losses = []
    train_proto_losses = []
    train_align_losses = []
    train_rank_losses = []
    train_multiphase_er_ce_losses = []
    train_multiphase_proto_losses = []
    train_multiphase_rank_losses = []
    train_multiphase_proto_sep_losses = []
    train_multiphase_combined_losses = []
    train_singlephase_er_ce_losses = []
    train_singlephase_proto_losses = []
    train_singlephase_align_losses = []
    train_singlephase_rank_losses = []
    train_singlephase_proto_sep_losses = []
    train_singlephase_combined_losses = []
    
    val_er_ce_losses = []
    val_proto_losses = []
    val_align_losses = []
    val_rank_losses = []
    val_multiphase_er_ce_losses = []
    val_multiphase_proto_losses = []
    val_multiphase_rank_losses = []
    val_multiphase_proto_sep_losses = []
    val_multiphase_combined_losses = []
    val_singlephase_er_ce_losses = []
    val_singlephase_proto_losses = []
    val_singlephase_align_losses = []
    val_singlephase_rank_losses = []
    val_singlephase_proto_sep_losses = []
    val_singlephase_combined_losses = []
    
    train_er_accuracies = []
    val_er_accuracies = []
    train_er_aurocs = []
    val_er_aurocs = []
    train_er_auprcs = []
    val_er_auprcs = []
    
    feature_save_dir = f"./output/features/"
    os.makedirs(feature_save_dir, exist_ok=True)
    
    # Early stopping setup
    early_stopping_enabled = getattr(args, 'early_stopping', False)
    if early_stopping_enabled:
        early_stopping_counter = 0
        early_stopping_patience = getattr(args, 'patience', 10)
        early_stopping_monitor = getattr(args, 'monitor_metric', 'accuracy')
        best_early_stopping_value = float('-inf') if early_stopping_monitor != 'loss' else float('inf')
        print(f"Early stopping enabled with patience {early_stopping_patience}, monitoring {early_stopping_monitor}")

    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=[multiphase_warmup_lambda, singlephase_warmup_lambda]
    )
    for i, group in enumerate(optimizer.param_groups):
        print(f"Current learning rate for param_group[{i}]: {group['lr']}")

    
    plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",          
        factor=0.5,             
        patience=3,         
        verbose=True,
    )
    for i, group in enumerate(optimizer.param_groups):
        print(f"Current learning rate for param_group[{i}]: {group['lr']}")
    
    # Transforms for metrics calculation
    post_pred = Compose([Activations(softmax=True)])
    post_label = Compose([AsDiscrete(to_onehot=args.num_classes)])
    
    # Training loop
    for epoch in range(args.epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{args.epochs}")
        
        model.train()
        epoch_loss = 0
        epoch_er_ce_loss = 0
        epoch_proto_loss = 0
        epoch_align_loss = 0
        epoch_rank_loss = 0
        epoch_multiphase_er_ce_loss = 0
        epoch_multiphase_proto_loss = 0
        epoch_multiphase_rank_loss = 0
        epoch_multiphase_proto_sep_loss = 0
        epoch_multiphase_combined_loss = 0
        epoch_singlephase_er_ce_loss = 0
        epoch_singlephase_proto_loss = 0
        epoch_singlephase_align_loss = 0
        epoch_singlephase_rank_loss = 0
        epoch_singlephase_proto_sep_loss = 0
        epoch_singlephase_combined_loss = 0
        step = 0
                     
        # Create iterators for both data loaders
        train_mp_iterator = iter(train_mp_loader)
        train_sp_iterator = iter(train_sp_loader)
        print(f"Training with {len(train_mp_loader)} multi-phase batches and {len(train_sp_loader)} single-phase batches")
        
        # Containers for predictions and targets
        y_er_pred = torch.tensor([], dtype=torch.float32, device=device)
        y_er = torch.tensor([], dtype=torch.long, device=device)
        
        for train_step in range(len(train_sp_loader)):
            sp_batch = next(train_sp_iterator)

            if train_step % (len(train_sp_loader) // len(train_mp_loader)) == 0:
                print(f"Training step {train_step} - loading multi-phase batch")
                try:
                    mp_batch = next(train_mp_iterator)
                except StopIteration:
                    mp_iter = iter(train_mp_loader)
                    mp_batch = next(mp_iter)
            else:
                mp_batch = None  
            
            print(f"Training step {train_step}")
            step += 1
            
            # Prepare single-phase input for singlephase model
            sp_img = sp_batch["PVimg"].to(device)
            sp_er_labels = sp_batch["label"].to(device)
            sp_bclc_labels = sp_batch["bclc"].to(device) if "bclc" in sp_batch else None
            
            # Forward pass
            if mp_batch is not None:
                multi_phase = tuple(t.to(device) for t in (mp_batch["preimg"], mp_batch["Aimg"], mp_batch["PVimg"], mp_batch["Delayimg"]))
            else:
                multi_phase = None
                
            print("Multi-phase is None or not:", multi_phase is None)
            
            inputs = {
                "multi_phase": multi_phase,
                "pv": sp_batch["PVimg"].to(device)
            }

            optimizer.zero_grad()
            outputs = model(inputs, sp_er_labels, sp_bclc_labels, training=True)
            
            # Extract losses
            combined_loss = outputs["combined_loss"]
            multiphase_losses = outputs["multiphase"]["losses"]
            singlephase_losses = outputs["singlephase"]["losses"]
            
            # Accumulate losses for logging
            er_ce_loss = multiphase_losses.get("er_ce_loss", 0) + singlephase_losses.get("er_ce_loss", 0)
            proto_loss = multiphase_losses.get("proto_loss", 0) + singlephase_losses.get("proto_loss", 0)
            align_loss = singlephase_losses.get("align_loss", 0)
            
            multiphase_er_ce_loss = multiphase_losses.get("er_ce_loss", 0)
            multiphase_proto_loss = multiphase_losses.get("proto_loss", 0)
            multiphase_rank_loss = multiphase_losses.get("rank_loss", 0)
            multiphase_proto_sep_loss = multiphase_losses.get("proto_sep_loss", 0)
            singlephase_er_ce_loss = singlephase_losses.get("er_ce_loss", 0)
            singlephase_proto_loss = singlephase_losses.get("proto_loss", 0)
            singlephase_align_loss = singlephase_losses.get("align_loss", 0)
            singlephase_rank_loss = singlephase_losses.get("rank_loss", 0)
            singlephase_proto_sep_loss = singlephase_losses.get("proto_sep_loss", 0)
            
            train_multiphase_combine_loss = 0.0
            train_singlephase_combine_loss = 0.0
            for loss_name, weight in model.loss_weights.items():
                train_multiphase_combine_loss += weight * multiphase_losses.get(loss_name, 0.0)
                train_singlephase_combine_loss += weight * singlephase_losses.get(loss_name, 0.0)
            
            rank_loss = 0
            if "rank_loss" in multiphase_losses:
                rank_loss += multiphase_losses["rank_loss"]
            if "rank_loss" in singlephase_losses:
                rank_loss += singlephase_losses["rank_loss"]
            
            # Backward and optimize
            combined_loss.backward()
            optimizer.step()
            
            # Accumulate losses
            epoch_loss += combined_loss.item() if isinstance(combined_loss, torch.Tensor) else combined_loss
            epoch_er_ce_loss += er_ce_loss.item() if isinstance(er_ce_loss, torch.Tensor) else er_ce_loss
            epoch_proto_loss += proto_loss.item() if isinstance(proto_loss, torch.Tensor) else proto_loss
            epoch_align_loss += align_loss.item() if isinstance(align_loss, torch.Tensor) else align_loss
            epoch_rank_loss += rank_loss.item() if isinstance(rank_loss, torch.Tensor) else rank_loss

            epoch_multiphase_er_ce_loss += multiphase_er_ce_loss.item() if isinstance(multiphase_er_ce_loss, torch.Tensor) else multiphase_er_ce_loss
            epoch_multiphase_proto_loss += multiphase_proto_loss.item() if isinstance(multiphase_proto_loss, torch.Tensor) else multiphase_proto_loss
            epoch_multiphase_rank_loss += multiphase_rank_loss.item() if isinstance(multiphase_rank_loss, torch.Tensor) else multiphase_rank_loss
            epoch_multiphase_proto_sep_loss += multiphase_proto_sep_loss.item() if isinstance(multiphase_proto_sep_loss, torch.Tensor) else multiphase_proto_sep_loss

            epoch_singlephase_er_ce_loss += singlephase_er_ce_loss.item() if isinstance(singlephase_er_ce_loss, torch.Tensor) else singlephase_er_ce_loss
            epoch_singlephase_proto_loss += singlephase_proto_loss.item() if isinstance(singlephase_proto_loss, torch.Tensor) else singlephase_proto_loss
            epoch_singlephase_align_loss += singlephase_align_loss.item() if isinstance(singlephase_align_loss, torch.Tensor) else singlephase_align_loss
            epoch_singlephase_rank_loss += singlephase_rank_loss.item() if isinstance(singlephase_rank_loss, torch.Tensor) else singlephase_rank_loss
            epoch_singlephase_proto_sep_loss += singlephase_proto_sep_loss.item() if isinstance(singlephase_proto_sep_loss, torch.Tensor) else singlephase_proto_sep_loss

            epoch_singlephase_combined_loss += train_singlephase_combine_loss.item() if isinstance(train_singlephase_combine_loss, torch.Tensor) else train_singlephase_combine_loss
            epoch_multiphase_combined_loss += train_multiphase_combine_loss.item() if isinstance(train_multiphase_combine_loss, torch.Tensor) else train_multiphase_combine_loss

            # Accumulate predictions for metrics (use singlephase model predictions)
            singlephase_er_logits = outputs["singlephase"]["er_logits"]
            y_er_pred = torch.cat([y_er_pred, singlephase_er_logits], dim=0)
            y_er = torch.cat([y_er, sp_er_labels], dim=0)
            

        # Calculate average losses
        epoch_loss /= step
        epoch_er_ce_loss /= step
        epoch_proto_loss /= step
        epoch_align_loss /= step
        epoch_rank_loss /= step if step > 0 else 1
        epoch_multiphase_er_ce_loss /= step
        epoch_multiphase_proto_loss /= step
        epoch_multiphase_rank_loss /= step
        epoch_multiphase_proto_sep_loss /= step
        epoch_singlephase_er_ce_loss /= step
        epoch_singlephase_proto_loss /= step
        epoch_singlephase_align_loss /= step
        epoch_singlephase_rank_loss /= step
        epoch_singlephase_proto_sep_loss /= step
        epoch_multiphase_combined_loss /= step
        epoch_singlephase_combined_loss /= step
    
        # Calculate metrics for ER prediction
        er_acc_value = torch.eq(y_er_pred.argmax(dim=1), y_er)
        er_acc_metric = er_acc_value.sum().item() / len(er_acc_value)
        
        # Calculate AUROC for ER
        y_er_onehot = [post_label(i) for i in decollate_batch(y_er, detach=False)]
        y_er_pred_act = [post_pred(i) for i in decollate_batch(y_er_pred)]
        y_er_pred_probs = F.softmax(y_er_pred, dim=1)
        er_auc_metric(y_er_pred_probs, y_er_onehot)
        er_auc_result = er_auc_metric.aggregate()
        er_auc_metric.reset()
        
        # Calculate additional metrics (sensitivity, specificity, etc.)
        y_er_np = y_er.detach().cpu().numpy()
        y_er_pred_class = y_er_pred.argmax(dim=1).detach().cpu().numpy()
        y_er_pred_prob = y_er_pred_probs.detach().cpu().numpy()
        
        cm = confusion_matrix(y_er_np, y_er_pred_class)
        if len(cm) > 1:  # Binary classification
            tn, fp, fn, tp = cm.ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            # Calculate AUPRC
            y_er_pred_prob = y_er_pred_prob[:, 1]  # Use positive class probability
            precision, recall, _ = precision_recall_curve(y_er_np, y_er_pred_prob)
            auprc = auc(recall, precision)
            
            # Calculate Balanced Accuracy and MCC
            bal_acc = balanced_accuracy_score(y_er_np, y_er_pred_class)
            if len(set(y_er_np)) > 1:
                mcc = matthews_corrcoef(y_er_np, y_er_pred_class)
            else:
                mcc = 0
        else:
            sensitivity = specificity = auprc = bal_acc = mcc = 0
        
        # Log training metrics
        print("-" * 10)
        print(f"epoch {epoch + 1} training average loss: {epoch_loss:.4f}")
        print(f"epoch {epoch + 1} Multiphase ER CE loss: {epoch_multiphase_er_ce_loss:.4f}")
        print(f"epoch {epoch + 1} Multiphase Prototype loss: {epoch_multiphase_proto_loss:.4f}")
        print(f"epoch {epoch + 1} Multiphase Rank loss: {epoch_multiphase_rank_loss:.4f}")
        print(f"epoch {epoch + 1} Multiphase Proto Sep loss: {epoch_multiphase_proto_sep_loss:.4f}")
        print(f"epoch {epoch + 1} Multiphase combined loss: {epoch_multiphase_combined_loss:.4f}")
        print(f"epoch {epoch + 1} Singlephase ER CE loss: {epoch_singlephase_er_ce_loss:.4f}")
        print(f"epoch {epoch + 1} Singlephase Prototype loss: {epoch_singlephase_proto_loss:.4f}")
        print(f"epoch {epoch + 1} Singlephase Alignment loss: {epoch_singlephase_align_loss:.4f}")
        print(f"epoch {epoch + 1} Singlephase Rank loss: {epoch_singlephase_rank_loss:.4f}")
        print(f"epoch {epoch + 1} Singlephase Proto Sep loss: {epoch_singlephase_proto_sep_loss:.4f}")
        print(f"epoch {epoch + 1} Singlephase combined loss: {epoch_singlephase_combined_loss:.4f}")
        print(f"epoch {epoch + 1} ER accuracy: {er_acc_metric:.4f}")
        print(f"epoch {epoch + 1} ER AUPRC: {auprc:.4f}")
        print(f"epoch {epoch + 1} ER AUROC: {er_auc_result:.4f}")
        print(f"epoch {epoch + 1} ER Sensitivity: {sensitivity:.4f}")
        print(f"epoch {epoch + 1} ER Specificity: {specificity:.4f}")
        print(f"epoch {epoch + 1} ER Balanced Accuracy: {bal_acc:.4f}")
        print(f"epoch {epoch + 1} ER MCC: {mcc:.4f}")
        
        # Store metrics for plotting
        train_losses.append(epoch_loss)
        train_er_ce_losses.append(epoch_er_ce_loss)
        train_proto_losses.append(epoch_proto_loss)
        train_align_losses.append(epoch_align_loss)
        train_rank_losses.append(epoch_rank_loss)
        train_multiphase_er_ce_losses.append(epoch_multiphase_er_ce_loss)
        train_multiphase_proto_losses.append(epoch_multiphase_proto_loss)
        train_multiphase_rank_losses.append(epoch_multiphase_rank_loss)
        train_multiphase_proto_sep_losses.append(epoch_multiphase_proto_sep_loss)
        train_singlephase_er_ce_losses.append(epoch_singlephase_er_ce_loss)
        train_singlephase_proto_losses.append(epoch_singlephase_proto_loss)
        train_singlephase_align_losses.append(epoch_singlephase_align_loss)
        train_singlephase_rank_losses.append(epoch_singlephase_rank_loss)
        train_singlephase_proto_sep_losses.append(epoch_singlephase_proto_sep_loss)
        train_singlephase_combined_losses.append(epoch_singlephase_combined_loss)
        train_multiphase_combined_losses.append(epoch_multiphase_combined_loss)
        
        train_er_accuracies.append(er_acc_metric)
        train_er_aurocs.append(er_auc_result)
        train_er_auprcs.append(auprc)
        

        if (epoch + 1) % 5 == 1:
            with torch.no_grad():
                model.eval()
                for val_mp_batch in val_mp_loader:
                    save_path = os.path.join(feature_save_dir, f"epoch_{epoch+1}")
                    os.makedirs(save_path, exist_ok=True)
                    inputs = {
                        "multi_phase": (
                            val_mp_batch["preimg"].to(device),
                            val_mp_batch["Aimg"].to(device),
                            val_mp_batch["PVimg"].to(device),
                            val_mp_batch["Delayimg"].to(device),
                        ),
                        "pv": None,
                    }
                    model(inputs, val_mp_batch["label"].to(device), val_mp_batch.get("bclc", None), training=False)
                    save_feature_dict(
                        model.multiphase_features_dict,
                        val_mp_batch["PID"],
                        val_mp_batch["label"],
                        val_mp_batch.get("bclc", None),
                        save_path,
                        print,
                        model_type="multiphase"
                    )

                for val_sp_batch in val_sp_loader:
                    save_path = os.path.join(feature_save_dir, f"epoch_{epoch+1}")
                    os.makedirs(save_path, exist_ok=True)
                    
                    inputs = {"multi_phase": None, "pv": val_sp_batch["PVimg"].to(device)}
                    model(inputs, val_sp_batch["label"].to(device), val_sp_batch.get("bclc", None), training=False)
                    save_feature_dict(
                        model.singlephase_features_dict,
                        val_sp_batch["PID"],
                        val_sp_batch["label"],
                        val_sp_batch.get("bclc", None),
                        save_path,
                        print,
                        model_type="singlephase"
                    )
            
        if (epoch + 1) % 5 == 1:
            proto_save_dir = os.path.join(feature_save_dir, f"epoch_{epoch+1}")
            os.makedirs(proto_save_dir, exist_ok=True)

            if hasattr(model, "multiphase_prototypes"):
                torch.save(model.multiphase_prototypes.detach().cpu(), os.path.join(proto_save_dir, "multiphase_prototypes.pt"))

            if hasattr(model, "singlephase_prototypes"):
                torch.save(model.singlephase_prototypes.detach().cpu(), os.path.join(proto_save_dir, "singlephase_prototypes.pt"))

        
        # Validation phase 
        if (epoch + 1) % val_interval == 0:
            model.eval()
            val_epoch_loss = 0
            val_epoch_er_ce_loss = 0
            val_epoch_proto_loss = 0
            val_epoch_align_loss = 0
            val_epoch_rank_loss = 0
            
            val_multiphase_er_ce_loss = 0
            val_multiphase_proto_loss = 0
            val_multiphase_rank_loss = 0
            val_multiphase_proto_sep_loss = 0
            val_multiphase_combined_loss = 0
            val_singlephase_er_ce_loss = 0
            val_singlephase_proto_loss = 0
            val_singlephase_align_loss = 0
            val_singlephase_rank_loss = 0
            val_singlephase_proto_sep_loss = 0
            val_singlephase_combined_loss = 0
            
            val_step = 0
            val_multiphase_step = 0
            val_singlephase_step = 0
            
            # Create iterators for validation data
            val_mp_iterator = iter(val_mp_loader)
            val_sp_iterator = iter(val_sp_loader)
            print(f"Validating with {len(val_mp_loader)} multi-phase batches and {len(val_sp_loader)} single-phase batches")
            # Number of validation iterations
            val_iterations = min(len(val_mp_loader), len(val_sp_loader))
            print(f"Validating for {val_iterations} iterations")
            
            # Containers for validation predictions and targets
            val_y_er_pred = torch.tensor([], dtype=torch.float32, device=device)
            val_y_er = torch.tensor([], dtype=torch.long, device=device)
            
            with torch.no_grad():
                # multiphase model validation
                print(f"Validating with {len(val_mp_loader)} multi-phase batches")
                for _ in range(len(val_mp_loader)):
                    try:
                        val_mp_batch = next(val_mp_iterator)
                    except StopIteration:
                        # If one of the iterators is exhausted, reset both
                        val_mp_iterator = iter(val_mp_loader)
                        val_mp_batch = next(val_mp_iterator)
                    
                    val_multiphase_step += 1
                    
                    # Prepare multi-phase inputs for multiphase model
                    val_pre_img = val_mp_batch["preimg"].to(device)
                    val_a_img = val_mp_batch["Aimg"].to(device)
                    val_pv_img = val_mp_batch["PVimg"].to(device)
                    val_delay_img = val_mp_batch["Delayimg"].to(device)
                    val_mp_er_labels = val_mp_batch["label"].to(device)
                    val_mp_bclc_labels = val_mp_batch["bclc"].to(device) if "bclc" in val_mp_batch else None
                    
                    # Forward pass
                    val_inputs = {
                        "multi_phase": (val_pre_img, val_a_img, val_pv_img, val_delay_img),
                        "pv": None
                    }
                    
                    val_outputs = model(val_inputs, val_mp_er_labels, val_mp_bclc_labels, training=False)

                    val_multiphase_losses = val_outputs["multiphase"]["losses"]
                    
                    for loss_name, weight in model.loss_weights.items():
                        val_multiphase_combined_loss += weight * val_multiphase_losses.get(loss_name, 0.0)
                    
                    val_multiphase_er_ce_loss += val_multiphase_losses.get("er_ce_loss", 0)
                    val_multiphase_proto_loss += val_multiphase_losses.get("proto_loss", 0)
                    val_multiphase_rank_loss += val_multiphase_losses.get("rank_loss", 0)
                    val_multiphase_proto_sep_loss += val_multiphase_losses.get("proto_sep_loss", 0)
                
                # singlephase model validation
                print(f"Validating with {len(val_sp_loader)} single-phase batches")
                for _ in range(len(val_sp_loader)):
                    try:
                        val_sp_batch = next(val_sp_iterator)
                    except StopIteration:
                        # If one of the iterators is exhausted, reset both
                        val_sp_iterator = iter(val_sp_loader)
                        val_sp_batch = next(val_sp_iterator)
                    
                    val_singlephase_step += 1
                    
                    # Prepare single-phase input for singlephase model
                    val_sp_img = val_sp_batch["PVimg"].to(device)
                    val_sp_er_labels = val_sp_batch["label"].to(device)
                    val_sp_bclc_labels = val_sp_batch["bclc"].to(device) if "bclc" in val_sp_batch else None
                    
                    # Forward pass
                    val_inputs = {
                        "multi_phase": None,
                        "pv": val_sp_img
                    }
                    
                    val_outputs = model(val_inputs, val_sp_er_labels, val_sp_bclc_labels, training=False)
                    
                    val_singlephase_losses = val_outputs["singlephase"]["losses"]
                    
                    for loss_name, weight in model.loss_weights.items():
                        val_singlephase_combined_loss += weight * val_singlephase_losses.get(loss_name, 0.0)

                    # Accumulate losses for logging
                    val_singlephase_er_ce_loss += val_singlephase_losses.get("er_ce_loss", 0)
                    val_singlephase_proto_loss += val_singlephase_losses.get("proto_loss", 0)
                    val_singlephase_align_loss += val_singlephase_losses.get("align_loss", 0)
                    val_singlephase_rank_loss += val_singlephase_losses.get("rank_loss", 0)
                    val_singlephase_proto_sep_loss += val_singlephase_losses.get("proto_sep_loss", 0)
                    
                    # Accumulate predictions for metrics (use singlephase model predictions)
                    val_singlephase_er_logits = val_outputs["singlephase"]["er_logits"]
                    val_y_er_pred = torch.cat([val_y_er_pred, val_singlephase_er_logits], dim=0)
                    val_y_er = torch.cat([val_y_er, val_sp_er_labels], dim=0)
                    
                
                # Calculate average validation losses
                val_multiphase_er_ce_loss /= val_multiphase_step
                val_multiphase_proto_loss /= val_multiphase_step
                val_multiphase_rank_loss /= val_multiphase_step
                val_multiphase_proto_sep_loss /= val_multiphase_step
                val_multiphase_combined_loss /= val_multiphase_step
                
                val_singlephase_er_ce_loss /= val_singlephase_step
                val_singlephase_proto_loss /= val_singlephase_step
                val_singlephase_align_loss /= val_singlephase_step
                val_singlephase_rank_loss /= val_singlephase_step
                val_singlephase_proto_sep_loss /= val_singlephase_step
                val_singlephase_combined_loss /= val_singlephase_step
                
                # Calculate ER metrics for validation
                val_er_acc_value = torch.eq(val_y_er_pred.argmax(dim=1), val_y_er)
                val_er_acc_metric = val_er_acc_value.sum().item() / len(val_er_acc_value)
                
                # Calculate AUROC for ER
                val_y_er_onehot = [post_label(i) for i in decollate_batch(val_y_er, detach=False)]
                val_y_er_pred_act = [post_pred(i) for i in decollate_batch(val_y_er_pred)]
                val_y_er_pred_probs = F.softmax(val_y_er_pred, dim=1)
                val_er_auc_metric(val_y_er_pred_act, val_y_er_onehot)
                val_er_auc_result = val_er_auc_metric.aggregate()
                val_er_auc_metric.reset()
                
                warmup_scheduler.step()
                plateau_scheduler.step(val_singlephase_combined_loss)

                
                # Calculate additional validation metrics
                val_y_er_np = val_y_er.detach().cpu().numpy()
                val_y_er_pred_class = val_y_er_pred.argmax(dim=1).detach().cpu().numpy()
                val_y_er_pred_prob = val_y_er_pred_probs.detach().cpu().numpy()
                
                if val_y_er_pred_prob.shape[1] > 1:
                    val_y_er_pred_prob = val_y_er_pred_prob[:, 1]  # Use positive class probability
                
                val_cm = confusion_matrix(val_y_er_np, val_y_er_pred_class)
                if len(val_cm) > 1:  # Binary classification
                    val_tn, val_fp, val_fn, val_tp = val_cm.ravel()
                    val_sensitivity = val_tp / (val_tp + val_fn) if (val_tp + val_fn) > 0 else 0
                    val_specificity = val_tn / (val_tn + val_fp) if (val_tn + val_fp) > 0 else 0
                    
                    # Calculate AUPRC
                    val_precision, val_recall, _ = precision_recall_curve(val_y_er_np, val_y_er_pred_prob)
                    val_auprc = auc(val_recall, val_precision)
                    
                    # Calculate Balanced Accuracy and MCC
                    val_bal_acc = balanced_accuracy_score(val_y_er_np, val_y_er_pred_class)
                    
                    if len(set(val_y_er_np)) > 1:
                        val_mcc = matthews_corrcoef(val_y_er_np, val_y_er_pred_class)
                    else:
                        val_mcc = 0
                else:
                    # Handle case when only one class is present
                    val_sensitivity = val_specificity = val_auprc = val_bal_acc = val_mcc = 0
                
                # Log validation metrics
                print("-" * 10)
                print(f"epoch {epoch + 1} validation multiphase ER CE loss: {val_multiphase_er_ce_loss:.4f}")
                print(f"epoch {epoch + 1} validation multiphase Prototype loss: {val_multiphase_proto_loss:.4f}")
                print(f"epoch {epoch + 1} validation multiphase Rank loss: {val_multiphase_rank_loss:.4f}")
                print(f"epoch {epoch + 1} validation multiphase Proto Sep loss: {val_multiphase_proto_sep_loss:.4f}")
                print(f"epoch {epoch + 1} validation multiphase Combine loss: {val_multiphase_combined_loss:.4f}")
                print(f"epoch {epoch + 1} validation singlephase ER CE loss: {val_singlephase_er_ce_loss:.4f}")
                print(f"epoch {epoch + 1} validation singlephase Prototype loss: {val_singlephase_proto_loss:.4f}")
                print(f"epoch {epoch + 1} validation singlephase Alignment loss: {val_singlephase_align_loss:.4f}")
                print(f"epoch {epoch + 1} validation singlephase Rank loss: {val_singlephase_rank_loss:.4f}")
                print(f"epoch {epoch + 1} validation singlephase Proto Sep loss: {val_singlephase_proto_sep_loss:.4f}")
                print(f"epoch {epoch + 1} validation singlephase Combine loss: {val_singlephase_combined_loss:.4f}")
                print(f"epoch {epoch + 1} validation ER accuracy: {val_er_acc_metric:.4f}")
                print(f"epoch {epoch + 1} validation ER AUPRC: {val_auprc:.4f}")
                print(f"epoch {epoch + 1} validation ER AUROC: {val_er_auc_result:.4f}")
                print(f"epoch {epoch + 1} validation ER Sensitivity: {val_sensitivity:.4f}")
                print(f"epoch {epoch + 1} validation ER Specificity: {val_specificity:.4f}")
                print(f"epoch {epoch + 1} validation ER Balanced Accuracy: {val_bal_acc:.4f}")
                print(f"epoch {epoch + 1} validation ER MCC: {val_mcc:.4f}")
                
                # Store validation metrics for plotting
                val_epoch_er_ce_loss = val_singlephase_er_ce_loss  
                val_epoch_proto_loss = val_singlephase_proto_loss + val_multiphase_proto_loss
                val_epoch_align_loss = val_singlephase_align_loss
                val_epoch_rank_loss = val_singlephase_rank_loss + val_multiphase_rank_loss
                

                val_losses.append(val_epoch_loss.item() if isinstance(val_epoch_loss, torch.Tensor) else val_epoch_loss)
                val_er_ce_losses.append(val_epoch_er_ce_loss.item() if isinstance(val_epoch_er_ce_loss, torch.Tensor) else val_epoch_er_ce_loss)
                val_proto_losses.append(val_epoch_proto_loss.item() if isinstance(val_epoch_proto_loss, torch.Tensor) else val_epoch_proto_loss)
                val_align_losses.append(val_epoch_align_loss.item() if isinstance(val_epoch_align_loss, torch.Tensor) else val_epoch_align_loss)
                val_rank_losses.append(val_epoch_rank_loss.item() if isinstance(val_epoch_rank_loss, torch.Tensor) else val_epoch_rank_loss)
                val_multiphase_er_ce_losses.append(val_multiphase_er_ce_loss.item() if isinstance(val_multiphase_er_ce_loss, torch.Tensor) else val_multiphase_er_ce_loss)
                val_multiphase_proto_losses.append(val_multiphase_proto_loss.item() if isinstance(val_multiphase_proto_loss, torch.Tensor) else val_multiphase_proto_loss)
                val_multiphase_rank_losses.append(val_multiphase_rank_loss.item() if isinstance(val_multiphase_rank_loss, torch.Tensor) else val_multiphase_rank_loss)
                val_multiphase_proto_sep_losses.append(val_multiphase_proto_sep_loss.item() if isinstance(val_multiphase_proto_sep_loss, torch.Tensor) else val_multiphase_proto_sep_loss)
                val_singlephase_er_ce_losses.append(val_singlephase_er_ce_loss.item() if isinstance(val_singlephase_er_ce_loss, torch.Tensor) else val_singlephase_er_ce_loss)
                val_singlephase_proto_losses.append(val_singlephase_proto_loss.item() if isinstance(val_singlephase_proto_loss, torch.Tensor) else val_singlephase_proto_loss)
                val_singlephase_align_losses.append(val_singlephase_align_loss.item() if isinstance(val_singlephase_align_loss, torch.Tensor) else val_singlephase_align_loss)  
                val_singlephase_rank_losses.append(val_singlephase_rank_loss.item() if isinstance(val_singlephase_rank_loss, torch.Tensor) else val_singlephase_rank_loss)
                val_singlephase_proto_sep_losses.append(val_singlephase_proto_sep_loss.item() if isinstance(val_singlephase_proto_sep_loss, torch.Tensor) else val_singlephase_proto_sep_loss)
                val_singlephase_combined_losses.append(val_singlephase_combined_loss.item() if isinstance(val_singlephase_combined_loss, torch.Tensor) else val_singlephase_combined_loss)
                val_multiphase_combined_losses.append(val_multiphase_combined_loss.item() if isinstance(val_multiphase_combined_loss, torch.Tensor) else val_multiphase_combined_loss)
                
                val_er_accuracies.append(val_er_acc_metric)
                val_er_aurocs.append(val_er_auc_result)
                val_er_auprcs.append(val_auprc)
                                
                # Check early stopping criteria
                if early_stopping_enabled:
                    current_value = None
                    improved = False
                    
                    if early_stopping_monitor == 'loss':
                        current_value = val_singlephase_combined_loss.item() if isinstance(val_singlephase_combined_loss, torch.Tensor) else val_singlephase_combined_loss 
                        improved = current_value < best_early_stopping_value
                    elif early_stopping_monitor == 'accuracy':
                        current_value = val_er_acc_metric
                        improved = current_value > best_early_stopping_value
                    elif early_stopping_monitor == 'auroc':
                        current_value = val_er_auc_result
                        improved = current_value > best_early_stopping_value
                    elif early_stopping_monitor == 'auprc':
                        current_value = val_auprc
                        improved = current_value > best_early_stopping_value
                    
                    if improved:
                        best_early_stopping_value = current_value
                        early_stopping_counter = 0
                        print(f"Early stopping monitor ({early_stopping_monitor}) improved: {best_early_stopping_value:.4f}")
                    else:
                        early_stopping_counter += 1
                        print(f"Early stopping counter: {early_stopping_counter}/{early_stopping_patience}")
                        
                        if early_stopping_counter >= early_stopping_patience:
                            print(f"Early stopping triggered at epoch {epoch + 1}")
                            break
                
                # Save best model based on AUPRC
                if val_auprc >= best_metric:
                    best_metric = val_auprc
                    best_metric_epoch = epoch + 1
                    save_path = f"./output/models"
                    os.makedirs(save_path, exist_ok=True)
                    torch.save(model.state_dict(), f"{save_path}/best_model.pth")

                    model.eval()
                    with torch.no_grad():
                        for val_mp_batch in val_mp_loader:
                            save_path = os.path.join(feature_save_dir, f"epoch_{best_metric_epoch}_best")
                            os.makedirs(save_path, exist_ok=True)
                            inputs = {
                                "multi_phase": (
                                    val_mp_batch["preimg"].to(device),
                                    val_mp_batch["Aimg"].to(device),
                                    val_mp_batch["PVimg"].to(device),
                                    val_mp_batch["Delayimg"].to(device),
                                ),
                                "pv": None,
                            }
                            model(inputs, val_mp_batch["label"].to(device), val_mp_batch.get("bclc", None), training=False)
                            save_feature_dict(model.multiphase_features_dict, val_mp_batch["PID"], val_mp_batch["label"], val_mp_batch.get("bclc", None), save_path, print, model_type="multiphase")
                        

                        for val_sp_batch in val_sp_loader:
                            save_path = os.path.join(feature_save_dir, f"epoch_{best_metric_epoch}_best")
                            os.makedirs(save_path, exist_ok=True)
                            inputs = {"multi_phase": None, "pv": val_sp_batch["PVimg"].to(device)}
                            model(inputs, val_sp_batch["label"].to(device), val_sp_batch.get("bclc", None), training=False)
                            save_feature_dict(model.singlephase_features_dict, val_sp_batch["PID"], val_sp_batch["label"], val_sp_batch.get("bclc", None), save_path, print, model_type="singlephase")
                            
                    
                    proto_save_dir = os.path.join(feature_save_dir, f"epoch_{epoch+1}")
                    os.makedirs(proto_save_dir, exist_ok=True)

                    if hasattr(model, "multiphase_prototypes"):
                        torch.save(model.multiphase_prototypes.detach().cpu(), os.path.join(proto_save_dir, "multiphase_prototypes.pt"))

                    if hasattr(model, "singlephase_prototypes"):
                        torch.save(model.singlephase_prototypes.detach().cpu(), os.path.join(proto_save_dir, "singlephase_prototypes.pt"))
                        
                
                print(f"Best AUPRC: {best_metric:.4f} at epoch: {best_metric_epoch}")
            
        
        # Check if early stopping was triggered
        if early_stopping_enabled and early_stopping_counter >= early_stopping_patience:
            print(f"Training stopped early at epoch {epoch + 1}")
            break
    
    # Get the actual number of epochs trained (for plotting)
    actual_epochs = len(train_losses)
    epoch_range = range(1, actual_epochs + 1)
    
    # Load the best model for evaluation
    model.load_state_dict(torch.load(f"./output/models/best_model.pth"))
    print(f"Training completed, best AUPRC: {best_metric:.4f} at epoch: {best_metric_epoch}")
    
    # Create directory for saving plots
    save_dir = f"./output/plots/"
    os.makedirs(save_dir, exist_ok=True)
    

    plt.figure(figsize=(18, 12))
    plt.subplot(2, 3, 1)
    plt.plot(epoch_range, train_multiphase_er_ce_losses, label='Train Multiphase ER CE Loss')
    plt.plot(epoch_range, val_multiphase_er_ce_losses, label='Validation Multiphase ER CE Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Multiphase ER Cross Entropy Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.subplot(2, 3, 2)
    plt.plot(epoch_range, train_multiphase_proto_losses, label='Train Multiphase Prototype Loss')
    plt.plot(epoch_range, val_multiphase_proto_losses, label='Validation Multiphase Prototype Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Multiphase Prototype Contrastive Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.subplot(2, 3, 3)
    plt.plot(epoch_range, train_multiphase_proto_sep_losses, label='Train Multiphase Proto Sep Loss')
    plt.plot(epoch_range, val_multiphase_proto_sep_losses, label='Validation Multiphase Proto Sep Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Multiphase Prototype Separation Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    
    plt.subplot(2, 3, 4)
    plt.plot(epoch_range, train_multiphase_rank_losses, label='Train Multiphase Rank Loss')
    plt.plot(epoch_range, val_multiphase_rank_losses, label='Validation Multiphase Rank Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Multiphase Rank Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.subplot(2, 3, 5)
    plt.plot(epoch_range, train_multiphase_combined_losses, label='Train Multiphase Combined Loss')
    plt.plot(epoch_range, val_multiphase_combined_losses, label='Validation Multiphase Combined Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Multiphase Combined Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "multiphase_losses.png"))
    plt.close()

    plt.figure(figsize=(18, 12))
    plt.subplot(2, 3, 1)
    plt.plot(epoch_range, train_singlephase_er_ce_losses, label='Train Singlephase ER CE Loss')
    plt.plot(epoch_range, val_singlephase_er_ce_losses, label='Validation Singlephase ER CE Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Singlephase ER Cross Entropy Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.subplot(2, 3, 2)
    plt.plot(epoch_range, train_singlephase_proto_losses, label='Train Singlephase Prototype Loss')
    plt.plot(epoch_range, val_singlephase_proto_losses, label='Validation Singlephase Prototype Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Singlephase Prototype Contrastive Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.subplot(2, 3, 3)
    plt.plot(epoch_range, train_singlephase_proto_sep_losses, label='Train Singlephase Proto Sep Loss')
    plt.plot(epoch_range, val_singlephase_proto_sep_losses, label='Validation Singlephase Proto Sep Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Singlephase Prototype Separation Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.subplot(2, 3, 4)
    plt.plot(epoch_range, train_singlephase_rank_losses, label='Train Singlephase Rank Loss')
    plt.plot(epoch_range, val_singlephase_rank_losses, label='Validation Singlephase Rank Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Singlephase Rank Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.subplot(2, 3, 5)
    plt.plot(epoch_range, train_singlephase_align_losses, label='Train Singlephase Alignment Loss')
    plt.plot(epoch_range, val_singlephase_align_losses, label='Validation Singlephase Alignment Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Singlephase Alignment Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.subplot(2, 3, 6)
    plt.plot(epoch_range, train_singlephase_combined_losses, label='Train Singlephase Combined Loss')
    plt.plot(epoch_range, val_singlephase_combined_losses, label='Validation Singlephase Combined Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Singlephase Combined Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)    
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "singlephase_losses.png"))
    plt.close()
    
    
    # Plot ER metrics
    plt.figure(figsize=(18, 6))
    plt.subplot(1, 3, 1)
    plt.plot(epoch_range, train_er_accuracies, label='Train Accuracy')
    plt.plot(epoch_range, val_er_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('ER Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.subplot(1, 3, 2)
    plt.plot(epoch_range, train_er_aurocs, label='Train AUROC')
    plt.plot(epoch_range, val_er_aurocs, label='Validation AUROC')
    plt.xlabel('Epochs')
    plt.ylabel('AUROC')
    plt.title('ER Area Under ROC Curve')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.subplot(1, 3, 3)
    plt.plot(epoch_range, train_er_auprcs, label='Train AUPRC')
    plt.plot(epoch_range, val_er_auprcs, label='Validation AUPRC')
    plt.xlabel('Epochs')
    plt.ylabel('AUPRC')
    plt.title('ER Area Under Precision-Recall Curve')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "er_metrics.png"))
    plt.close()
    

    return model
