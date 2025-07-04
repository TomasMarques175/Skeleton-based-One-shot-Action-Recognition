#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 18:10:29 2020

@author: asabater
"""

# --- Imports ---
import os
import random
import torch
import torch.nn as nn
# import torch.nn.functional as F # TCN_classifier might use it
import torch.optim as optim
from torch.utils.data import DataLoader # Dataset is imported from pytorch_dataset
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
from torch.cuda.amp import autocast, GradScaler

from scipy.special import comb # Used by get_num_feats
import numpy as np
import json
from shutil import copyfile
import glob # For file searching if adapting remove_suboptimal_weights
import time # For timing epochs
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score

# --- PyTorch specific imports ---
from models.TCN_classifier import TCN_clf # Your PyTorch model
from pytorch_dataset import TripletPoseDataset # Your PyTorch Dataset


# --- Framework-agnostic or to-be-adapted utility imports ---
# Ensure these functions do not have hard TensorFlow dependencies.
# It's assumed these are available in your environment, possibly from data_generator.py
# If data_generator.py has TF dependencies, these functions need to be extracted/rewritten.
try:
    from data_generator import get_scaler_filename, get_num_feats
except ImportError:
    print("Warning: Could not import from data_generator.py. Ensure get_scaler_filename and get_num_feats are defined or TF-free.")
    # Define placeholders if they are critical and not available, but this is not ideal
    def get_num_feats(**kwargs): raise NotImplementedError("get_num_feats is not defined/imported")
    def get_scaler_filename(**kwargs): raise NotImplementedError("get_scaler_filename is not defined/imported")

try:
    import train_utils # For create_model_folder
except ImportError:
    print("Warning: train_utils not found. create_model_folder might fail.")
    class train_utils_placeholder: # Basic placeholder
        @staticmethod
        def create_model_folder(path_results, model_name_base):
            from datetime import datetime
            ts = datetime.now().strftime("%m%d_%H%M%S")
            model_name_folder = f"{str(model_name_base)}_{ts}_run{np.random.randint(1000):03d}"
            path_model = os.path.join(path_results, model_name_folder)
            return path_model
    train_utils = train_utils_placeholder


# Seed PyTorch
SEED = 123
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)

torch.backends.cudnn.benchmark = False


def init_weights_constant(m):
    if hasattr(m, 'weight') and m.weight is not None:
        nn.init.constant_(m.weight, 0.01)
    if hasattr(m, 'bias') and m.bias is not None:
        nn.init.constant_(m.bias, 0.01)


def main(model_params):
    train_verbose = model_params.get('train_verbose', 1)
    log_interval = model_params.get('log_interval', 10)
    
    # --- Path and Feature Calculation (Cleaned Up) ---
    print("--- Initializing Parameters and Paths ---")
    model_params['path_model'] = train_utils.create_model_folder(
        model_params['path_results'], model_params['model_name']
    )
    os.makedirs(model_params['path_model'], exist_ok=True)

    # Calculate num_jcd_feats and add to model_params (for record-keeping and if get_num_feats uses it)
    try:
        model_params['num_jcd_feats'] = int(comb(model_params.get('joints_num', 25), 2))
    except Exception as e:
        print(f"Warning: Could not calculate num_jcd_feats using comb: {e}. Setting to None or a fallback.")
        model_params['num_jcd_feats'] = None # Or a fallback value if critical

    # Calculate final num_feats using get_num_feats
    try:
        calculated_num_feats = get_num_feats(**model_params)
        # Check if a 'num_feats' was pre-set in model_params and if it differs.
        # The value from get_num_feats should be authoritative.
        if model_params.get('num_feats') != calculated_num_feats and model_params.get('num_feats') is not None:
             print(f"Warning: Initial model_params['num_feats'] ({model_params.get('num_feats')}) "
                   f"differs from get_num_feats() calculation ({calculated_num_feats}). "
                   f"Using calculated value from get_num_feats: {calculated_num_feats}.")
        model_params['num_feats'] = calculated_num_feats
    except Exception as e:
        print(f"CRITICAL Error calling get_num_feats: {e}. num_feats might be incorrect.")
        if 'num_feats' not in model_params or model_params['num_feats'] is None:
            # If get_num_feats fails and no num_feats is set, it's a critical issue.
            print("CRITICAL: num_feats is not set and get_num_feats failed. Aborting.")
            return # Exit if essential num_feats cannot be determined

    # Save the complete model_params to JSON
    model_params_save_path = os.path.join(model_params['path_model'], 'model_params.json')
    try:
        with open(model_params_save_path, 'w') as f:
            json.dump(model_params, f, indent=4)
        print(f"Saved model parameters to {model_params_save_path}")
    except Exception as e:
        print(f"Error saving model_params.json: {e}")

    print(' * Final Model params for this run:', json.dumps(model_params, indent=2))

    # --- Annotation File Counts (for informational purposes) ---
    num_train_files, num_val_files = 0, 0
    try:
        if model_params.get('train_annotations'):
            with open(model_params['train_annotations'], 'r') as f:
                num_train_files = len(f.read().splitlines())
        if model_params.get('val_annotations'):
            with open(model_params['val_annotations'], 'r') as f:
                num_val_files = len(f.read().splitlines())
        print(f"Num train annotation lines: {num_train_files}, Num val annotation lines: {num_val_files}")
    except FileNotFoundError as e:
        print(f"Warning: Annotation file not found: {e}. Counts will be 0.")
    except Exception as e:
        print(f"Warning: Error reading annotation files: {e}. Counts might be inaccurate.")

    # --- Scaler File Copying ---
    if model_params.get('scale_data', False):
        try:
            scaler_filename_src = get_scaler_filename(**model_params)
            if scaler_filename_src: # Ensure a filename was returned
                scaler_filename_dst = os.path.join(model_params['path_model'], 'scaler.pckl')
                if os.path.exists(scaler_filename_src):
                    copyfile(scaler_filename_src, scaler_filename_dst)
                    print(f"Copied scaler: {scaler_filename_src} to {scaler_filename_dst}")
                else:
                    print(f"Warning: Scaler source file '{scaler_filename_src}' not found. Not copied.")
            else:
                print("Warning: get_scaler_filename did not return a path. Scaler not copied.")
        except Exception as e:
            print(f"Warning: Error handling scaler file: {e}. Skipping scaler copy.")
    
    
    # --- PyTorch Model Instantiation ---
    print('\n * Setting model parameters (PyTorch)') # Mimicking Keras log
    # Ensure num_classes is correctly derived if dataset re-indexing happened,
    # though for this debug block, it might not matter if we don't train.
    num_classes_for_model = model_params.get('actual_num_classes', model_params['num_classes'])
    model = TCN_clf(
        num_feats=model_params['num_feats'], conv_params=model_params['conv_params'],
        lstm_dropout=model_params['lstm_dropout'], masking=model_params['masking'],
        triplet=model_params.get('triplet', False), classification=model_params.get('classification', True),
        clf_neurons=model_params['clf_neurons'], num_classes=num_classes_for_model
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"\n* Model moved to device: {device}\n")

    # --- Keras-like Debugging Block ---
    print('\n* Building model (PyTorch does not require explicit build)\n')
    # No model.build() in PyTorch, it's built on first forward pass or by knowing input shape

    print('\n* Initializing inputs and outputs (PyTorch)')
    # Use parameters from model_params for dummy input size
    # Keras dummy_inpt was (batch_size, max(abs(max_seq_len), 123), num_feats)
    dummy_batch_size = model_params.get('batch_size', 32)
    dummy_seq_len_abs = abs(model_params.get('max_seq_len', -32))
    dummy_seq_len = max(dummy_seq_len_abs if dummy_seq_len_abs > 0 else 32, 123) # Match Keras logic
    dummy_num_feats = model_params['num_feats']
    print(f"\n* Dummy input shape: (batch_size={dummy_batch_size}, seq_len={dummy_seq_len}, num_feats={dummy_num_feats})")
    dummy_inpt_np = np.random.rand(dummy_batch_size, dummy_seq_len, dummy_num_feats).astype(np.float32)
    print(f"\n* Dummy input shape (NumPy): {dummy_inpt_np.shape}")

    # Convert NumPy to PyTorch tensor
    dummy_inpt_torch = torch.from_numpy(dummy_inpt_np).to(device)

    model.eval() # Set model to evaluation mode for consistent output
    with torch.no_grad(): # Disable gradients for this test
        # PyTorch model call (equivalent to Keras model(dummy_inpt))
        dummy_pred_torch_call = model(dummy_inpt_torch)
        if isinstance(dummy_pred_torch_call, list):
            print(' * dummy_pred_torch_call shape (PyTorch model(input)):', [p.shape for p in dummy_pred_torch_call])
        elif isinstance(dummy_pred_torch_call, torch.Tensor):
            print(' * dummy_pred_torch_call shape (PyTorch model(input)):', dummy_pred_torch_call.shape)
        else:
            print(' * dummy_pred_torch_call type (PyTorch model(input)):', type(dummy_pred_torch_call))

        # PyTorch get_embedding call
        # Ensure get_embedding also takes a PyTorch tensor
        dummy_emb_torch = model.get_embedding(dummy_inpt_torch)
        print(' * dummy_emb_torch shape (PyTorch get_embedding(input)):', dummy_emb_torch.shape)

    # If model returns a single tensor instead of a list
    if isinstance(dummy_pred_torch_call, torch.Tensor):
        dummy_pred_torch_call = [dummy_pred_torch_call]

    # Save each full output (not each sample)
    for i, out in enumerate(dummy_pred_torch_call):
        np.save(f'pytorch_output_{i}.npy', out.cpu().numpy())
        # Optional readable version
        reshaped = out.view(out.size(0), -1).cpu().numpy()
        np.savetxt(f'pytorch_output_{i}.txt', reshaped)


    # --- PyTorch Optimizer and Loss (Mimicking Keras printouts) ---
    print('\n * Setting optimizer (PyTorch)')
    optimizer = optim.Adam(model.parameters(), lr=model_params['init_lr'])
    # Note: clipnorm is applied manually in PyTorch training loop (torch.nn.utils.clip_grad_norm_)
    print(f"   Optimizer: {type(optimizer)}, LR: {model_params['init_lr']}")

    print(' * Defining losses and loss_weights (PyTorch)')
    active_losses = {}
    loss_weights_pytorch_pt = {}
    if model_params.get('classification', True):
        criterion_clf = nn.CrossEntropyLoss()
        active_losses['classification'] = criterion_clf
        # Match Keras loss_weights['output_1'] = 0.4 if classification is the primary/first output
        loss_weights_pytorch_pt['classification'] = model_params.get('clf_loss_weight', 0.4)
    if model_params.get('triplet', False): # If you also had triplet loss in Keras
        criterion_triplet_pt = nn.TripletMarginLoss(margin=model_params.get('triplet_margin', 1.0))
        active_losses['triplet'] = criterion_clf
        loss_weights_pytorch_pt['triplet'] = model_params.get('triplet_loss_weight', 0.6) # Example

    print(' * losses (PyTorch types):', active_losses)
    print(' * loss_weights (PyTorch):', loss_weights_pytorch_pt)
    # sample_weights_mode is not a direct PyTorch concept, handled manually if needed


    # --- PyTorch Model Summary ---
    print('\n * Model summary (PyTorch - torchinfo)')
    try:
        # Use the dummy input shape for summary
        summary_input_shape_for_debug = (dummy_batch_size, dummy_seq_len, dummy_num_feats)
        print(f"* Using input shape for summary: {summary_input_shape_for_debug}\n")
        summary(model, input_size=summary_input_shape_for_debug, col_names=["input_size", "output_size", "num_params", "kernel_size"], verbose=1)
    except Exception as e:
        print(f"torchinfo summary failed: {e}.\nBasic model structure:\n{model}")

    tb_log_dir = os.path.join(model_params['path_model'], 'tensorboard_logs')
    os.makedirs(tb_log_dir, exist_ok=True)
    tb_writer = SummaryWriter(log_dir=tb_log_dir)
    print(f"TensorBoard: Logging to {tb_log_dir}")

    monitor_metric_name = model_params.get('monitor', 'val_loss')
    monitor_is_loss = 'loss' in monitor_metric_name.lower()
    # Keras min_monitor=False means maximize. PyTorch mode='max'.
    # Keras min_monitor=True means minimize. PyTorch mode='min'.
    # Default to minimizing if loss, maximizing otherwise, unless min_monitor overrides.
    default_mode_for_metric = 'min' if monitor_is_loss else 'max'
    if 'min_monitor' in model_params: # User explicitly set min_monitor
        monitor_mode = 'min' if model_params['min_monitor'] else 'max'
    else: # Infer from metric name
        monitor_mode = default_mode_for_metric

    best_monitor_metric_val = float('-inf') if monitor_mode == 'max' else float('inf')
    weights_save_path = os.path.join(model_params['path_model'], 'weights')
    os.makedirs(weights_save_path, exist_ok=True)

    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=monitor_mode,
                                                        factor=model_params.get('lr_factor', 0.1),
                                                        patience=model_params.get('lr_patience', 3),
                                                        verbose=True, min_lr=model_params.get('min_lr', 1e-7))
    early_stopping_patience = model_params.get('es_patience', 6)
    early_stopping_counter = 0
    best_val_for_early_stop = float('-inf') if monitor_mode == 'max' else float('inf')


    # --- Data Loading ---
    print("\n--- Setting up PyTorch DataLoaders ---")
    dataset_params_for_loader = model_params.copy()
    try:
        train_dataset = TripletPoseDataset(pose_annotations_file=model_params['train_annotations'],
                                           validation_mode=False,
                                           in_memory=model_params['in_memory_generator_train'],
                                           **dataset_params_for_loader)
        train_loader = DataLoader(train_dataset, batch_size=model_params['batch_size'], shuffle=True,
                                  num_workers=model_params.get('num_workers', 0),
                                  pin_memory=True if device.type == 'cuda' else False, drop_last=True)
        print(f"Train DataLoader: Batches per epoch approx {len(train_loader)}")
    except Exception as e:
        print(f"CRITICAL Error creating train_dataset or train_loader: {e}")
        import traceback; traceback.print_exc(); return

    val_loader = None
    if model_params.get('val_annotations') and model_params['val_annotations'] != '':
        try:
            val_dataset = TripletPoseDataset(pose_annotations_file=model_params['val_annotations'],
                                             validation_mode=True,
                                             in_memory=model_params['in_memory_generator_val'],
                                             **dataset_params_for_loader)
            val_loader = DataLoader(val_dataset, batch_size=model_params['batch_size'], shuffle=True,
                                    num_workers=model_params.get('num_workers', 0),
                                    pin_memory=True if device.type == 'cuda' else False, drop_last=False)
            print(f"Validation DataLoader: Batches per epoch approx {len(val_loader)}")
        except Exception as e:
            print(f"Error creating val_dataset or val_loader: {e}. Proceeding without validation.")
            val_loader = None
    else:
        print("No validation annotations provided, val_loader will be None.")


    # --- Training Loop ---
    print("\n--- Starting PyTorch Training ---")
    num_epochs = model_params.get('epochs', 1)
    print(f"Training for {num_epochs} epochs with batch size {model_params['batch_size']}")
    time.sleep(5)  # Small delay for readability in logs
    
    if model_params.get('resume_training', False) and model_params.get('checkpoint_file_path', None) is not None:
        checkpoint_file_path = model_params.get('checkpoint_file_path')
        if checkpoint_file_path is None:
            raise ValueError("resume_training is True, but no checkpoint_file_path was provided in model_params.")
        checkpoint = torch.load(checkpoint_file_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_monitor_metric_val = checkpoint['best_monitor_metric_val']
        early_stopping_counter = checkpoint['early_stopping_counter']
        print(f"Resuming training from checkpoint at epoch {start_epoch}")
    else:
        start_epoch = 0
        best_monitor_metric_val = float('-inf') if monitor_mode == 'max' else float('inf')
        early_stopping_counter = 0

    softmax_outputs = [] # To store softmax outputs if needed
    train_losses = []
    val_losses = []
    val_f1_scores = []
    val_auc_scores = []  # Move this to the top, before the epoch loop

    for epoch in range(start_epoch, num_epochs):
        epoch_start_time = time.time()
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        model.train()
        running_train_loss = 0.0
        # Store individual loss components if needed for logging
        running_train_loss_clf = 0.0
        running_train_loss_triplet = 0.0
        scaler = GradScaler()

        for batch_idx, (features_batch, labels_batch) in enumerate(train_loader):
            start_batch_time = time.time()
            
            t0 = time.time()
            features_batch = features_batch.to(device, non_blocking=True)
            labels_batch = labels_batch.to(device, non_blocking=True) # Integer class labels
            t1 = time.time()
            
            # Debugging print in order to see the batch shapes and labels
            # if train_verbose > 0 and batch_idx < 2 and epoch < 2:
            #     for i, feature in enumerate(features_batch):
            #         print(f"  Batch {batch_idx+1}, Sample {i}: Feature shape: {feature.shape}, Label: {labels_batch[i]}")
            # Reset gradients
            optimizer.zero_grad()
            
            current_batch_total_loss = torch.tensor(0.0, device=device)

            # Forward pass
            # Model output: [embedding, clf_logits] if both active, or just one of them
            t2 = time.time()
            with autocast():
                model_output = model(features_batch)
            t3 = time.time()

            embeddings_batch = None
            clf_logits_batch = None

            if model_params.get('triplet', False) and model_params.get('classification', True):
                if isinstance(model_output, list) and len(model_output) == 2:
                    embeddings_batch, clf_logits_batch = model_output
                else:
                    print("Warning: Model output format unexpected for triplet+classification. Skipping batch.")
                    continue
            elif model_params.get('triplet', False):
                embeddings_batch = model_output # Assumes model returns only embeddings
            elif model_params.get('classification', True):
                clf_logits_batch = model_output # Assumes model returns only clf_logits
            else:
                print("Warning: Neither triplet nor classification is enabled. No loss to compute. Skipping batch.")
                continue
                
            t4 = time.time()
            
            # Calculate Classification Loss
            with autocast():
                if 'classification' in active_losses and clf_logits_batch is not None:
                    loss_c = active_losses['classification'](clf_logits_batch, labels_batch)
                    current_batch_total_loss += loss_weights_pytorch_pt['classification'] * loss_c
                    running_train_loss_clf += loss_c.item()
                    if batch_idx == 0 and epoch == 0: print(f"  Classification loss component active. Example batch loss_c: {loss_c.item():.4f}")
            t5 = time.time()
            
            # Calculate Triplet Loss (Requires Batch Mining)
            if 'triplet' in active_losses and embeddings_batch is not None:
                # TODO: Implement Triplet Mining here if not using a library that does it.
                # This is a placeholder. You'd typically use a library like pytorch-metric-learning
                # or implement your own batch hard/semi-hard mining.
                # For simplicity, if K=1 in Keras generator, this might not have been batch mining.
                # If your Dataset was returning A,P,N, this would be simpler.
                # Since Dataset returns (sample, label), we need to mine here.
                # This is a complex step. For now, we'll skip the actual triplet loss calculation
                # unless you have a specific mining strategy ready.
                # loss_t = active_losses['triplet'](anchor_embs, positive_embs, negative_embs)
                # current_batch_total_loss += loss_weights_pytorch['triplet'] * loss_t
                # running_train_loss_triplet += loss_t.item()
                if batch_idx == 0 and epoch == 0: print("  Triplet loss component active, but MINING LOGIC IS A PLACEHOLDER.")
                pass # Placeholder for triplet loss calculation

            if isinstance(current_batch_total_loss, torch.Tensor): # Check if any loss was added
                scaler.scale(current_batch_total_loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                running_train_loss += current_batch_total_loss.item()
            elif current_batch_total_loss == 0.0 and not (model_params.get('triplet',False) or model_params.get('classification',True)):
                pass # No losses active, normal
            else:
                if train_verbose > 0: print(f"  Train Batch: {batch_idx+1}/{len(train_loader)} - No loss computed for this batch.")
                continue
            t6 = time.time()

            end_batch_time = time.time()

            if train_verbose > 0 and batch_idx % log_interval == 0 and isinstance(current_batch_total_loss, torch.Tensor):
                print(f"  Train Batch: {batch_idx+1}/{len(train_loader)} Loss: {current_batch_total_loss.item():.4f}")
                print(
                    f"[Batch {batch_idx+1}] "
                    f"load: {(t1-t0)*1000:.1f}ms | "
                    f"fwd: {(t3-t2)*1000:.1f}ms | "
                    f"parse: {(t4-t3)*1000:.1f}ms | "
                    f"loss: {(t5-t4)*1000:.1f}ms | "
                    f"bkwd+step: {(t6-t5)*1000:.1f}ms | "
                    f"total: {(end_batch_time-start_batch_time)*1000:.1f}ms"
                )

        # End of epoch training loop
        avg_epoch_train_loss = running_train_loss / len(train_loader) if len(train_loader) > 0 else 0
        train_losses.append(avg_epoch_train_loss)
        tb_writer.add_scalar('LossEpoch_Train/Total', avg_epoch_train_loss, epoch)
        if 'classification' in active_losses:
            tb_writer.add_scalar('LossEpoch_Train/Classification', running_train_loss_clf / len(train_loader) if len(train_loader) > 0 else 0, epoch)
        if 'triplet' in active_losses: # Add if triplet loss is calculated
            tb_writer.add_scalar('LossEpoch_Train/Triplet', running_train_loss_triplet / len(train_loader) if len(train_loader) > 0 else 0, epoch)

        current_lr = optimizer.param_groups[0]['lr']
        tb_writer.add_scalar('LearningRate', current_lr, epoch)
        print(f"Epoch {epoch+1} Train Summary: Avg Total Loss: {avg_epoch_train_loss:.4f}, LR: {current_lr}")

        # --- Validation Phase ---
        val_metrics = {}  # To store metrics like loss, accuracy
        if val_loader:
            model.eval()
            running_val_loss = 0.0
            running_val_loss_clf = 0.0
            running_val_loss_triplet = 0.0
            all_clf_preds_val = []
            all_clf_labels_val = []
            all_clf_probs_val = []

            with torch.no_grad():
                for features_batch_val, labels_batch_val in val_loader:
                    features_batch_val = features_batch_val.to(device, non_blocking=True)
                    labels_batch_val = labels_batch_val.to(device, non_blocking=True)

                    model_output_val = model(features_batch_val)
                    current_batch_val_total_loss = 0.0
                    embeddings_batch_val = None
                    clf_logits_batch_val = None

                    if model_params.get('triplet', False) and model_params.get('classification', True):
                        if isinstance(model_output_val, list) and len(model_output_val) == 2:
                            embeddings_batch_val, clf_logits_batch_val = model_output_val
                        else:
                            continue  # Skip if format is wrong
                    elif model_params.get('triplet', True):
                        embeddings_batch_val = model_output_val
                    elif model_params.get('classification', True):
                        clf_logits_batch_val = model_output_val

                    if 'classification' in active_losses and clf_logits_batch_val is not None:
                        loss_c_val = active_losses['classification'](clf_logits_batch_val, labels_batch_val)
                        current_batch_val_total_loss += loss_weights_pytorch_pt['classification'] * loss_c_val
                        running_val_loss_clf += loss_c_val.item()

                        probs_val = torch.softmax(clf_logits_batch_val, dim=1)  # (batch_size, num_classes)
                        all_clf_probs_val.extend(probs_val.cpu().numpy())  # Collect probabilities
                        _, predicted_indices = torch.max(probs_val, 1)
                        all_clf_preds_val.extend(predicted_indices.cpu().numpy())
                        all_clf_labels_val.extend(labels_batch_val.cpu().numpy())

                    if 'triplet' in active_losses and embeddings_batch_val is not None:
                        triplet_loss_val = active_losses['triplet'](embeddings_batch_val, labels_batch_val)
                        current_batch_val_total_loss += loss_weights_pytorch_pt['triplet'] * triplet_loss_val
                        running_val_loss_triplet += triplet_loss_val.item()

                    if isinstance(current_batch_val_total_loss, torch.Tensor):
                        running_val_loss += current_batch_val_total_loss.item()

            avg_epoch_val_loss = running_val_loss / len(val_loader) if len(val_loader) > 0 else 0
            val_metrics['val_loss'] = avg_epoch_val_loss
            val_losses.append(avg_epoch_val_loss)
            tb_writer.add_scalar('LossEpoch_Val/Total', avg_epoch_val_loss, epoch)

            # AUC-ROC Calculation (if classification and probabilities available)
            if all_clf_labels_val and all_clf_probs_val:
                y_true = np.array(all_clf_labels_val)
                y_scores = np.array(all_clf_probs_val)

                if y_scores.shape[1] == 2:
                    # Binary classification: take score for class 1
                    auc = roc_auc_score(y_true, y_scores[:, 1])
                else:
                    # Multiclass classification
                    auc = roc_auc_score(y_true, y_scores, multi_class='ovr')

                val_metrics['val_auc'] = auc
                tb_writer.add_scalar('AUC_ROC/val', auc, epoch)
                val_auc_scores.append(auc)

            if 'classification' in active_losses:
                avg_val_loss_clf = running_val_loss_clf / len(val_loader)
                tb_writer.add_scalar('LossEpoch_Val/Classification', avg_val_loss_clf, epoch)
                val_metrics['val_clf_loss'] = avg_val_loss_clf

            if all_clf_labels_val:
                correct_val = sum(p == t for p, t in zip(all_clf_preds_val, all_clf_labels_val))
                val_accuracy = correct_val / len(all_clf_labels_val)
                val_metrics['val_accuracy'] = val_accuracy
                tb_writer.add_scalar('Accuracy/val', val_accuracy, epoch)

                f1_macro = f1_score(all_clf_labels_val, all_clf_preds_val, average='macro')
                val_metrics['val_f1_macro'] = f1_macro
                tb_writer.add_scalar('F1Score/val_macro', f1_macro, epoch)
                val_f1_scores.append(f1_macro)

                print(f"Epoch {epoch+1} Val Summary:")
                print(f"  - Total Loss       : {avg_epoch_val_loss:.4f}")
                print(f"  - Accuracy         : {val_accuracy:.4f}")
                print(f"  - F1 Score (macro) : {f1_macro:.4f}")
                print(f"  - AUC-ROC          : {auc:.4f}")

            else:
                print(f"Epoch {epoch+1} Val Summary: Avg Total Loss: {avg_epoch_val_loss:.4f} (No classification preds for accuracy)")

            if 'triplet' in active_losses:
                avg_val_loss_triplet = running_val_loss_triplet / len(val_loader)
                val_metrics['val_triplet_loss'] = avg_val_loss_triplet
                tb_writer.add_scalar('LossEpoch_Val/Triplet', avg_val_loss_triplet, epoch)

            if 'classification' not in active_losses:
                print(f"Epoch {epoch+1} Val Summary: Avg Total Loss: {avg_epoch_val_loss:.4f}")

            # --- "Callbacks" logic for this epoch ---
            current_metric_for_scheduler_es = val_metrics.get(monitor_metric_name, avg_epoch_val_loss)

            lr_scheduler.step(current_metric_for_scheduler_es) # Step LR scheduler

            # ModelCheckpoint
            if (monitor_mode == 'max' and current_metric_for_scheduler_es > best_monitor_metric_val) or \
            (monitor_mode == 'min' and current_metric_for_scheduler_es < best_monitor_metric_val):
                best_monitor_metric_val = current_metric_for_scheduler_es
                # Keras format: 'ep{epoch:03d}-loss{loss:.5f}-' + monitor + '{' + monitor + ':.5f}.ckpt'
                # Using train loss for 'loss' part of filename for consistency with Keras.
                checkpoint_filename = f"ep{epoch+1:03d}-trainloss{avg_epoch_train_loss:.5f}-{monitor_metric_name.replace('val_','')}{current_metric_for_scheduler_es:.5f}.pt"
                torch.save({
                    'epoch': epoch+1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_monitor_metric_val': best_monitor_metric_val,
                    'early_stopping_counter': early_stopping_counter
                }, os.path.join(weights_save_path, checkpoint_filename))
                print(f"  Saved checkpoint: {os.path.join(weights_save_path, checkpoint_filename)} (Monitored '{monitor_metric_name}': {current_metric_for_scheduler_es:.5f})")
                # Update best_val_for_early_stop if this is the metric early stopping also monitors
                if monitor_metric_name == model_params.get('monitor', 'val_loss'): # Check if it's the same metric
                    best_val_for_early_stop = best_monitor_metric_val
                    early_stopping_counter = 0
            else: # Only increment early stopping counter if not improving on its specific metric
                if monitor_metric_name == model_params.get('monitor', 'val_loss'):
                    early_stopping_counter += 1

            # Confusion Matrix (Every 10 epochs)
            if epoch == 0 or (epoch + 1) % 10 == 0: # Every 10 epochs, save confusion matrix (starting from epoch 0)
                # Get the current directory (where the script is)
                current_dir = os.path.dirname(os.path.abspath(__file__))

                # Navigate one level up and into "Conversion comparison"
                parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
                confusion_matrix_dir = os.path.join(parent_dir, 'Conversion comparison')
                os.makedirs(confusion_matrix_dir, exist_ok=True)

                # Compute confusion matrix (example)
                conf_mat = confusion_matrix(all_clf_labels_val, all_clf_preds_val)

                # Save the raw matrix as .npy
                npy_path = os.path.join(confusion_matrix_dir, f'pytorch_conf_matrix_epoch_{epoch+1:03d}.npy')
                np.save(npy_path, conf_mat)

                # Optionally save a visualisation as PNG
                plt.figure(figsize=(40, 30))  # Adjust size as needed
                ax = sns.heatmap(
                    conf_mat,
                    annot=True,
                    fmt='d',
                    cmap='Blues',
                    annot_kws={"size": 5},  # Smaller font to fit more
                    cbar=True
                )
                plt.xlabel("Predicted")
                plt.ylabel("Actual")
                plt.title(f"Confusion Matrix - Epoch {epoch+1}")
                plt.xticks(rotation=90, fontsize=5)  # Rotate and reduce font size
                plt.yticks(fontsize=5)
                plt.tight_layout()
                png_path = os.path.join(confusion_matrix_dir, f'conf_matrix_epoch_{epoch+1:03d}.png')
                plt.savefig(png_path)
                plt.close()

            # EarlyStopping (check against its own best metric, which might be same as checkpointing or different)
            if early_stopping_counter >= early_stopping_patience:
                print(f"  Early stopping triggered after {early_stopping_patience} epochs without improvement on '{monitor_metric_name}'.")
                break # Break from epoch loop
        else: # No val_loader
            print("  No validation loader. Skipping validation phase, LR scheduling based on val_metrics, and early stopping.")
            # Optionally, save model at end of epoch if no validation
            checkpoint_filename = f"ep{epoch+1:03d}-trainloss{avg_epoch_train_loss:.5f}-no_val.pt"
            torch.save(model.state_dict(), os.path.join(weights_save_path, checkpoint_filename))
            print(f"  Saved model checkpoint (no validation): {checkpoint_filename}")

        epoch_duration = time.time() - epoch_start_time
        print(f"Epoch {epoch+1} duration: {epoch_duration:.2f} seconds")
        if epoch_duration > 0 : tb_writer.add_scalar('Performance/epoch_duration_sec', epoch_duration, epoch)

    # Determine folder one level up
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
    metrics_save_dir = os.path.join(parent_dir, 'Conversion comparison')
    os.makedirs(metrics_save_dir, exist_ok=True)

    np.savez(os.path.join(metrics_save_dir, 'pytorch_train_loss_val_loss_val_f1_val_auc.npz'),
        train_losses=np.array(train_losses),
        val_losses=np.array(val_losses),
        val_f1_scores=np.array(val_f1_scores),
        val_auc_scores=np.array(val_auc_scores),
    )

    tb_writer.close()
    print("\n--- PyTorch Training Finished ---")

    # --- Post-training actions ---
    # TODO: Example: remove_suboptimal_weights (needs careful implementation)
    # if model_params.get('remove_suboptimal_weights', False):
    #     try:
    #         # This function needs to be defined and robust
    #         from remove_suboptimal_weights import remove_path_weights_pytorch
    #         remove_path_weights_pytorch(weights_save_path, monitor_metric_name, mode=monitor_mode)
    #     except ImportError:
    #         print("Warning: remove_suboptimal_weights_pytorch not found.")
    #     except Exception as e_rsw:
    #         print(f"Error during remove_suboptimal_weights: {e_rsw}")


if __name__ == "__main__":

    print(torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU found")

    model_params = {
        "train_verbose": 1,  # Set to 0 for no training logs, 1 for basic logs, >1 for more detailed logs
        "num_workers": 1,  # Number of workers for DataLoader, adjust based on your system
        "path_results": "./pretrained_models_Pytorch/",
        "resume_training": False,  # Set to True to resume training from a checkpoint
        "checkpoint_file_path":  None,  # Path to the checkpoint file if resuming training 
        
        "epochs": 100,
        "in_memory_generator_train": False,
        "in_memory_generator_val": False,

        # # NTU-120 Data sets to optimize the therapy data
        "train_annotations": "./ntu_annotations/one_shot_aux_set_train_full8.txt",
        "val_annotations": "./ntu_annotations/one_shot_aux_set_val_full8.txt",
        "eval_therapies": True,       ### Therapy data needed for its evaluation
        # "eval_therapies_triplets_dataset": "./therapies_annotations/triplets/triplets_dataset.pckl",
        # "eval_therapies_triplets_bgnd_dataset": "./therapies_annotations/triplets/triplets_ther_pat_bgnd_dataset.pckl",
        # "eval_therapies_video_skels": "./therapies_annotations/video_skels.pckl",
        # "h_flip": True,
        # "skip_frames": [2, 3],

        # NTU-120 Data sets to optimize the NTU one-shot benchmark
        #"train_annotations": "./ntu_annotations/one_shot_aux_set.txt",
        # "val_annotations": "",
        # "eval_therapies": False,
        #"h_flip": False,
        #"monitor": "ntu_one_shot_acc_euc",
        #"min_monitor": False,
        #"skip_frames": [2],

        #"in_memory_callback": True,

        "eval_ntu": True,
        "eval_ntu_one_shot_eval_anchors_file": "./ntu_annotations/one_shot_eval_anchors.txt",
        "eval_ntu_one_shot_eval_set_file": "./ntu_annotations/one_shot_eval_set.txt",

        "joints_num": 25,
        "joints_dim": 3,
        "init_lr": 0.0001,
        "max_seq_len": -32,

        # Set True to use a fitted data scaler. The one from the pre-trained models can also be used
        "scale_data": False,       
        "lstm_recurrent_dropout": 0.0,
        "lstm_dropout": 0.2,
        "num_layers": 2,
        "num_neurons": 256,
        "batch_size": 64,
        "masking": True,
        "center_skels": True,
        "scale_by_torso": True,
        "temporal_scale": [0.8, 1.2],
        "classification": True,
        "triplet": False,
        "decoder": False,
        "reverse_decoder": False,
        "num_classes": 120,
        "clf_neurons": 0,

        "model_name": "train_TCN_Pytorch_NTU120_one_shot_aux_set_full",
        "conv_params": [256, 4, 2, True, "causal", [4]],
        "is_tcn": False,
        "use_jcd_features": True,
        "use_speeds": False,
        "use_coords_raw": False,
        "use_coords": True,
        "use_jcd_diff": False,
        "use_bone_angles": True,
        "use_bone_angles_cent": False,
        "average_wrong_skels": True,
        "average_wrong_skels_method": 'mean',   
        }
    
        # Correct max_seq_len if negative for use in summary/testing
    if model_params['max_seq_len'] <= 0:
        print(f"Warning: model_params['max_seq_len'] is {model_params['max_seq_len']}. Using 32 as effective_seq_len for non-dataset parts.")
        model_params['effective_seq_len'] = 32
    else:
        model_params['effective_seq_len'] = model_params['max_seq_len']

    # --- Call the main function ---
    main(model_params)

    print("\n--- Training script finished ---")
