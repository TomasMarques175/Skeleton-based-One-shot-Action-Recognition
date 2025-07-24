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
# Dataset is imported from pytorch_dataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
import re

# --- Utility imports ---
from scipy.special import comb  # Used by get_num_feats
import numpy as np
import json
from shutil import copyfile
import glob  # For file searching if adapting remove_suboptimal_weights
import time  # For timing epochs
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score
import argparse
from sklearn.model_selection import train_test_split

from sklearn.model_selection import StratifiedKFold


# --- PyTorch specific imports ---
from models.TCN_classifier import TCN_clf  # Your PyTorch model
from pytorch_dataset import TripletPoseDataset, get_num_feats, get_scaler_filename # Your PyTorch Dataset

# --- TensorFlow imports ---
import tensorflow as tf
import copy

# --- PyTorch Dataset imports ---
from torch.utils.data import DataLoader, WeightedRandomSampler # Dataset is imported from pytorch_dataset
import os
import pickle
from sklearn.model_selection import StratifiedShuffleSplit
from pytorch_dataset import TripletPoseDataset, TherapyDataset # Your PyTorch Dataset

from collections import Counter
import optuna

# --- Framework-agnostic or to-be-adapted utility imports ---
# Ensure these functions do not have hard TensorFlow dependencies.
# It's assumed these are available in your environment, possibly from data_generator.py
# If data_generator.py has TF dependencies, these functions need to be extracted/rewritten.
# try:
#     from data_generator import get_scaler_filename
# except ImportError:
#     print("Warning: Could not import from data_generator.py. Ensure get_scaler_filename and get_num_feats are defined or TF-free.")
#     # Define placeholders if they are critical and not available, but this is not ideal
# 
#     def get_num_feats(
#         **kwargs): raise NotImplementedError("get_num_feats is not defined/imported")
# 
#     def get_scaler_filename(
#         **kwargs): raise NotImplementedError("get_scaler_filename is not defined/imported")

try:
    import train_utils  # For create_model_folder
except ImportError:
    print("Warning: train_utils not found. create_model_folder might fail.")

    class train_utils_placeholder:  # Basic placeholder
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


def copy_scaler_if_needed(model_params):
    """
    Copy the scaler file to the model output directory if scaling is enabled.
    Uses `get_scaler_filename(**model_params)` to find the source file.

    Args:
        model_params (dict): Dictionary containing model configuration, including 'scale_data' and 'path_model'.
    """
    if model_params.get('scale_data', False):
        try:
            scaler_filename_src = get_scaler_filename(**model_params)
            if scaler_filename_src:
                scaler_filename_dst = os.path.join(
                    model_params['path_model'], 'scaler.pckl')
                if os.path.exists(scaler_filename_src):
                    copyfile(scaler_filename_src, scaler_filename_dst)
                    print(f"Copied scaler: {scaler_filename_src} to {scaler_filename_dst}")
                else:
                    print(f"Warning: Scaler source file '{scaler_filename_src}' not found. Not copied.")
            else:
                print("Warning: get_scaler_filename did not return a path. Scaler not copied.")
        except Exception as e:
            print(f"Warning: Error handling scaler file: {e}. Skipping scaler copy.")

def train_model(model_params, running_train_loss_clf, running_train_loss, pytorch_model, softmax_outputs, device, train_loader, optimizer, active_losses, loss_weights_pytorch_pt, epoch=0, train_verbose=1, log_interval=100):
    
    for batch_idx, (features_batch, labels_batch) in enumerate(train_loader):
        features_batch = features_batch.to(device, non_blocking=True)
        labels_batch = labels_batch.to(
            device, non_blocking=True)  # Integer class labels
        optimizer.zero_grad()

        # Forward pass
        # Model output: [embedding, clf_logits] if both active, or just one of them
        model_output = pytorch_model(features_batch)

        current_batch_total_loss = 0.0
        embeddings_batch = None
        clf_logits_batch = None

        if model_params.get('triplet', False) and model_params.get('classification', True):
            if isinstance(model_output, list) and len(model_output) == 2:
                embeddings_batch, clf_logits_batch = model_output
            else:
                print(
                    "Warning: Model output format unexpected for triplet+classification. Skipping batch.")
                continue
        elif model_params.get('triplet', False):
            embeddings_batch = model_output  # Assumes model returns only embeddings
        elif model_params.get('classification', True):
            clf_logits_batch = model_output  # Assumes model returns only clf_logits
        else:
            print(
                "Warning: Neither triplet nor classification is enabled. No loss to compute. Skipping batch.")
            continue

        if clf_logits_batch is not None:
            with torch.no_grad():
                probs = torch.softmax(clf_logits_batch, dim=1)
                softmax_outputs.append(probs.cpu().numpy())

        # Calculate Classification Loss
        if 'classification' in active_losses and clf_logits_batch is not None:
            # Ensure labels_batch is of type LongTensor for CrossEntropyLoss
            loss_c = active_losses['classification'](
                clf_logits_batch, labels_batch.to(device).long())
            current_batch_total_loss += loss_weights_pytorch_pt['classification'] * loss_c
            running_train_loss_clf += loss_c.item()
            # if batch_idx == 0 and epoch == 0:
            #     print(
            #         f"  Classification loss component active. Example batch loss_c: {loss_c.item():.4f}")

        # Calculate Triplet Loss (Requires Batch Mining)
        if 'triplet' in active_losses and embeddings_batch is not None:
            # TODO: NOT IMPLEMENTED: Triplet loss mining logic
            # Placeholder for triplet loss calculation
            if batch_idx == 0 and epoch == 0:
                print(
                    "  Triplet loss component active, but MINING LOGIC IS A PLACEHOLDER.")
            pass  # Placeholder for triplet loss calculation

        if isinstance(current_batch_total_loss, torch.Tensor):  # Check if any loss was added
            current_batch_total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                pytorch_model.parameters(), max_norm=1.0)  # Gradient Clipping
            optimizer.step()
            running_train_loss += current_batch_total_loss.item()
        elif current_batch_total_loss == 0.0 and not (model_params.get('triplet', False) or model_params.get('classification', True)):
            pass  # No losses active, normal
        else:
            if train_verbose > 0:
                print(
                    f"  Train Batch: {batch_idx+1}/{len(train_loader)} - No loss computed for this batch.")
            continue

        return running_train_loss, running_train_loss_clf
        # if train_verbose > 0 and batch_idx % log_interval == 0 and isinstance(current_batch_total_loss, torch.Tensor):
        #     print(
        #         f"  Train Batch: {batch_idx+1}/{len(train_loader)} Loss: {current_batch_total_loss.item():.4f}")

def validate_model(model_params, pytorch_model, active_losses, device, val_loader, loss_weights_pytorch_pt):
    """
    Validate the model on the validation dataset.
    Args:
        model_params (dict): Dictionary containing model configuration.
        device (torch.device): Device to run the validation on.
        val_loader (DataLoader): DataLoader for the validation dataset.
        tb_writer (SummaryWriter, optional): TensorBoard writer for logging.
        epoch (int, optional): Current epoch number for logging.
    Returns:
        float: Average validation loss.
    """
    val_metrics = {}  # To store metrics like loss, accuracy
    if val_loader:
        pytorch_model.eval()
        running_val_loss = 0.0
        running_val_loss_clf = 0.0
        running_val_loss_triplet = 0.0
        all_clf_preds_val = []
        all_clf_labels_val = []
        all_clf_probs_val = []

        with torch.no_grad():
            for features_batch_val, labels_batch_val in val_loader:
                features_batch_val = features_batch_val.to(
                    device, non_blocking=True)
                labels_batch_val = labels_batch_val.to(
                    device, non_blocking=True)

                model_output_val = pytorch_model(features_batch_val)
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
                    loss_c_val = active_losses['classification'](
                        clf_logits_batch_val, labels_batch_val)
                    current_batch_val_total_loss += loss_weights_pytorch_pt['classification'] * loss_c_val
                    running_val_loss_clf += loss_c_val.item()

                    # (batch_size, num_classes)
                    probs_val = torch.softmax(clf_logits_batch_val, dim=1)
                    # Collect probabilities
                    all_clf_probs_val.extend(probs_val.cpu().numpy())
                    _, predicted_indices = torch.max(probs_val, 1)
                    all_clf_preds_val.extend(
                        predicted_indices.cpu().numpy())
                    all_clf_labels_val.extend(
                        labels_batch_val.cpu().numpy())

                if 'triplet' in active_losses and embeddings_batch_val is not None:
                    triplet_loss_val = active_losses['triplet'](
                        embeddings_batch_val, labels_batch_val)
                    current_batch_val_total_loss += loss_weights_pytorch_pt['triplet'] * \
                        triplet_loss_val
                    running_val_loss_triplet += triplet_loss_val.item()

                if isinstance(current_batch_val_total_loss, torch.Tensor):
                    running_val_loss += current_batch_val_total_loss.item()

    return val_metrics, running_val_loss, running_val_loss_clf, running_val_loss_triplet, all_clf_preds_val, all_clf_labels_val, all_clf_probs_val

def Setup_optimizer_and_loss(pytorch_model, model_params, device, train_dataset=None):
    """ Set up the optimizer and loss functions for the PyTorch model.
    Args:
        model_params (dict): Dictionary containing model configuration.
    """
    # --- PyTorch Optimizer and Loss (Mimicking Keras printouts) ---
    # print(' * Defining losses and loss_weights (PyTorch)')
    active_losses = {}
    loss_weights_pytorch_pt = {}
    if model_params.get('classification', True):
        # Get inverse frequency weights
                # Extract all labels from the training dataset
        train_labels_from_dataset = [label for _, label in train_dataset]
        # Convert tensors to integers if needed
        train_labels_from_dataset = [label.item() if torch.is_tensor(label) else label for label in train_labels_from_dataset]
        counts = Counter(train_labels_from_dataset)
        num_classes = len(counts)
        total = sum(counts.values())
        weights = [total / (num_classes * counts[i]) for i in range(num_classes)]
        weights_tensor = torch.tensor(weights, dtype=torch.float32).to(device)
        criterion_clf = nn.CrossEntropyLoss(weight=weights_tensor)
        active_losses['classification'] = criterion_clf
        # Match Keras loss_weights['output_1'] = 0.4 if classification is the primary/first output
        loss_weights_pytorch_pt['classification'] = model_params.get(
            'clf_loss_weight', 0.4)
    if model_params.get('triplet', False):  # If you also had triplet loss in Keras
        criterion_triplet_pt = nn.TripletMarginLoss(
            margin=model_params.get('triplet_margin', 1.0))
        active_losses['triplet'] = criterion_clf
        loss_weights_pytorch_pt['triplet'] = model_params.get(
            'triplet_loss_weight', 0.6)  # Example
        print(' * losses (PyTorch types):', active_losses)
        print(' * loss_weights (PyTorch):', loss_weights_pytorch_pt)
        # sample_weights_mode is not a direct PyTorch concept, handled manually if needed

    # print('\n * Setting optimizer (PyTorch)')
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, pytorch_model.parameters()),
                    lr=model_params['init_lr'])
    # Note: clipnorm is applied manually in PyTorch training loop (torch.nn.utils.clip_grad_norm_)
    # print(f"   Optimizer: {type(optimizer)}, LR: {model_params['init_lr']}")


    return active_losses, loss_weights_pytorch_pt, optimizer

def create_pytorch_model(model_params):
    # --- PyTorch Model Instantiation ---
    # print('\n* Setting model parameters (PyTorch)')  # Mimicking Keras log
    # Ensure num_classes is correctly derived if dataset re-indexing happened,
    # though for this debug block, it might not matter if we don't train.
    num_classes_for_model = model_params.get(
        'actual_num_classes', model_params['num_classes'])
    initial_state_dict = None  # Initialize to None, will be set later
    
    # If the model is in Pytorch format, we assume it has been converted or is ready to be used.
    if model_params.get("use_pretrained_model", False) and not model_params.get('model_converter', False):
        # --- PyTorch Model Instantiation ---
        # If the model is in PyTorch format, we will load it directly.
        pytorch_model = TCN_clf(
            num_feats=model_params['num_feats'], conv_params=model_params['conv_params'],
            lstm_dropout=model_params['lstm_dropout'], masking=model_params['masking'],
            triplet=model_params.get('triplet', False), classification=model_params.get('classification', True),
            clf_neurons=model_params['clf_neurons'], num_classes=num_classes_for_model
        )

        # After loading the model with the pretrained weights
        checkpoint_path = model_params["pretrained_model_path"]
        pytorch_model.load_state_dict(torch.load(checkpoint_path))

        # Define which keys are *excluded from training* (i.e., frozen)
        excluded_pt_keys = model_params.get('excluded_pt_keys', [])

        # Freeze only the layers you want
        for name, param in pytorch_model.named_parameters():
            if name in excluded_pt_keys:
                param.requires_grad = False
                print(f"Froze: {name}")
            else:
                param.requires_grad = True  # Unfrozen by default

        # Capture the initial state_dict *after freezing*
        initial_state_dict = {k: v.clone().detach().cpu() for k, v in pytorch_model.state_dict().items()}

    # If the model is in TensorFlow/Keras format, we will convert it to PyTorch.
    if model_params.get('model_converter', False):
        # --- PyTorch Model Instantiation ---
        pytorch_model = TCN_clf(
            num_feats=model_params['num_feats'], conv_params=model_params['conv_params'],
            lstm_dropout=model_params['lstm_dropout'], masking=model_params['masking'],
            triplet=model_params.get('triplet', False), classification=model_params.get('classification', True),
            clf_neurons=model_params['clf_neurons'], num_classes=num_classes_for_model
        )
        
        #print("\n=== PyTorch state_dict keys ===")
        #for k, v in pytorch_model.state_dict().items():
        #    print(f"{k} {tuple(v.shape)}")

        # ------------------------------
        # 2. Load the TensorFlow SavedModel
        tf_model = tf.saved_model.load(model_params["pretrained_model_path"])
        #print("\n=== TensorFlow variables ===")
        
        #for var in tf_model.variables:
        #    print(f"{var.name} {tuple(var.shape)}")
            
        # --- 3. Extract TensorFlow weights to numpy dict ---
        tf_weights = {}
        for var in tf_model.variables:
            # Convert tensor to numpy
            tf_weights[var.name] = var.numpy()
        #    print(var.name, var.shape)

        excluded_pt_keys = [
            'clf_out.weight', 
            'clf_out.bias',
            # 'encoder_net.encoder.0.residual_blocks.0.conv1.weight',
            # 'encoder_net.encoder.0.residual_blocks.0.conv1.bias',
            # 'encoder_net.encoder.0.residual_blocks.0.downsample.weight',
            # 'encoder_net.encoder.0.residual_blocks.0.downsample.bias'
        ]

        excluded_tf_prefixes = [
            'out_clf/kernel:0',
            'out_clf/bias:0',
            # 'encoder_tcn/sequential/tcn/residual_block_0/conv1D_0/kernel:0',
            # 'encoder_tcn/sequential/tcn/residual_block_0/conv1D_0/bias:0',
            # 'encoder_tcn/sequential/tcn/residual_block_0/matching_conv1D/kernel:0',
            # 'encoder_tcn/sequential/tcn/residual_block_0/matching_conv1D/bias:0',
        ]

        # Filter and display PyTorch parameters
        included_pt_params = [(name, p.shape) for name, p in pytorch_model.named_parameters() if name not in excluded_pt_keys]
        total_pt_params = sum(p.numel() for name, p in pytorch_model.named_parameters() if name not in excluded_pt_keys)

        #print(f"\nTotal PyTorch parameters (excluding excluded layers): {total_pt_params}")
        #print("[Included PyTorch parameter keys with shapes:]")
        #for name, shape in included_pt_params:
        #    print(f"  {name}: {tuple(shape)}")

        # Filter and display TensorFlow weights
        tf_weights_filtered = {
            k: v for k, v in tf_weights.items()
            if not any(k.startswith(prefix) for prefix in excluded_tf_prefixes)
        }
        total_tf_params = sum(np.prod(v.shape) for v in tf_weights_filtered.values())

        #print(f"\nTotal TF parameters from checkpoint (excluding excluded layers): {total_tf_params}")
        #print("[Included TensorFlow keys with shapes:]")
        #for k, v in tf_weights_filtered.items():
        #    print(f"  {k}: {v.shape}")
        
        # --- 4. Get PyTorch state dict ---
        pt_state_dict = pytorch_model.state_dict()

        # --- 5. Convert TensorFlow weights to PyTorch ---
        converted_weights = convert_tf_to_torch(tf_weights_filtered, pt_state_dict)

        # --- 6. Update PyTorch state dict and load weights ---
        pt_state_dict.update(converted_weights)
        pytorch_model.load_state_dict(pt_state_dict)
        
        # Validate weights transferred correctly
        with torch.no_grad():
            for name, pt_w in pytorch_model.named_parameters():
                if name in converted_weights:
                    tf_w = converted_weights[name]
                    if pt_w.shape != tf_w.shape:
                        print(f"[MISMATCH] {name}: shape mismatch {pt_w.shape} vs {tf_w.shape}")
                        continue
                    diff = torch.abs(pt_w - tf_w).mean().item()
                    #print(f"{name}: mean abs diff = {diff:.6f}")
                # else:
                #     print(f"[SKIP] {name}: not found in converted weights")

        converted_param_count = sum(
            v.numel() for k, v in converted_weights.items() if k not in excluded_pt_keys
        )
        #print(f"Total converted parameters (excluding excluded layers): {converted_param_count}")
        #print(f"Coverage: {converted_param_count} / {total_pt_params} = {converted_param_count / total_pt_params:.2%}")

        # --- 7. Freeze unconverted parameters ---
        #print("\n=== Freezing unconverted parameters ===")
        # Freeze all parameters that are NOT in the excluded list
        for name, param in pytorch_model.named_parameters():
            if name not in excluded_pt_keys:
                param.requires_grad = False
        # for name, param in pytorch_model.named_parameters():
        #     print(f"{name}: {'❄️ Frozen' if not param.requires_grad else '🔥 Trainable'}")
        initial_state_dict = copy.deepcopy(pytorch_model.state_dict())
    
    # If not converting or using a pre-trained model, initialize a new model
    if not model_params.get('model_converter', False) and not model_params.get("use_pretrained_model", False):
        # If not converting or using a pre-trained model, initialize a new model
        pytorch_model = TCN_clf(
            num_feats=model_params['num_feats'], conv_params=model_params['conv_params'],
            lstm_dropout=model_params['lstm_dropout'], masking=model_params['masking'],
            triplet=model_params.get('triplet', False), classification=model_params.get('classification', True),
            clf_neurons=model_params['clf_neurons'], num_classes=num_classes_for_model
        )
    
    return pytorch_model, initial_state_dict

def Setup_training(model_params, pytorch_model, device, optimizer):

    # print("\n--- Setting up PyTorch Training ---")
    monitor_metric_name = model_params.get('monitor', 'val_loss')
    monitor_is_loss = 'loss' in monitor_metric_name.lower()
    # Keras min_monitor=False means maximize. PyTorch mode='max'.
    # Keras min_monitor=True means minimize. PyTorch mode='min'.
    # Default to minimizing if loss, maximizing otherwise, unless min_monitor overrides.
    default_mode_for_metric = 'min' if monitor_is_loss else 'max'
    if 'min_monitor' in model_params:  # User explicitly set min_monitor
        monitor_mode = 'min' if model_params['min_monitor'] else 'max'
    else:  # Infer from metric name
        monitor_mode = default_mode_for_metric

    best_monitor_metric_val = float(
        '-inf') if monitor_mode == 'max' else float('inf')
    weights_save_path = os.path.join(model_params['path_model'], 'weights')
    os.makedirs(weights_save_path, exist_ok=True)

    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                        mode=monitor_mode,
                                                        factor=model_params.get(
                                                            'lr_factor', 0.1),
                                                        patience=model_params.get(
                                                            'lr_patience', 25), # 3
                                                        min_lr=model_params.get('lr_min_delta', 1e-5)) # 1e-7
    early_stopping_patience = model_params.get('es_patience', 50) # 6
    min_delta = model_params.get("early_stopping_min_delta", 0.001)
    early_stopping_counter = 0

    return monitor_metric_name, monitor_mode, best_monitor_metric_val, weights_save_path, lr_scheduler, early_stopping_patience, min_delta, early_stopping_counter

def convert_tf_to_torch(tf_weights, pt_state_dict):
    import re
    import torch

    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    RESET = "\033[0m"

    converted = {}
    matched_keys = set()
    unmatched_keys = []
    available_tf_keys = set(tf_weights.keys())

    for k in pt_state_dict.keys():
        if k.startswith("encoder_net.encoder.0.residual_blocks."):
            # Extract block index and layer info
            m = re.match(
                r"encoder_net\.encoder\.0\.residual_blocks\.(\d+)\.(\w+)(?:\.(weight|bias))?", k)
            if not m:
                print(f"{YELLOW}[WARN]{RESET} Couldn't parse key: {k}")
                unmatched_keys.append(k)
                continue

            block_idx = int(m.group(1))
            layer_name = m.group(2)
            param_type = m.group(3)

            tf_prefix = f"encoder_tcn/sequential/tcn/residual_block_{block_idx}"

            try:
                tf_key = None

                if block_idx == 0:
                    if layer_name == "conv1":
                        if param_type == "weight":
                            tf_key = tf_prefix + "/conv1D_0/kernel:0"
                            if tf_key not in available_tf_keys:
                                # print(f"{YELLOW}[SKIP]{RESET} Skipping {k} — TF key not in filtered weights ({tf_key})")
                                continue
                            converted[k] = torch.from_numpy(tf_weights[tf_key]).permute(2, 1, 0)

                        elif param_type == "bias":
                            tf_key = tf_prefix + "/conv1D_0/bias:0"
                            if tf_key not in available_tf_keys:
                                # print(f"{YELLOW}[SKIP]{RESET} Skipping {k} — TF key not in filtered weights ({tf_key})")
                                continue
                            converted[k] = torch.from_numpy(tf_weights[tf_key])

                    elif layer_name == "conv2":
                        if param_type == "weight":
                            tf_key = tf_prefix + "/conv1D_1/kernel:0"
                            if tf_key not in available_tf_keys:
                                # print(f"{YELLOW}[SKIP]{RESET} Skipping {k} — TF key not in filtered weights ({tf_key})")
                                continue
                            converted[k] = torch.from_numpy(tf_weights[tf_key]).permute(2, 1, 0)

                        elif param_type == "bias":
                            tf_key = tf_prefix + "/conv1D_1/bias:0"
                            if tf_key not in available_tf_keys:
                                # print(f"{YELLOW}[SKIP]{RESET} Skipping {k} — TF key not in filtered weights ({tf_key})")
                                continue
                            converted[k] = torch.from_numpy(tf_weights[tf_key])

                    elif layer_name == "downsample":
                        if param_type == "weight":
                            tf_key = tf_prefix + "/matching_conv1D/kernel:0"
                            if tf_key not in available_tf_keys:
                                # print(f"{YELLOW}[SKIP]{RESET} Skipping {k} — TF key not in filtered weights ({tf_key})")
                                continue
                            converted[k] = torch.from_numpy(tf_weights[tf_key]).permute(2, 1, 0)

                        elif param_type == "bias":
                            tf_key = tf_prefix + "/matching_conv1D/bias:0"
                            if tf_key not in available_tf_keys:
                                # print(f"{YELLOW}[SKIP]{RESET} Skipping {k} — TF key not in filtered weights ({tf_key})")
                                continue
                            converted[k] = torch.from_numpy(tf_weights[tf_key])

                else:
                    if layer_name == "conv1":
                        if param_type == "weight":
                            tf_key = tf_prefix + "/conv1D_0/kernel:0"
                            if tf_key not in available_tf_keys:
                                # print(f"{YELLOW}[SKIP]{RESET} Skipping {k} — TF key not in filtered weights ({tf_key})")
                                continue
                            converted[k] = torch.from_numpy(tf_weights[tf_key]).permute(2, 1, 0)

                        elif param_type == "bias":
                            tf_key = tf_prefix + "/conv1D_0/bias:0"
                            if tf_key not in available_tf_keys:
                                # print(f"{YELLOW}[SKIP]{RESET} Skipping {k} — TF key not in filtered weights ({tf_key})")
                                continue
                            converted[k] = torch.from_numpy(tf_weights[tf_key])

                    elif layer_name == "conv2":
                        if param_type == "weight":
                            tf_key = tf_prefix + "/conv1D_1/kernel:0"
                            if tf_key not in available_tf_keys:
                                # print(f"{YELLOW}[SKIP]{RESET} Skipping {k} — TF key not in filtered weights ({tf_key})")
                                continue
                            converted[k] = torch.from_numpy(tf_weights[tf_key]).permute(2, 1, 0)

                        elif param_type == "bias":
                            tf_key = tf_prefix + "/conv1D_1/bias:0"
                            if tf_key not in available_tf_keys:
                                # print(f"{YELLOW}[SKIP]{RESET} Skipping {k} — TF key not in filtered weights ({tf_key})")
                                continue
                            converted[k] = torch.from_numpy(tf_weights[tf_key])

                if tf_key is not None:
                    matched_keys.add(k)
                    # print(f"{GREEN}[OK]{RESET} Converted {k} from {tf_key}")

            except KeyError:
                # print(f"{RED}[ERROR]{RESET} Missing TF key: {tf_key}")
                unmatched_keys.append(k)

    all_keys = set(pt_state_dict.keys())
    unmatched_keys = sorted(all_keys - matched_keys)

    # for k in unmatched_keys:
    #     print(f"{YELLOW}[UNMATCHED]{RESET} Did not match {k}")

    # print(f"\n[SUMMARY]")
    # print(f"{GREEN}Matched keys: {len(matched_keys)}{RESET}")
    # print(f"{RED if unmatched_keys else GREEN}Unmatched keys: {len(unmatched_keys)}{RESET}")

    return converted

def init_weights_constant(m):
    if hasattr(m, 'weight') and m.weight is not None:
        nn.init.constant_(m.weight, 0.01)
    if hasattr(m, 'bias') and m.bias is not None:
        nn.init.constant_(m.bias, 0.01)

def collate_fn_classification_pre_pad(batch):
    """
    Pads variable-length sequences on the left (pre-padding).
    Assumes batch = [(sample, label), ...] where:
    - sample: Tensor of shape [seq_len, feat_dim]
    - label: int or tensor
    """
    samples, labels = zip(*batch)  # Unzip list of tuples
    lengths = [s.shape[0] for s in samples]
    max_len = max(lengths)
    feat_dim = samples[0].shape[1]

    padded_samples = []
    for s in samples:
        pad_len = max_len - s.shape[0]
        pad_tensor = torch.zeros((pad_len, feat_dim), dtype=s.dtype)
        padded = torch.cat((pad_tensor, s), dim=0)  # 👈 Pre-padding
        padded_samples.append(padded)

    padded_samples = torch.stack(padded_samples)
    labels = torch.tensor(labels)

    return padded_samples, labels

def Data_Loader_Classification(model_params, train_data, val_data, test_data, batch_size, device):
    # TODO: MP dataset
    # --- Data Loading ---
    print("\n--- Setting up PyTorch DataLoaders ---")
    dataset_params_for_loader = model_params.copy()
    try:
        train_dataset = TripletPoseDataset(pose_annotations_file=model_params['train_annotations'],
                                        validation_mode=False,
                                        in_memory=model_params['in_memory_generator_train'],
                                        **dataset_params_for_loader)
        train_loader = DataLoader(train_dataset, batch_size=model_params['batch_size'], shuffle=True,
                                num_workers=model_params.get(
                                    'num_workers', 0),
                                pin_memory=True if device.type == 'cuda' else False, drop_last=True)
        print(
            f"Train DataLoader: Batches per epoch approx {len(train_loader)}")
    except Exception as e:
        print(f"CRITICAL Error creating train_dataset or train_loader: {e}")
        import traceback
        traceback.print_exc()
        return

    val_loader = None
    if model_params.get('val_annotations') and model_params['val_annotations'] != '':
        try:
            val_dataset = TripletPoseDataset(pose_annotations_file=model_params['val_annotations'],
                                            validation_mode=True,
                                            in_memory=model_params['in_memory_generator_val'],
                                            **dataset_params_for_loader)
            val_loader = DataLoader(val_dataset, batch_size=model_params['batch_size'], shuffle=True,
                                    num_workers=model_params.get(
                                        'num_workers', 0),
                                    pin_memory=True if device.type == 'cuda' else False, drop_last=False)
            print(
                f"Validation DataLoader: Batches per epoch approx {len(val_loader)}")
        except Exception as e:
            print(
                f"Error creating val_dataset or val_loader: {e}. Proceeding without validation.")
            val_loader = None
    else:
        print("No validation annotations provided, val_loader will be None.")
    
    return train_loader, val_loader

def Create_Therapy_Dataloader(model_params, train_data, video_skels, val_data):
    # Create datasets
    train_dataset = TherapyDataset(train_data, video_skels,
                                in_memory=model_params['in_memory_generator_train'],
                                validation=False, **model_params)
    val_dataset = TherapyDataset(val_data, video_skels,
                                in_memory=model_params['in_memory_generator_val'],
                                validation=True, **model_params)

    # Create sampler
    class_counts = train_data['action'].value_counts()
    class_weights = 1. / class_counts
    sample_weights = train_data['action'].map(class_weights).values
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, 
                            batch_size=model_params['batch_size'],
                            sampler=sampler, 
                            num_workers=model_params['num_workers'],
                            drop_last=True, 
                            collate_fn=collate_fn_classification_pre_pad)

    val_loader = DataLoader(val_dataset, 
                            batch_size=model_params['batch_size'],
                            shuffle=False, 
                            num_workers=model_params['num_workers'],
                            drop_last=False, 
                            collate_fn=collate_fn_classification_pre_pad)

    return train_loader, val_loader, train_dataset, val_dataset

def Get_Confusion_Matrix(epoch, all_clf_preds_val, all_clf_labels_val):
    """
    Compute and save confusion matrix for the validation set.
    Args:
        model_params (dict): Dictionary containing model configuration.
        pytorch_model (torch.nn.Module): The trained PyTorch model.
        device (torch.device): Device to run the validation on.
        epoch (int): Current epoch number for logging.
        all_clf_preds_val (list): List of predicted labels from validation set.
        all_clf_labels_val (list): List of true labels from validation set.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Navigate one level up and into "Conversion comparison"
    parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
    confusion_matrix_dir = os.path.join(
        parent_dir, 'Conversion comparison')
    os.makedirs(confusion_matrix_dir, exist_ok=True)

    # Compute confusion matrix (example)
    conf_mat = confusion_matrix(
        all_clf_labels_val, all_clf_preds_val)

    # Save the raw matrix as .npy
    npy_path = os.path.join(
        confusion_matrix_dir, f'conf_matrix_epoch_{epoch+1:03d}.npy')
    np.save(npy_path, conf_mat)

    # Optionally save a visualisation as PNG
    num_classes = conf_mat.shape[0]
    # Dynamic sizing
    scale = min(max(num_classes / 5, 5), 40)  # Clamp scale between 5 and 40
    fontsize = min(max(300 // num_classes, 5), 20)  # Clamp font size between 5 and 20

    plt.figure(figsize=(scale, scale * 0.75))  # width x height
    ax = sns.heatmap(
        conf_mat,
        annot=True,
        fmt='d',
        cmap='Blues',
        annot_kws={"size": fontsize},
        cbar=True
    )
    plt.xlabel("Predicted", fontsize=fontsize)
    plt.ylabel("Actual", fontsize=fontsize)
    plt.title(f"Confusion Matrix - Epoch {epoch+1}", fontsize=fontsize + 2)
    plt.xticks(rotation=90, fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.tight_layout()

    png_path = os.path.join(
        confusion_matrix_dir, f'conf_matrix_epoch_{epoch+1:03d}.png')
    plt.savefig(png_path)
    plt.close()

def objective(trial):
    # Define hyperparameter search space
    optuna_params = {        
        # Training-related
        "init_lr": trial.suggest_float("init_lr", 1e-5, 1e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [8, 16, 32, 64]),

        # LSTM
        "lstm_dropout": trial.suggest_float("lstm_dropout", 0.0, 0.5),
        "lstm_recurrent_dropout": trial.suggest_float("lstm_recurrent_dropout", 0.0, 0.5),

        # ReduceLROnPlateau
        "lr_min_delta": trial.suggest_float("lr_min_delta", 1e-5, 1e-2, log=True),
        "lr_factor": trial.suggest_float("lr_factor", 0.1, 0.9),
        "lr_patience": trial.suggest_int("lr_patience", 2, 10),
        "min_lr": trial.suggest_float("min_lr", 1e-7, 1e-4, log=True),

        # EarlyStopping
        "early_stopping_min_delta": trial.suggest_float("early_stopping_min_delta", 1e-5, 1e-2, log=True),
        "es_patience": trial.suggest_int("es_patience", 3, 12),
    }
    
    # Fixed params — the rest of what your model expects
    static_params = {
        "epochs": 300, # Number of training epochs
        
        # Set to 0 for no training logs, 1 for basic logs, >1 for more detailed logs
        "train_verbose": 1,
        "num_workers": 0,  # Number of workers for DataLoader, adjust based on your system
        "K": 5,  # Number of folds for cross-validation
        "path_results": "./pretrained_models_Pytorch/",
        "model_name": "Models_Therapist_Classifier",
        # "model_name": "Models_Therapist_Classifier_Block_5_4_3_2_1_0_From_Zero",

        # Convert Keras parameters to PyTorch equivalents (Set True if The model you want to fine tune is in TensorFlow/Keras format)
        "model_converter": True,

        # Use a pre-trained model (Set True if you want to use a pre-trained model)
        "use_pretrained_model": True,  # Set to True if you want to use a pre-trained model
        
        # Path to the pre-trained model in Pytorch format
        # "pretrained_model_path": "./pretrained_models_Pytorch/Models_Therapist_Classifier_Block_5_4_3_2_1/0720_0313_model_12\weights\ep002-trainloss20.46306-loss0.81176-f10.54457.pt",
        "pretrained_model_path": "./ntu_benchmark_model/model",
        
        # Path to the pre-trained model
        # "pre-trained_model": "./ntu_benchmark_model/model",  # Path to the pre-trained model for NTU-120 one-shot benchmark
        # "pre-trained_model": "./therapies_model_7/model",   # Path to the pre-trained model for the therapies dataset


        # # NTU-120 Data sets to optimize the therapy data
        # "train_annotations": "./datasets_annotations/mp_train.txt",
        # "val_annotations": "./datasets_annotations/mp_val.txt",
        "eval_therapies": True,  # Therapy data needed for its evaluation
        # "h_flip": True,
        # "skip_frames": [2, 3],

        # NTU-120 Data sets to optimize the NTU one-shot benchmark
        # "train_annotations": "./ntu_annotations/one_shot_aux_set.txt",
        # "val_annotations": "",
        # "eval_therapies": False,
        # "h_flip": False,
        # "monitor": "ntu_one_shot_acc_euc",

        "excluded_pt_keys": [
            # "encoder_net.encoder.0.residual_blocks.0.conv1.weight",
            # "encoder_net.encoder.0.residual_blocks.0.conv1.bias",
            # "encoder_net.encoder.0.residual_blocks.0.conv2.weight",
            # "encoder_net.encoder.0.residual_blocks.0.conv2.bias",
            # "encoder_net.encoder.0.residual_blocks.0.downsample.weight",
            # "encoder_net.encoder.0.residual_blocks.0.downsample.bias",
            # "encoder_net.encoder.0.residual_blocks.1.conv1.weight",
            # "encoder_net.encoder.0.residual_blocks.1.conv1.bias",
            # "encoder_net.encoder.0.residual_blocks.1.conv2.weight",
            # "encoder_net.encoder.0.residual_blocks.1.conv2.bias",
            # "encoder_net.encoder.0.residual_blocks.2.conv1.weight",
            # "encoder_net.encoder.0.residual_blocks.2.conv1.bias",
            # "encoder_net.encoder.0.residual_blocks.2.conv2.weight",
            # "encoder_net.encoder.0.residual_blocks.2.conv2.bias",
            # "encoder_net.encoder.0.residual_blocks.3.conv1.weight",
            # "encoder_net.encoder.0.residual_blocks.3.conv1.bias",
            # "encoder_net.encoder.0.residual_blocks.3.conv2.weight",
            # "encoder_net.encoder.0.residual_blocks.3.conv2.bias",
            # "encoder_net.encoder.0.residual_blocks.4.conv1.weight",
            # "encoder_net.encoder.0.residual_blocks.4.conv1.bias",
            # "encoder_net.encoder.0.residual_blocks.4.conv2.weight",
            # "encoder_net.encoder.0.residual_blocks.4.conv2.bias",
            # "encoder_net.encoder.0.residual_blocks.5.conv1.weight",
            # "encoder_net.encoder.0.residual_blocks.5.conv1.bias",
            # "encoder_net.encoder.0.residual_blocks.5.conv2.weight",
            # "encoder_net.encoder.0.residual_blocks.5.conv2.bias",
            "clf_out.weight",
            "clf_out.bias",
        ],


        "in_memory_generator_train": False,
        "in_memory_generator_val": False,
        # "in_memory_callback": True,
        
        "joints_num": 25, # 24 for MP
        "joints_dim": 3,
        "num_classes": 14, # Number of classes for classification (NTU-120 has 120, MP has 12 and Therapies has 14)
        
        # Set True to use a fitted data scaler. The one from the pre-trained models can also be used
        "scale_data": False,
        "max_seq_len": -32,
        "num_layers": 2,
        "num_neurons": 256,
        "masking": True,
        "center_skels": True,
        "scale_by_torso": True,
        "temporal_scale": [0.8, 1.2],
        "classification": True,
        "triplet": False,
        "decoder": False,
        "reverse_decoder": False,
        "clf_neurons": 0,

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

    # Combine the two
    model_params = {**static_params, **optuna_params}

    # Correct max_seq_len if negative for use in summary/testing
    if model_params['max_seq_len'] <= 0:
        # print(
        #     f"Warning: model_params['max_seq_len'] is {model_params['max_seq_len']}. Using 32 as effective_seq_len for non-dataset parts.")
        model_params['effective_seq_len'] = 32
    else:
        model_params['effective_seq_len'] = model_params['max_seq_len']

    
    # Train your model and return the validation F1 or loss
    f1_score = main(model_params)  # Use cross-validation here
    return f1_score


def main(model_params):
    train_verbose = model_params.get('train_verbose', 1)
    log_interval = model_params.get('log_interval', 10)

    # --- Path and Feature Calculation (Cleaned Up) ---
    # print("--- Initializing Parameters and Paths ---")
    model_params['path_model'] = train_utils.create_model_folder(
        model_params['path_results'], model_params['model_name']
    )
    os.makedirs(model_params['path_model'], exist_ok=True)

    # Calculate num_jcd_feats and add to model_params (for record-keeping and if get_num_feats uses it)
    try:
        model_params['num_jcd_feats'] = int(
            comb(model_params.get('joints_num', 23), 2))
    except Exception as e:
        print(
            f"Warning: Could not calculate num_jcd_feats using comb: {e}. Setting to None or a fallback.")
        model_params['num_jcd_feats'] = None  # Or a fallback value if critical

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
        print(
            f"CRITICAL Error calling get_num_feats: {e}. num_feats might be incorrect.")
        if 'num_feats' not in model_params or model_params['num_feats'] is None:
            # If get_num_feats fails and no num_feats is set, it's a critical issue.
            print("CRITICAL: num_feats is not set and get_num_feats failed. Aborting.")
            return  # Exit if essential num_feats cannot be determined

    # Save the complete model_params to JSON
    model_params_save_path = os.path.join(
        model_params['path_model'], 'model_params.json')
    try:
        with open(model_params_save_path, 'w') as f:
            json.dump(model_params, f, indent=4)
        print(f"Saved model parameters to {model_params_save_path}")
    except Exception as e:
        print(f"Error saving model_params.json: {e}")

    # print(' * Final Model params for this run:',
    #       json.dumps(model_params, indent=2))

    copy_scaler_if_needed(model_params)

    # --- Annotation File Counts (for informational purposes) ---
    num_train_files, num_val_files = 0, 0
    try:
        if model_params.get('train_annotations'):
            with open(model_params['train_annotations'], 'r') as f:
                num_train_files = len(f.read().splitlines())
        if model_params.get('val_annotations'):
            with open(model_params['val_annotations'], 'r') as f:
                num_val_files = len(f.read().splitlines())
        print(
            f"Num train annotation lines: {num_train_files}, Num val annotation lines: {num_val_files}")
    except FileNotFoundError as e:
        print(f"Warning: Annotation file not found: {e}. Counts will be 0.")
    except Exception as e:
        print(
            f"Warning: Error reading annotation files: {e}. Counts might be inaccurate.")


    """ # --- Keras-like Debugging Block ---
    print('\n* Building model (PyTorch does not require explicit build)\n')
    # No model.build() in PyTorch, it's built on first forward pass or by knowing input shape

    print('\n* Initializing inputs and outputs (PyTorch)')
    # Use parameters from model_params for dummy input size
    # Keras dummy_inpt was (batch_size, max(abs(max_seq_len), 123), num_feats)
    dummy_batch_size = model_params.get('batch_size', 32)
    dummy_seq_len_abs = abs(model_params.get('max_seq_len', -32))
    dummy_seq_len = max(dummy_seq_len_abs if dummy_seq_len_abs >
                        0 else 32, 123)  # Match Keras logic
    dummy_num_feats = model_params['num_feats']
    print(
        f"\n* Dummy input shape: (batch_size={dummy_batch_size}, seq_len={dummy_seq_len}, num_feats={dummy_num_feats})")
    dummy_inpt_np = np.random.rand(
        dummy_batch_size, dummy_seq_len, dummy_num_feats).astype(np.float32)
    print(f"\n* Dummy input shape (NumPy): {dummy_inpt_np.shape}")

    # Convert NumPy to PyTorch tensor
    dummy_inpt_torch = torch.from_numpy(dummy_inpt_np).to(device)

    pytorch_model.eval()  # Set model to evaluation mode for consistent output
    with torch.no_grad():  # Disable gradients for this test
        # PyTorch model call (equivalent to Keras model(dummy_inpt))
        dummy_pred_torch_call = pytorch_model(dummy_inpt_torch)
        if isinstance(dummy_pred_torch_call, list):
            print(' * dummy_pred_torch_call shape (PyTorch model(input)):',
                  [p.shape for p in dummy_pred_torch_call])
        elif isinstance(dummy_pred_torch_call, torch.Tensor):
            print(' * dummy_pred_torch_call shape (PyTorch model(input)):',
                  dummy_pred_torch_call.shape)
        else:
            print(' * dummy_pred_torch_call type (PyTorch model(input)):',
                  type(dummy_pred_torch_call))

        # PyTorch get_embedding call
        # Ensure get_embedding also takes a PyTorch tensor
        dummy_emb_torch = pytorch_model.get_embedding(dummy_inpt_torch)
        print(' * dummy_emb_torch shape (PyTorch get_embedding(input)):',
              dummy_emb_torch.shape)

    # If model returns a single tensor instead of a list
    if isinstance(dummy_pred_torch_call, torch.Tensor):
        dummy_pred_torch_call = [dummy_pred_torch_call]

    # Save each full output (not each sample)
    for i, out in enumerate(dummy_pred_torch_call):
        np.save(f'pytorch_output_{i}.npy', out.cpu().numpy())
        # Optional readable version
        reshaped = out.view(out.size(0), -1).cpu().numpy()
        np.savetxt(f'pytorch_output_{i}.txt', reshaped)
    """

    # --- Data Loading Therapist ---
    # Load your raw data
    raw_data_path = './datasets/therapies_dataset/'
    actions_data = pickle.load(open(os.path.join(raw_data_path, 'actions_data_v2.pckl'), 'rb'))
    video_skels = pickle.load(open(os.path.join(raw_data_path, 'video_skels_v2.pckl'), 'rb'))

    # Filter out unwanted actions (same as your TF code)
    actions_data = actions_data[~actions_data.action.isin(['no', 'si'])]
    print(f"Loaded actions_data with {len(actions_data)} entries and video_skels with {len(video_skels)} videos.")
    
    actions_data = actions_data.sort_values(by=['patient', 'session', 'video', 'ex_num'])
    
    """ actions_data, actions_data_final_val = train_test_split(
        actions_data,
        test_size=0.15,  # 15% = ~24 samples
        stratify=actions_data['action'],
        random_state=42
    )

    # Debug print for actions_data_final_val
    print("First few rows of actions_data_final_val:")
    print(actions_data_final_val.head())
    
    # How many lines have the same action in actions_data_final_val?
    action_counts_final_val = actions_data_final_val['action'].value_counts()
    print(f"Action counts in final validation set:\n{action_counts_final_val}") """
    
    # Print first few rows of actions_data for debugging
    #print("First few rows of actions_data:")
    all_actions = actions_data['action'].unique()
    #print(f"Unique actions found: {all_actions}")
    #
    ## How many lines have the same action?
    #action_counts = actions_data['action'].value_counts()
    #print(f"Action counts:\n{action_counts}")
    #
    all_actions = actions_data['action'].unique()
    action_to_idx = {action: idx for idx, action in enumerate(sorted(all_actions))}
    labels = actions_data['action'].map(action_to_idx).values
    #
    #print(f"Action to index mapping: {action_to_idx}")
    #print(f"Total unique actions: {len(all_actions)}")
    #print(f"Loaded actions_data with {len(actions_data)} entries and actions_data_final_val with {len(actions_data_final_val)} entries.")
        
    # --- Cross-Validation Setup ---
    if model_params.get("K", None) is not None:
        k_folds = model_params.get("K", 5)
        skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
        
        # --- Training Loop ---
        print("\n--- Starting PyTorch Training ---")
        fold_val_f1_scores = []
        for fold, (train_idx, val_idx) in enumerate(skf.split(actions_data, labels)):
            print(f"\n🔁 Fold {fold+1}/{k_folds}")

            train_data = actions_data.iloc[train_idx].reset_index(drop=True)
            val_data = actions_data.iloc[val_idx].reset_index(drop=True)

            # --- Create DataLoaders for this fold ---
            train_loader, val_loader, train_dataset, val_dataset = Create_Therapy_Dataloader(model_params, train_data, video_skels, val_data)
            
            # --- PyTorch Model Instantiation ---
            pytorch_model, initial_state_dict = create_pytorch_model(model_params)
            
            # --- Model Summary ---
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            pytorch_model.to(device)

            # --- PyTorch Optimizer and Loss (Mimicking Keras printouts) ---
            active_losses, loss_weights_pytorch_pt, optimizer = Setup_optimizer_and_loss(pytorch_model, model_params, device, train_dataset)
            
            tb_log_dir = os.path.join(model_params['path_model'], 'tensorboard_logs')
            model_folder = os.path.basename(os.path.dirname(tb_log_dir))

            match = re.search(r'_model_(\d+)', model_folder)
            if match:
                model_number = match.group(1)
                # print(f"Extracted model number: {model_number}")
            else:
                # print("No model number found.")
                pass

            os.makedirs(tb_log_dir, exist_ok=True)
            tb_writer = SummaryWriter(log_dir=tb_log_dir)
            print(f"TensorBoard: Logging to {tb_log_dir}")

            # --- PyTorch Training Setup ---
            # print("\n--- Starting PyTorch Training Loop ---")

            monitor_metric_name, monitor_mode, best_monitor_metric_val, weights_save_path, \
                lr_scheduler, early_stopping_patience, min_delta, early_stopping_counter = \
                    Setup_training(model_params, pytorch_model, device, optimizer)

            num_epochs = model_params.get('epochs', 1)
            # print(
            #     f"Training for {num_epochs} epochs with batch size {model_params['batch_size']}")
            # time.sleep(5)  # Small delay for readability in logs
            softmax_outputs = []  # To store softmax outputs if needed
            train_losses = []
            val_losses = []
            val_f1_scores = []
            val_auc_scores = []  # Move this to the top, before the epoch loop

            for epoch in range(num_epochs):
                epoch_start_time = time.time()
                # print(f"\nEpoch {epoch+1}/{num_epochs}")
                
                pytorch_model.train()
                running_train_loss = 0.0
                # Store individual loss components if needed for logging
                running_train_loss_clf = 0.0
                running_train_loss_triplet = 0.0

                # --- Training Phase ---
                running_train_loss, running_train_loss_clf = train_model(model_params, running_train_loss_clf, running_train_loss, pytorch_model, softmax_outputs, \
                    device, train_loader, optimizer, active_losses, loss_weights_pytorch_pt, epoch, train_verbose, log_interval)
                
                avg_epoch_train_loss = running_train_loss / \
                    len(train_loader) if len(train_loader) > 0 else 0
                train_losses.append(avg_epoch_train_loss)
                tb_writer.add_scalar('LossEpoch_Train/Total',
                                    avg_epoch_train_loss, epoch)
                if 'classification' in active_losses:
                    tb_writer.add_scalar('LossEpoch_Train/Classification', running_train_loss_clf /
                                        len(train_loader) if len(train_loader) > 0 else 0, epoch)
                if 'triplet' in active_losses:  # Add if triplet loss is calculated
                    tb_writer.add_scalar('LossEpoch_Train/Triplet', running_train_loss_triplet /
                                        len(train_loader) if len(train_loader) > 0 else 0, epoch)

                current_lr = optimizer.param_groups[0]['lr']
                tb_writer.add_scalar('LearningRate', current_lr, epoch)
                # print(
                #     f"Epoch {epoch+1} Train Summary: \
                #         Avg Total Loss: {avg_epoch_train_loss:.4f}, \
                #             LR: {current_lr}    ", end='')

                # --- Validation Phase ---
                if val_loader is not None:
                    val_metrics, running_val_loss, running_val_loss_clf, running_val_loss_triplet, \
                    all_clf_preds_val, all_clf_labels_val, all_clf_probs_val \
                        = validate_model(model_params, pytorch_model, active_losses, device, val_loader, loss_weights_pytorch_pt)
                    
                    # Calculate average val losses
                    avg_epoch_val_loss = running_val_loss / \
                        len(val_loader) if len(val_loader) > 0 else 0
                    val_metrics['val_loss'] = avg_epoch_val_loss
                    val_losses.append(avg_epoch_val_loss)
                    tb_writer.add_scalar('LossEpoch_Val/Total',
                                        avg_epoch_val_loss, epoch)

                    # AUC-ROC Calculation (if classification and probabilities available)
                    if all_clf_labels_val and all_clf_probs_val:
                        y_true = np.array(all_clf_labels_val)
                        y_scores = np.array(all_clf_probs_val)
                        
                        if y_scores.shape[1] == 2:
                            # Binary classification: take score for class 1
                            auc = roc_auc_score(y_true, y_scores[:, 1])
                        else:
                            # Multiclass classification with only present classes
                            from sklearn.preprocessing import label_binarize
                            present_classes = np.unique(y_true)
                            y_true_bin = label_binarize(y_true, classes=present_classes)
                            y_scores_filtered = y_scores[:, present_classes]
                            auc = roc_auc_score(y_true_bin, y_scores_filtered, average="macro", multi_class="ovr")

                        val_metrics['val_auc'] = auc
                        tb_writer.add_scalar('AUC_ROC/val', auc, epoch)
                        val_auc_scores.append(auc)

                    if 'classification' in active_losses:
                        avg_val_loss_clf = running_val_loss_clf / len(val_loader)
                        tb_writer.add_scalar(
                            'LossEpoch_Val/Classification', avg_val_loss_clf, epoch)
                        val_metrics['val_clf_loss'] = avg_val_loss_clf

                    if all_clf_labels_val:
                        correct_val = sum(p == t for p, t in zip(
                            all_clf_preds_val, all_clf_labels_val))
                        val_accuracy = correct_val / len(all_clf_labels_val)
                        val_metrics['val_accuracy'] = val_accuracy
                        tb_writer.add_scalar('Accuracy/val', val_accuracy, epoch)
                        # TODO: Usar Wheighted F1 Score?
                        f1_macro = f1_score(all_clf_labels_val,
                                            all_clf_preds_val, average='macro')
                        val_metrics['val_f1_macro'] = f1_macro
                        tb_writer.add_scalar('F1Score/val_macro', f1_macro, epoch)
                        val_f1_scores.append(f1_macro)

                        # print(f"Epoch {epoch+1} Val Summary:")
                        # print(f"  - Total Loss       : {avg_epoch_val_loss:.4f}")
                        # print(f"  - Accuracy         : {val_accuracy:.4f}")
                        # print(f"  - F1 Score (macro) : {f1_macro:.4f}")
                        # print(f"  - AUC-ROC          : {auc:.4f}")

                    #else:
                    #    print(
                    #        f"Epoch {epoch+1} Val Summary: Avg Total Loss: {avg_epoch_val_loss:.4f} (No classification preds for accuracy)")

                    if 'triplet' in active_losses:
                        avg_val_loss_triplet = running_val_loss_triplet / \
                            len(val_loader)
                        val_metrics['val_triplet_loss'] = avg_val_loss_triplet
                        tb_writer.add_scalar(
                            'LossEpoch_Val/Triplet', avg_val_loss_triplet, epoch)

                    if 'classification' not in active_losses:
                        print(
                            f"Epoch {epoch+1} Val Summary: Avg Total Loss: {avg_epoch_val_loss:.4f}    ", end='')

                    # --- "Callbacks" logic for this epoch ---
                    current_metric_for_scheduler_es = val_metrics.get(
                        monitor_metric_name, avg_epoch_val_loss)

                    # Step LR scheduler
                    lr_scheduler.step(current_metric_for_scheduler_es)

                    # ModelCheckpoint
                    if (monitor_mode == 'max' and (current_metric_for_scheduler_es - best_monitor_metric_val) > min_delta) or \
                            (monitor_mode == 'min' and (best_monitor_metric_val - current_metric_for_scheduler_es) > min_delta):
                        best_monitor_metric_val = current_metric_for_scheduler_es
                        # Keras format: 'ep{epoch:03d}-loss{loss:.5f}-' + monitor + '{' + monitor + ':.5f}.ckpt'
                        # Using train loss for 'loss' part of filename for consistency with Keras.
                        best_val_f1 = np.max(val_f1_scores)
                        checkpoint_filename = (
                            f"ep{epoch+1:03d}-trainloss{avg_epoch_train_loss:.5f}-"
                            f"{monitor_metric_name.replace('val_', '')}{current_metric_for_scheduler_es:.5f}-"
                            f"f1{best_val_f1:.5f}.pt"
                        )
                        full_checkpoint_path = os.path.join(weights_save_path, checkpoint_filename)
                        torch.save(pytorch_model.state_dict(), full_checkpoint_path)
                        best_checkpoint_filename = checkpoint_filename
                        # Delete all other checkpoints except the best
                        for fname in os.listdir(weights_save_path):
                            if fname.endswith('.pt') and fname != best_checkpoint_filename:
                                try:
                                    os.remove(os.path.join(weights_save_path, fname))
                                except Exception as e:
                                    print(f"Could not delete {fname}: {e}")

                        # print(
                        #     f"  Saved checkpoint: {checkpoint_filename} (Monitored '{monitor_metric_name}': {current_metric_for_scheduler_es:.5f})")
                        # Update best_val_for_early_stop if this is the metric early stopping also monitors
                        # Check if it's the same metric
                        if monitor_metric_name == model_params.get('monitor', 'val_loss'):
                            best_val_for_early_stop = best_monitor_metric_val
                            early_stopping_counter = 0
                    else:  # Only increment early stopping counter if not improving on its specific metric
                        if monitor_metric_name == model_params.get('monitor', 'val_loss'):
                            early_stopping_counter += 1

                    # Confusion Matrix (Every 10 epochs)
                    # Every 10 epochs, save confusion matrix (starting from epoch 0)
                    if epoch == 0 or (epoch + 1) % 5 == 0:
                        Get_Confusion_Matrix(epoch, all_clf_preds_val, all_clf_labels_val)
                    # EarlyStopping (check against its own best metric, which might be same as checkpointing or different)
                    if early_stopping_counter >= early_stopping_patience:
                        # print(
                        #     f"  Early stopping triggered after {early_stopping_patience} epochs without improvement on '{monitor_metric_name}'.")
                        break  # Break from epoch loop
                else:  # No val_loader
                    print("  No validation loader. Skipping validation phase, LR scheduling based on val_metrics, and early stopping.")
                    # Optionally, save model at end of epoch if no validation
                    checkpoint_filename = f"ep{epoch+1:03d}-trainloss{avg_epoch_train_loss:.5f}-no_val.pt"
                    torch.save(pytorch_model.state_dict(), os.path.join(
                        weights_save_path, checkpoint_filename))
                    print(
                        f"  Saved model checkpoint (no validation): {checkpoint_filename}")

            epoch_duration = time.time() - epoch_start_time
            # print(f"Epoch {epoch+1} duration: {epoch_duration:.2f} seconds")
            if epoch_duration > 0:
                tb_writer.add_scalar(
                    'Performance/epoch_duration_sec', epoch_duration, epoch)
            fold_val_f1_scores.append(best_val_f1)

    else:
        # --- Only Training for the best Hyperparameters using all the data ---
        print("\n--- Training with all data (no K-Fold) ---")
        
        # --- Create DataLoaders for this fold ---
        # train_loader, val_loader, train_dataset, val_dataset = Create_Therapy_Dataloader(model_params, train_data, video_skels, val_data)
        train_loader, _, train_dataset, _ = Create_Therapy_Dataloader(model_params, actions_data, video_skels, None)

        # --- PyTorch Model Instantiation ---
        pytorch_model, initial_state_dict = create_pytorch_model(model_params)
        
        # --- Model Summary ---
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pytorch_model.to(device)

        # --- PyTorch Optimizer and Loss (Mimicking Keras printouts) ---
        active_losses, loss_weights_pytorch_pt, optimizer = Setup_optimizer_and_loss(pytorch_model, model_params, device, train_dataset)
        
        tb_log_dir = os.path.join(model_params['path_model'], 'tensorboard_logs')
        model_folder = os.path.basename(os.path.dirname(tb_log_dir))

        match = re.search(r'_model_(\d+)', model_folder)
        if match:
            model_number = match.group(1)
            print(f"Extracted model number: {model_number}")
        else:
            print("No model number found.")

        os.makedirs(tb_log_dir, exist_ok=True)
        tb_writer = SummaryWriter(log_dir=tb_log_dir)
        print(f"TensorBoard: Logging to {tb_log_dir}")

        # --- PyTorch Training Setup ---
        # print("\n--- Starting PyTorch Training Loop ---")

        monitor_metric_name, monitor_mode, best_monitor_metric_val, weights_save_path, \
            lr_scheduler, early_stopping_patience, min_delta, early_stopping_counter = \
                Setup_training(model_params, pytorch_model, device, optimizer)
        
        num_epochs = model_params.get('epochs', 1)
        # print(
        #     f"Training for {num_epochs} epochs with batch size {model_params['batch_size']}")
        # time.sleep(5)  # Small delay for readability in logs
        softmax_outputs = []  # To store softmax outputs if needed
        train_losses = []
        val_losses = []
        val_f1_scores = []
        val_auc_scores = []  # Move this to the top, before the epoch loop

        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            pytorch_model.train()
            running_train_loss = 0.0
            # Store individual loss components if needed for logging
            running_train_loss_clf = 0.0
            running_train_loss_triplet = 0.0

            # --- Training Phase ---
            running_train_loss, running_train_loss_clf = train_model(model_params, running_train_loss_clf, running_train_loss, pytorch_model, softmax_outputs, \
                device, train_loader, optimizer, active_losses, loss_weights_pytorch_pt, epoch, train_verbose, log_interval)
            
            avg_epoch_train_loss = running_train_loss / \
                len(train_loader) if len(train_loader) > 0 else 0
            train_losses.append(avg_epoch_train_loss)
            tb_writer.add_scalar('LossEpoch_Train/Total',
                                avg_epoch_train_loss, epoch)
            if 'classification' in active_losses:
                tb_writer.add_scalar('LossEpoch_Train/Classification', running_train_loss_clf /
                                    len(train_loader) if len(train_loader) > 0 else 0, epoch)
            if 'triplet' in active_losses:  # Add if triplet loss is calculated
                tb_writer.add_scalar('LossEpoch_Train/Triplet', running_train_loss_triplet /
                                    len(train_loader) if len(train_loader) > 0 else 0, epoch)

            current_lr = optimizer.param_groups[0]['lr']
            tb_writer.add_scalar('LearningRate', current_lr, epoch)
            print(
                f"Epoch {epoch+1} Train Summary: Avg Total Loss: {avg_epoch_train_loss:.4f}, LR: {current_lr}")

            # --- Validation Phase ---
            if val_loader is not None:
                val_metrics, running_val_loss, running_val_loss_clf, running_val_loss_triplet, \
                    all_clf_preds_val, all_clf_labels_val, all_clf_probs_val \
                        = validate_model(model_params, pytorch_model, active_losses, device, val_loader, loss_weights_pytorch_pt)
                
                # Calculate average val losses
                avg_epoch_val_loss = running_val_loss / \
                    len(val_loader) if len(val_loader) > 0 else 0
                val_metrics['val_loss'] = avg_epoch_val_loss
                val_losses.append(avg_epoch_val_loss)
                tb_writer.add_scalar('LossEpoch_Val/Total',
                                    avg_epoch_val_loss, epoch)

                # AUC-ROC Calculation (if classification and probabilities available)
                if all_clf_labels_val and all_clf_probs_val:
                    y_true = np.array(all_clf_labels_val)
                    y_scores = np.array(all_clf_probs_val)
                    
                    if y_scores.shape[1] == 2:
                        # Binary classification: take score for class 1
                        auc = roc_auc_score(y_true, y_scores[:, 1])
                    else:
                        # Multiclass classification with only present classes
                        from sklearn.preprocessing import label_binarize
                        present_classes = np.unique(y_true)
                        y_true_bin = label_binarize(y_true, classes=present_classes)
                        y_scores_filtered = y_scores[:, present_classes]
                        auc = roc_auc_score(y_true_bin, y_scores_filtered, average="macro", multi_class="ovr")

                    val_metrics['val_auc'] = auc
                    tb_writer.add_scalar('AUC_ROC/val', auc, epoch)
                    val_auc_scores.append(auc)

                if 'classification' in active_losses:
                    avg_val_loss_clf = running_val_loss_clf / len(val_loader)
                    tb_writer.add_scalar(
                        'LossEpoch_Val/Classification', avg_val_loss_clf, epoch)
                    val_metrics['val_clf_loss'] = avg_val_loss_clf

                if all_clf_labels_val:
                    correct_val = sum(p == t for p, t in zip(
                        all_clf_preds_val, all_clf_labels_val))
                    val_accuracy = correct_val / len(all_clf_labels_val)
                    val_metrics['val_accuracy'] = val_accuracy
                    tb_writer.add_scalar('Accuracy/val', val_accuracy, epoch)
                    # TODO: Usar Wheighted F1 Score?
                    f1_macro = f1_score(all_clf_labels_val,
                                        all_clf_preds_val, average='macro')
                    val_metrics['val_f1_macro'] = f1_macro
                    tb_writer.add_scalar('F1Score/val_macro', f1_macro, epoch)
                    val_f1_scores.append(f1_macro)

                    print(f"Epoch {epoch+1} Val Summary:")
                    print(f"  - Total Loss       : {avg_epoch_val_loss:.4f}")
                    print(f"  - Accuracy         : {val_accuracy:.4f}")
                    print(f"  - F1 Score (macro) : {f1_macro:.4f}")
                    print(f"  - AUC-ROC          : {auc:.4f}")

                else:
                    print(
                        f"Epoch {epoch+1} Val Summary: Avg Total Loss: {avg_epoch_val_loss:.4f} (No classification preds for accuracy)")

                if 'triplet' in active_losses:
                    avg_val_loss_triplet = running_val_loss_triplet / \
                        len(val_loader)
                    val_metrics['val_triplet_loss'] = avg_val_loss_triplet
                    tb_writer.add_scalar(
                        'LossEpoch_Val/Triplet', avg_val_loss_triplet, epoch)

                if 'classification' not in active_losses:
                    print(
                        f"Epoch {epoch+1} Val Summary: Avg Total Loss: {avg_epoch_val_loss:.4f}")

                # --- "Callbacks" logic for this epoch ---
                current_metric_for_scheduler_es = val_metrics.get(
                    monitor_metric_name, avg_epoch_val_loss)

                # Step LR scheduler
                lr_scheduler.step(current_metric_for_scheduler_es)

                # ModelCheckpoint
                if (monitor_mode == 'max' and (current_metric_for_scheduler_es - best_monitor_metric_val) > min_delta) or \
                        (monitor_mode == 'min' and (best_monitor_metric_val - current_metric_for_scheduler_es) > min_delta):
                    best_monitor_metric_val = current_metric_for_scheduler_es
                    # Keras format: 'ep{epoch:03d}-loss{loss:.5f}-' + monitor + '{' + monitor + ':.5f}.ckpt'
                    # Using train loss for 'loss' part of filename for consistency with Keras.
                    best_val_f1 = np.max(val_f1_scores)
                    checkpoint_filename = (
                        f"ep{epoch+1:03d}-trainloss{avg_epoch_train_loss:.5f}-"
                        f"{monitor_metric_name.replace('val_', '')}{current_metric_for_scheduler_es:.5f}-"
                        f"f1{best_val_f1:.5f}.pt"
                    )
                    full_checkpoint_path = os.path.join(weights_save_path, checkpoint_filename)
                    torch.save(pytorch_model.state_dict(), full_checkpoint_path)
                    best_checkpoint_filename = checkpoint_filename

                    print(
                        f"  Saved checkpoint: {checkpoint_filename} (Monitored '{monitor_metric_name}': {current_metric_for_scheduler_es:.5f})")
                    # Update best_val_for_early_stop if this is the metric early stopping also monitors
                    # Check if it's the same metric
                    if monitor_metric_name == model_params.get('monitor', 'val_loss'):
                        best_val_for_early_stop = best_monitor_metric_val
                        early_stopping_counter = 0
                else:  # Only increment early stopping counter if not improving on its specific metric
                    if monitor_metric_name == model_params.get('monitor', 'val_loss'):
                        early_stopping_counter += 1

                # Confusion Matrix (Every 10 epochs)
                # Every 10 epochs, save confusion matrix (starting from epoch 0)
                if epoch == 0 or (epoch + 1) % 5 == 0:
                    Get_Confusion_Matrix(epoch, all_clf_preds_val, all_clf_labels_val)
                # EarlyStopping (check against its own best metric, which might be same as checkpointing or different)
                if early_stopping_counter >= early_stopping_patience:
                    # print(
                    #     f"  Early stopping triggered after {early_stopping_patience} epochs without improvement on '{monitor_metric_name}'.")
                    break  # Break from epoch loop
            else:  # No val_loader
                print("  No validation loader. Skipping validation phase, LR scheduling based on val_metrics, and early stopping.")
                # Optionally, save model at end of epoch if no validation
                checkpoint_filename = (
                    f"Best_Model-ep{epoch+1:03d}-trainloss{avg_epoch_train_loss:.5f}-"
                    f"f1{model_params.get('best_val_f1', 0):.5f}.pt"
                )
                torch.save(pytorch_model.state_dict(), os.path.join(
                    weights_save_path, checkpoint_filename))
                print(
                    f"  Saved model checkpoint (no validation): {checkpoint_filename}")

        epoch_duration = time.time() - epoch_start_time
        # print(f"Epoch {epoch+1} duration: {epoch_duration:.2f} seconds")
        if epoch_duration > 0:
            tb_writer.add_scalar(
                'Performance/epoch_duration_sec', epoch_duration, epoch)
            
        fold_val_f1_scores.append(best_val_f1)

    # Determine folder one level up
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
    metrics_save_dir = os.path.join(parent_dir, 'Conversion comparison')
    os.makedirs(metrics_save_dir, exist_ok=True)

    # 1. Find the best values (assuming best means minimum loss, maximum score)
    best_train_loss = np.min(train_losses)
    best_val_loss = np.min(val_losses)
    best_val_f1 = np.max(val_f1_scores)
    best_val_auc = np.max(val_auc_scores)

    # 2. Create filename with best values embedded (rounded for readability)
    filename = (
        f"pytorch_therapy_classifier_train_loss-{best_train_loss:.4f}_"
        f"val_loss-{best_val_loss:.4f}_"
        f"val_f1-{best_val_f1:.4f}_"
        f"val_auc-{best_val_auc:.4f}_"
        f"model_{model_number}.npz"
    )
    
    # 3. Save arrays with this filename
    np.savez(os.path.join(metrics_save_dir, filename),
            train_losses=np.array(train_losses),
            val_losses=np.array(val_losses),
            val_f1_scores=np.array(val_f1_scores),
            val_auc_scores=np.array(val_auc_scores),
            )

    # tb_writer.close()

    # final_state_dict = pytorch_model.state_dict()

    # changed_keys = []

    # print("\n--- PyTorch Training Summary ---")
    # print(f"Total epochs: {num_epochs}")
    
    # for key in initial_state_dict:
    #     if not torch.equal(initial_state_dict[key], final_state_dict[key].cpu()):
    #         changed_keys.append(key)
    # for key in excluded_pt_keys:
    #     if not torch.equal(initial_state_dict[key], final_state_dict[key].cpu()):
    #         print(f"OK: Layer {key} changed!")
    #     else:
    #         print(f"Warning: Frozen layer {key} changed.")
# 
    # print("\n")
    # print("Weights that changed during training:")
    # for key in changed_keys:
    #     print(f" - {key}")
    # print("\n")
# 
    # for key in changed_keys:
    #     diff = torch.norm(final_state_dict[key].cpu() - initial_state_dict[key].cpu()).item()
    #     print(f"{key}: Δ = {diff:.6f}")
# 
    print("\n--- PyTorch Training Finished ---")
    
    return best_val_f1

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
    print("GPU name:", torch.cuda.get_device_name(0)
          if torch.cuda.is_available() else "No GPU found")
    time.sleep(2)  # Small delay for readability in logs
    
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--epochs', type=int, help='Number of training epochs')
    # parser.add_argument('--init_lr', type=float, help='Initial learning rate')
    # parser.add_argument('--batch_size', type=int, help='Batch size')
    # parser.add_argument('--train_annotations', type=str, help='Path to train annotations')
    # parser.add_argument('--val_annotations', type=str, help='Path to val annotations')
    # Add more if you want — just match the key names in model_params
    #
    # args = parser.parse_args()

    #model_params = {
    #    # Set to 0 for no training logs, 1 for basic logs, >1 for more detailed logs
    #    "train_verbose": 1,
    #    "num_workers": 0,  # Number of workers for DataLoader, adjust based on your system
    #    "path_results": "./pretrained_models_Pytorch/",
    #    "epochs": 300, # Number of training epochs
#
    #    # Convert Keras parameters to PyTorch equivalents (Set True if The model you want to fine tune is in TensorFlow/Keras format)
    #    "model_converter": True,
#
    #    # Path to the pre-trained model in Pytorch format
    #    "pretrained_model_path": "./pretrained_models_Pytorch/TCN_Models_Therapist_Only_First_Layer_new_classifier/0718_2035_model_52",  
    #    
    #    # Path to the pre-trained model
    #    # "pre-trained_model": "./ntu_benchmark_model/model",  # Path to the pre-trained model for NTU-120 one-shot benchmark
    #    # "pre-trained_model": "./therapies_model_7/model",   # Path to the pre-trained model for the therapies dataset
#
    #    # Path to save the model and results
    #    "path_model": "./TCN_Models_Therapist_Only_First_Layer/",
#
    #    # # NTU-120 Data sets to optimize the therapy data
    #    # "train_annotations": "./datasets_annotations/mp_train.txt",
    #    # "val_annotations": "./datasets_annotations/mp_val.txt",
    #    "eval_therapies": True,  # Therapy data needed for its evaluation
    #    # "h_flip": True,
    #    # "skip_frames": [2, 3],
#
    #    # NTU-120 Data sets to optimize the NTU one-shot benchmark
    #    # "train_annotations": "./ntu_annotations/one_shot_aux_set.txt",
    #    # "val_annotations": "",
    #    # "eval_therapies": False,
    #    # "h_flip": False,
    #    # "monitor": "ntu_one_shot_acc_euc",
#
    #    "in_memory_generator_train": False,
    #    "in_memory_generator_val": False,
    #    # "in_memory_callback": True,
#
    #    
    #    "joints_num": 25, # 24 for MP
    #    "joints_dim": 3,
    #    "num_classes": 14, # Number of classes for classification (NTU-120 has 120, MP has 12 and Therapies has 14)
#
    #    "batch_size": 8,
    #    "init_lr": 0.001,
    #    "lstm_recurrent_dropout": 0.0,
    #    "lstm_dropout": 0.2,
#
    #    # Set True to use a fitted data scaler. The one from the pre-trained models can also be used
    #    "scale_data": False,
    #    "max_seq_len": -32,
    #    "num_layers": 2,
    #    "num_neurons": 256,
    #    "masking": True,
    #    "center_skels": True,
    #    "scale_by_torso": True,
    #    "temporal_scale": [0.8, 1.2],
    #    "classification": True,
    #    "triplet": False,
    #    "decoder": False,
    #    "reverse_decoder": False,
    #    "clf_neurons": 0,
#
    #    "model_name": "TCN_Models_Therapist_Only_First_Layer_new_classifier",
    #    "conv_params": [256, 4, 2, True, "causal", [4]],
    #    "is_tcn": False,
    #    "use_jcd_features": True,
    #    "use_speeds": False,
    #    "use_coords_raw": False,
    #    "use_coords": True,
    #    "use_jcd_diff": False,
    #    "use_bone_angles": True,
    #    "use_bone_angles_cent": False,
    #    "average_wrong_skels": True,
    #    "average_wrong_skels_method": 'mean',
    #}

    ## Correct max_seq_len if negative for use in summary/testing
    #if model_params['max_seq_len'] <= 0:
    #    print(
    #        f"Warning: model_params['max_seq_len'] is {model_params['max_seq_len']}. Using 32 as effective_seq_len for non-dataset parts.")
    #    model_params['effective_seq_len'] = 32
    #else:
    #    model_params['effective_seq_len'] = model_params['max_seq_len']

    #--- Call the main function ---
    #main(model_params)
    
    
    study = optuna.create_study(direction="maximize")  # Or "minimize" for loss
    study.optimize(objective, n_trials=4)  # Try 30 different combinations

    print("Best hyperparameters:")
    print(study.best_params)
    
    # Determine folder one level up
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
    metrics_save_dir = os.path.join(parent_dir, 'Conversion comparison')
    os.makedirs(metrics_save_dir, exist_ok=True)
    
    best_params = study.best_trial.params
    
    print(f"Best parameters found: {best_params}")
    
    # Save best parameters to a JSON file
    import json
    with open(os.path.join(metrics_save_dir, 'best_hyperparams.json'), 'w') as f:
        json.dump(best_params, f, indent=4)
    
    # TODO: Change this after each run to avoid overwriting
    # Fixed params — the rest of what your model expects
    static_params = {
        "best_val_f1": study.best_value,
        
        "epochs": 300, # Number of training epochs
        
        # Set to 0 for no training logs, 1 for basic logs, >1 for more detailed logs
        "train_verbose": 1,
        "num_workers": 0,  # Number of workers for DataLoader, adjust based on your system
        "K": 5,  # Number of folds for cross-validation
        "path_results": "./pretrained_models_Pytorch/",
        "model_name": "Models_Therapist_Classifier",
        # "model_name": "Models_Therapist_Classifier_Block_5_4_3_2_1_0_From_Zero",

        # Convert Keras parameters to PyTorch equivalents (Set True if The model you want to fine tune is in TensorFlow/Keras format)
        "model_converter": True,

        # Use a pre-trained model (Set True if you want to use a pre-trained model)
        "use_pretrained_model": True,  # Set to True if you want to use a pre-trained model
        
        # Path to the pre-trained model in Pytorch format
        # "pretrained_model_path": "./pretrained_models_Pytorch/Models_Therapist_Classifier_Block_5_4_3_2_1/0720_0313_model_12\weights\ep002-trainloss20.46306-loss0.81176-f10.54457.pt",
        
        # Path to the pre-trained model
        "pre-trained_model": "./ntu_benchmark_model/model",  # Path to the pre-trained model for NTU-120 one-shot benchmark
        # "pre-trained_model": "./therapies_model_7/model",   # Path to the pre-trained model for the therapies dataset


        # # NTU-120 Data sets to optimize the therapy data
        # "train_annotations": "./datasets_annotations/mp_train.txt",
        # "val_annotations": "./datasets_annotations/mp_val.txt",
        "eval_therapies": True,  # Therapy data needed for its evaluation
        # "h_flip": True,
        # "skip_frames": [2, 3],

        # NTU-120 Data sets to optimize the NTU one-shot benchmark
        # "train_annotations": "./ntu_annotations/one_shot_aux_set.txt",
        # "val_annotations": "",
        # "eval_therapies": False,
        # "h_flip": False,
        # "monitor": "ntu_one_shot_acc_euc",

        "excluded_pt_keys": [
            # "encoder_net.encoder.0.residual_blocks.0.conv1.weight",
            # "encoder_net.encoder.0.residual_blocks.0.conv1.bias",
            # "encoder_net.encoder.0.residual_blocks.0.conv2.weight",
            # "encoder_net.encoder.0.residual_blocks.0.conv2.bias",
            # "encoder_net.encoder.0.residual_blocks.0.downsample.weight",
            # "encoder_net.encoder.0.residual_blocks.0.downsample.bias",
            # "encoder_net.encoder.0.residual_blocks.1.conv1.weight",
            # "encoder_net.encoder.0.residual_blocks.1.conv1.bias",
            # "encoder_net.encoder.0.residual_blocks.1.conv2.weight",
            # "encoder_net.encoder.0.residual_blocks.1.conv2.bias",
            # "encoder_net.encoder.0.residual_blocks.2.conv1.weight",
            # "encoder_net.encoder.0.residual_blocks.2.conv1.bias",
            # "encoder_net.encoder.0.residual_blocks.2.conv2.weight",
            # "encoder_net.encoder.0.residual_blocks.2.conv2.bias",
            # "encoder_net.encoder.0.residual_blocks.3.conv1.weight",
            # "encoder_net.encoder.0.residual_blocks.3.conv1.bias",
            # "encoder_net.encoder.0.residual_blocks.3.conv2.weight",
            # "encoder_net.encoder.0.residual_blocks.3.conv2.bias",
            # "encoder_net.encoder.0.residual_blocks.4.conv1.weight",
            # "encoder_net.encoder.0.residual_blocks.4.conv1.bias",
            # "encoder_net.encoder.0.residual_blocks.4.conv2.weight",
            # "encoder_net.encoder.0.residual_blocks.4.conv2.bias",
            # "encoder_net.encoder.0.residual_blocks.5.conv1.weight",
            # "encoder_net.encoder.0.residual_blocks.5.conv1.bias",
            # "encoder_net.encoder.0.residual_blocks.5.conv2.weight",
            # "encoder_net.encoder.0.residual_blocks.5.conv2.bias",
            # "clf_out.weight",
            # "clf_out.bias",
        ],


        "in_memory_generator_train": False,
        "in_memory_generator_val": False,
        # "in_memory_callback": True,
        
        "joints_num": 25, # 24 for MP
        "joints_dim": 3,
        "num_classes": 14, # Number of classes for classification (NTU-120 has 120, MP has 12 and Therapies has 14)
        
        # Set True to use a fitted data scaler. The one from the pre-trained models can also be used
        "scale_data": False,
        "max_seq_len": -32,
        "num_layers": 2,
        "num_neurons": 256,
        "masking": True,
        "center_skels": True,
        "scale_by_torso": True,
        "temporal_scale": [0.8, 1.2],
        "classification": True,
        "triplet": False,
        "decoder": False,
        "reverse_decoder": False,
        "clf_neurons": 0,

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

    # Combine the two
    model_params = {**static_params, **best_params}

    main(best_params)

    print("\n--- Training script finished ---")

    exit(0)
    
    import os
    import re

    # === CONFIGURATION ===
    checkpoint_dir = './pretrained_models_Pytorch/Models_Therapist_Classifier_Block_5_4_3_2_1_0_From_Zero/'
    metric_key = 'f1'  # 'loss' or 'f1'
    opt_mode = 'max'     # 'min' for loss, 'max' for f1

    # Pattern now supports trailing f1 or any other trailing field
    pattern = re.compile(
        r'ep(\d+)-trainloss([0-9.]+)-loss([0-9.]+)-f1([0-9.]+)\.pt'
    )

    # === SEARCH ===
    best_val = float('inf') if opt_mode == 'min' else -float('inf')
    best_checkpoint_path = None

    for dirpath, _, filenames in os.walk(checkpoint_dir):
        print(f"Searching in directory: {dirpath}")  # ← ADD THIS
        print(f"Files found: {filenames}")  # ← ADD THIS
        for fname in filenames:
            print(f"Checking file: {fname}")  # ← ADD THIS
            if fname.endswith('.pt'):
                print(f"Found .pt file: {fname}")  # ← ADD THIS
                match = pattern.match(fname)
                if match:
                    epoch = int(match.group(1))
                    train_loss = float(match.group(2))
                    val_loss = float(match.group(3))
                    val_f1 = float(match.group(4))

                    # Choose which metric to evaluate based on metric_key
                    val_metric = val_loss if metric_key == 'loss' else val_f1
                    is_better = (val_metric < best_val) if opt_mode == 'min' else (val_metric > best_val)
                    if is_better:
                        best_val = val_metric
                        best_checkpoint_path = os.path.join(dirpath, fname)

    # === RESULT ===
    if best_checkpoint_path:
        print(f"\n✅ Best checkpoint found:")
        print(f"   Path : {best_checkpoint_path}")
        print(f"   {metric_key} = {best_val:.5f}")
    else:
        print(f"\n❌ No valid checkpoints found for metric '{metric_key}'.")
