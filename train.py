#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 18:10:29 2020

@author: asabater
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim # PyTorch Optimizer
from torch.utils.data import Dataset, DataLoader # PyTorch Data Handling
from torch.utils.tensorboard import SummaryWriter # PyTorch TensorBoard
from torchinfo import summary # For model summary: pip install torchinfo

from scipy.special import comb
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, LambdaCallback
from torchviz import make_dot
import json

from data_generator import triplet_data_generator, get_scaler_filename, get_num_feats
from train_callbacks import get_lr_metric  # eval_one_shot_callback, eval_one_shot_therapies_callback, 
import train_utils
from shutil import copyfile

from models.TCN_classifier import TCN_clf
# tf.config.experimental_run_functions_eagerly(True)

from dataset_scripts.ntu120_utils.triplet_ntu_callback import eval_ntu_one_shot_triplets_callback
from dataset_scripts.therapies.triplet_therapies_callback import eval_therapies_triplet_callback

from remove_suboptimal_weights import remove_path_weights



# Seed PyTorch
torch.manual_seed(123)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(123)
np.random.seed(123)
tf.random.set_seed(123) # Remove TF seed



def main(model_params):
    
    train_verbose = 1

    model_params.update({
                'path_model': train_utils.create_model_folder(model_params['path_results'], model_params['model_name']),
                'num_jcd_feats': int(comb(model_params['joints_num'],2)), 
                'num_feats': int(comb(model_params['joints_num'],2)) + model_params['joints_dim']*model_params['joints_num'],
            })
    model_params['num_feats'] = get_num_feats(**model_params)
    json.dump(model_params, open(model_params['path_model']+'model_params.json', 'w'))
    
    print(' * Model params:', model_params)    
    
    with open(model_params['train_annotations'], 'r') as f: num_train_files = len(f.read().splitlines())
    if model_params['val_annotations']  == '': num_val_files = 0
    else:
        with open(model_params['val_annotations'], 'r') as f: num_val_files = len(f.read().splitlines())
    
    print(num_train_files, num_val_files)
    
    if model_params['scale_data']:
        scaler_filename = get_scaler_filename(**model_params)
        copyfile(scaler_filename, model_params['path_model'] + '/scaler.pckl')    
    
    #model = TCN_clf(**model_params)
    
    # --- Instantiate PyTorch Model ---
    print("Creating PyTorch TCN_clf model...")
    # Make sure all necessary params from model_params are passed correctly
    model = TCN_clf(
        num_feats=model_params['num_feats'],
        conv_params=model_params['conv_params'],
        lstm_dropout=model_params['lstm_dropout'],
        masking=model_params['masking'],
        triplet=model_params['triplet'],             # Use value from params
        classification=model_params['classification'], # Use value from params
        clf_neurons=model_params['clf_neurons'],
        num_classes=model_params['num_classes']
        # Add other relevant params from model_params if needed by TCN_clf __init__
        # prediction_mode=False # Default for training/testing like this
    )

    # --- Model Testing ---
    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Model moved to device: {device}")

    batch_size = 2
    seq_len = 32 # Use a positive sequence length for testing
    num_feats = model_params['num_feats'] # Use the calculated features

    print(f"Creating test input: batch={batch_size}, seq_len={seq_len}, feats={num_feats}")
    # Dummy input tensor (random float values) - Shape (N, L, C)
    test_input = torch.randn(batch_size, seq_len, num_feats, device=device)

    print(f"Input shape: {test_input.shape}")
    
    # Set model to evaluation mode for testing inference
    model.eval()
    with torch.no_grad(): # Disable gradients for inference
        try:
            output = model(test_input)
            print("Model forward pass successful.")
            # Check output type and shape
            if isinstance(output, list):
                print(f"Output is a list with {len(output)} tensors.")
                print("Output shapes:", [o.shape for o in output])
                print("Output devices:", [o.device for o in output])
            elif isinstance(output, torch.Tensor):
                print(f"Output shape: {output.shape}")
                print(f"Output device: {output.device}")
            else:
                print(f"Output type: {type(output)}")

            # Test get_embedding
            print("\nTesting get_embedding...")
            embedding = model.get_embedding(test_input) # Uses batch=None internally
            print("get_embedding successful.")
            print(f"Embedding shape: {embedding.shape}")
            print(f"Embedding device: {embedding.device}")

            # Test get_embedding with batch > 0 (if needed and implemented)
            # print("\nTesting get_embedding with batch=10...")
            # embedding_batch = model.get_embedding(test_input, batch=10)
            # print("get_embedding (batch>0) potentially successful.")
            # print(f"Embedding (batch>0) shape: {embedding_batch.shape}")


        except Exception as e:
            print("\n !!! Error during model forward pass or embedding !!!")
            print(e)
            import traceback
            traceback.print_exc()

    # Create a visualization of the computational graph
    dot = make_dot(output, params=dict(model.named_parameters()))

    # Save or display the generated graph
    dot.format = 'png'
    dot.render('larger_net')

    exit()
    
    # ==================================================================================
    # PyTorch Setup for Training (equivalent to Keras compile and callbacks setup)
    # ==================================================================================

    print("\n--- Setting up PyTorch Training Components ---")

    # 1. Optimizer (Keras: Adam(model_params['init_lr'], clipnorm=1.))
    optimizer = optim.Adam(model.parameters(), lr=model_params['init_lr'])
    print(f"Optimizer: Adam with LR={model_params['init_lr']}")
    # Gradient clipping (max_norm=1.0) will be applied in the training loop:
    # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    # 2. Loss Functions (Keras: losses['output_1'] = tf.keras.losses.CategoricalCrossentropy())
    #    (Keras: loss_weights['output_1'] = 0.4)
    active_losses = {}
    if model_params.get('classification', True):
        criterion_clf = nn.CrossEntropyLoss() # Expects raw logits and integer labels
        active_losses['classification'] = criterion_clf
        print("Loss: Classification - CrossEntropyLoss")
    if model_params.get('triplet', False):
        criterion_triplet = nn.TripletMarginLoss(margin=model_params.get('triplet_margin', 1.0), p=2)
        active_losses['triplet'] = criterion_triplet
        print(f"Loss: Triplet - TripletMarginLoss with margin={model_params.get('triplet_margin', 1.0)}")

    # Loss weights (applied manually in training loop)
    loss_weights_pytorch = {
        'classification': model_params.get('clf_loss_weight', 0.4 if 'classification' in active_losses else 0.0), # Defaulting to Keras example
        'triplet': model_params.get('triplet_loss_weight', 0.6 if 'triplet' in active_losses else 0.0) # Assuming triplet takes the rest
    }
    print(f"Loss Weights (PyTorch): {loss_weights_pytorch}")


    # 3. Metrics (Keras: metrics = [ 'accuracy', get_lr_metric(optimizer) ])
    #    In PyTorch, metrics are calculated manually.
    #    - Accuracy: Calculated in validation loop.
    #    - Learning Rate: Logged from optimizer.param_groups[0]['lr'].
    print("Metrics: Accuracy (manual), Learning Rate (manual log)")


    # 4. Model Summary (Keras: model.summary(100))
    try:
        # Use the same shapes as the test input for summary
        summary_input_shape = (batch_size, seq_len, num_feats)
        print(f"\nModel Summary (input shape for summary: {summary_input_shape}):")
        summary(model, input_size=summary_input_shape, col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"], verbose=0)
    except Exception as e:
        print(f"Could not print model summary using torchinfo: {e}")
        print("Basic model structure:\n", model)


    # 5. "Callbacks" Setup
    #    - TensorBoard (Keras: TensorBoard(...))
    tb_log_dir = os.path.join(model_params['path_model'], 'tensorboard_logs')
    os.makedirs(tb_log_dir, exist_ok=True)
    tb_writer = SummaryWriter(log_dir=tb_log_dir)
    print(f"TensorBoard: Logging to {tb_log_dir}")

    #    - ModelCheckpoint (Keras: ModelCheckpoint(...))
    #      Manual implementation variables
    monitor_metric_name = model_params.get('monitor', 'val_loss') # e.g., 'val_loss', 'val_accuracy'
    # Keras uses 'val_loss', 'ntu_one_shot_acc_euc'. For PyTorch, map these to keys you'll use in your metrics dict.
    # Example: if Keras 'ntu_one_shot_acc_euc' -> PyTorch 'val_ntu_one_shot_acc'
    monitor_mode = 'min' if 'loss' in monitor_metric_name else 'max' # min for loss, max for accuracy
    if model_params.get('min_monitor', False) is False and monitor_mode == 'min': # Keras 'min_monitor': False for accuracy
        monitor_mode = 'max' # if min_monitor is False, it means we want to maximize the metric
    
    best_monitor_metric_val = float('-inf') if monitor_mode == 'max' else float('inf')
    weights_save_path = os.path.join(model_params['path_model'], 'weights')
    os.makedirs(weights_save_path, exist_ok=True)
    print(f"Model Checkpoints: Monitoring '{monitor_metric_name}' ({monitor_mode} is better). Saving to {weights_save_path}")

    #    - ReduceLROnPlateau (Keras: ReduceLROnPlateau(...))
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                        mode=monitor_mode,
                                                        factor=0.1,
                                                        patience=model_params.get('lr_patience', 3), # Keras default for ReduceLROnPlateau is 10, TF is often 3-5
                                                        verbose=True,
                                                        min_lr=1e-7) # min_delta equivalent is tricky, not direct. Threshold is used.
    print(f"LR Scheduler: ReduceLROnPlateau on '{monitor_metric_name}'")

    #    - EarlyStopping (Keras: EarlyStopping(...))
    #      Manual implementation variables
    early_stopping_patience = model_params.get('es_patience', 6) # Keras default 10, TF example 6
    early_stopping_counter = 0
    best_val_for_early_stop = float('-inf') if monitor_mode == 'max' else float('inf')
    print(f"Early Stopping: Patience {early_stopping_patience} on '{monitor_metric_name}'")


    # 6. No `model.compile()` in PyTorch. The above setup replaces it.

    # 7. Initial Model Saving (Keras `model.save(...)` before fit is less common)
    #    Usually, models are saved during/after training via checkpoints.
    #    If you want to save the initial state:
    #    torch.save(model.state_dict(), os.path.join(model_params['path_model'], 'model_initial_state.pt'))
    #    print("Saved initial model state_dict.")


    # 8. Data Loading (Keras: triplet_data_generator -> PyTorch Dataset & DataLoader)
    print("\n--- Setting up PyTorch DataLoaders ---")
    # You need to fully implement TripletPoseDataset based on your Keras generator's logic
    # Pass all relevant model_params to the Dataset constructor if it needs them for data processing
    dataset_params_for_loader = model_params.copy() # Pass a copy to avoid modification issues
    
    train_dataset = TripletPoseDataset(pose_annotations_file=model_params['train_annotations'],
                                       validation_mode=False,
                                       in_memory=model_params['in_memory_generator_train'],
                                       **dataset_params_for_loader) # Pass all params
    train_loader = DataLoader(train_dataset,
                              batch_size=model_params['batch_size'],
                              shuffle=True,
                              num_workers=model_params.get('num_workers', 2), # Add num_workers to model_params if desired
                              pin_memory=True if device.type == 'cuda' else False)
    print(f"Train DataLoader: Batch size {model_params['batch_size']}, Num samples approx {len(train_dataset)}")

    val_loader = None
    if model_params['val_annotations'] and model_params['val_annotations'] != '':
        val_dataset = TripletPoseDataset(pose_annotations_file=model_params['val_annotations'],
                                         validation_mode=True,
                                         in_memory=model_params['in_memory_generator_val'],
                                         **dataset_params_for_loader)
        val_loader = DataLoader(val_dataset,
                                batch_size=model_params['batch_size'],
                                shuffle=False,
                                num_workers=model_params.get('num_workers', 2),
                                pin_memory=True if device.type == 'cuda' else False)
        print(f"Validation DataLoader: Batch size {model_params['batch_size']}, Num samples approx {len(val_dataset)}")
    else:
        print("No validation annotations provided, validation_loader will be None.")


    # 9. Training Loop (Replaces Keras `model.fit()`)
    #    This is where the actual training happens.
    #    The following is a SKELETON and needs to be filled based on your specific model outputs and loss needs.
    print("\n--- Starting PyTorch Training Loop (Skeleton) ---")
    num_epochs = model_params.get('epochs', 1) # From your Keras code, it was 1 for testing

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        # --- Training Phase ---
        model.train()
        epoch_train_loss = 0.0
        # Example for a model that outputs [emb, clf] and uses triplet + clf loss
        # Adjust if your model output or loss structure is different
        # for batch_idx, (anchor_data, positive_data, negative_data, clf_targets) in enumerate(train_loader):
        #     anchor_data = anchor_data.to(device, non_blocking=True)
        #     positive_data = positive_data.to(device, non_blocking=True)
        #     negative_data = negative_data.to(device, non_blocking=True)
        #     clf_targets = clf_targets.to(device, non_blocking=True) # Ensure this is integer class indices

        #     optimizer.zero_grad()

        #     # Forward pass
        #     # Assuming your model's forward takes one input at a time
        #     # And returns [embedding_output, classification_output]
        #     total_loss_batch = 0.0
        #     emb_anchor, clf_out_anchor = model(anchor_data)

        #     if 'triplet' in active_losses:
        #         emb_positive, _ = model(positive_data)
        #         emb_negative, _ = model(negative_data)
        #         loss_t = active_losses['triplet'](emb_anchor, emb_positive, emb_negative)
        #         total_loss_batch += loss_weights_pytorch['triplet'] * loss_t
        #         # tb_writer.add_scalar('LossBatch/triplet_train', loss_t.item(), epoch * len(train_loader) + batch_idx)


        #     if 'classification' in active_losses:
        #         loss_c = active_losses['classification'](clf_out_anchor, clf_targets)
        #         total_loss_batch += loss_weights_pytorch['classification'] * loss_c
        #         # tb_writer.add_scalar('LossBatch/clf_train', loss_c.item(), epoch * len(train_loader) + batch_idx)

        #     if total_loss_batch == 0.0:
        #         print(f"Warning: Batch {batch_idx} had zero loss. Check loss configuration and model outputs.")
        #         continue


        #     total_loss_batch.backward()
        #     torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Gradient Clipping
        #     optimizer.step()

        #     epoch_train_loss += total_loss_batch.item()

        #     if train_verbose > 0 and batch_idx % model_params.get('log_interval', 10) == 0: # Add log_interval to model_params
        #         print(f"  Train Batch: {batch_idx+1}/{len(train_loader)} Loss: {total_loss_batch.item():.4f}")
        
        # avg_epoch_train_loss = epoch_train_loss / len(train_loader) if len(train_loader) > 0 else 0.0
        # tb_writer.add_scalar('LossEpoch/train', avg_epoch_train_loss, epoch)
        # current_lr = optimizer.param_groups[0]['lr']
        # tb_writer.add_scalar('LearningRate', current_lr, epoch)
        # print(f"Epoch {epoch+1} Train Summary: Avg Loss: {avg_epoch_train_loss:.4f}, LR: {current_lr}")

        # --- Validation Phase ---
        # if val_loader:
        #     model.eval()
        #     epoch_val_loss = 0.0
        #     all_clf_targets_val = []
        #     all_clf_predictions_val = [] # Store predicted indices

        #     with torch.no_grad():
        #         for batch_idx_val, (anchor_data_v, positive_data_v, negative_data_v, clf_targets_v) in enumerate(val_loader):
        #             # ... (similar to training forward pass, calculate loss, collect predictions for metrics)
        #             # ...
        #             # Example:
        #             # emb_anchor_v, clf_out_anchor_v = model(anchor_data_v.to(device))
        #             # if 'classification' in active_losses:
        #             #     loss_c_v = active_losses['classification'](clf_out_anchor_v, clf_targets_v.to(device))
        #             #     epoch_val_loss += loss_weights_pytorch['classification'] * loss_c_v.item()
        #             #     _, predicted_indices = torch.max(clf_out_anchor_v, 1)
        #             #     all_clf_predictions_val.extend(predicted_indices.cpu().numpy())
        #             #     all_clf_targets_val.extend(clf_targets_v.cpu().numpy())
        #             # ... (add triplet loss if applicable to val loss calculation) ...

        #     avg_epoch_val_loss = epoch_val_loss / len(val_loader) if val_loader and len(val_loader) > 0 else 0.0
        #     tb_writer.add_scalar('LossEpoch/val', avg_epoch_val_loss, epoch)
            
        #     val_accuracy = 0.0
        #     if 'classification' in active_losses and len(all_clf_targets_val) > 0:
        #         # Calculate accuracy
        #         correct_val = sum(p == t for p, t in zip(all_clf_predictions_val, all_clf_targets_val))
        #         val_accuracy = correct_val / len(all_clf_targets_val)
        #         tb_writer.add_scalar('Accuracy/val', val_accuracy, epoch)
        #     print(f"Epoch {epoch+1} Val Summary: Avg Loss: {avg_epoch_val_loss:.4f}, Accuracy: {val_accuracy:.4f}")

            # --- "Callbacks" logic for this epoch ---
            # Metric to monitor (choose from your calculated val metrics)
            # current_monitor_val = avg_epoch_val_loss if 'loss' in monitor_metric_name else val_accuracy # Example
            
            # ModelCheckpoint
            # if (monitor_mode == 'max' and current_monitor_val > best_monitor_metric_val) or \
            #    (monitor_mode == 'min' and current_monitor_val < best_monitor_metric_val):
            #     best_monitor_metric_val = current_monitor_val
            #     # Adapt Keras filename format for PyTorch
            #     # 'ep{epoch:03d}-loss{loss:.5f}-' + monitor + '{' + monitor + ':.5f}.ckpt'
            #     # The 'loss' here was likely training loss from Keras ModelCheckpoint.
            #     # Let's use avg_epoch_train_loss for consistency if needed.
            #     checkpoint_filename = f"ep{epoch+1:03d}-trainloss{avg_epoch_train_loss:.5f}-{monitor_metric_name.replace('val_','')}{current_monitor_val:.5f}.pt"
            #     torch.save(model.state_dict(), os.path.join(weights_save_path, checkpoint_filename))
            #     print(f"  Saved checkpoint: {checkpoint_filename} (Monitored: {current_monitor_val:.5f})")
            #     early_stopping_counter = 0 # Reset for early stopping
            # else:
            #     early_stopping_counter += 1

            # ReduceLROnPlateau
            # lr_scheduler.step(current_monitor_val)

            # EarlyStopping
            # if early_stopping_counter >= early_stopping_patience:
            #     print(f"  Early stopping triggered after {early_stopping_patience} epochs without improvement.")
            #     break # Break from epoch loop
        # else: # No val_loader
        #     # If no validation, you might save model every N epochs or just at the end
        #     # And ReduceLROnPlateau won't step, EarlyStopping won't trigger based on val metrics.
        #     # Optionally, scheduler can step on training loss: lr_scheduler.step(avg_epoch_train_loss)
        #     pass
    
    # tb_writer.close() # Close TensorBoard writer after training
    print("--- Training Loop Skeleton Finished ---")


    # 10. Post-training actions
    # Model summary again if desired
    summary(model, input_size=summary_input_shape, col_names=["input_size", "output_size", "num_params"])
    summary(model, input_size=summary_input_shape)

    # `remove_suboptimal_weights` adaptation:
    # This function would need to:
    # 1. List all *.pt files in `weights_save_path`.
    # 2. Parse filenames to extract the monitored metric value.
    # 3. Keep only the best one (or top N) and delete others.
    # This requires careful implementation to avoid deleting desired weights.
    # def remove_suboptimal_weights_pytorch(path_to_weights, monitor_name, higher_is_better):
    #     # Placeholder logic
    #     all_checkpoints = glob.glob(os.path.join(path_to_weights, "*.pt"))
    #     if not all_checkpoints: return
    #     # Parse filenames, sort, find best, delete others
    #     print(f"Placeholder: Logic for removing suboptimal weights in {path_to_weights} based on {monitor_name}")
    # remove_suboptimal_weights_pytorch(weights_save_path, monitor_metric_name, monitor_mode == 'max')

    
    # Build model
    model.build((None, None, model_params['num_feats']))
    
    # Initialize inputs and outputs
    dummy_inpt = (np.random.rand(model_params['batch_size'], max(abs(model_params['max_seq_len']), 123), model_params['num_feats']))
    print(' * dummy_shape:', dummy_inpt.shape)
    dummy_pred = model(dummy_inpt);
    print(' * dummy_pred shape', [ p.shape for p in dummy_pred ])
    dummy_pred = model.predict(dummy_inpt);
    print(' * dummy_pred predict shape', [ p.shape for p in dummy_pred ])
    dummy_emb = model.get_embedding(dummy_inpt);
    print(' * dummy_emb shape', dummy_emb.shape)
    
    
    optimizer = Adam(model_params['init_lr'], clipnorm=1.)
    losses, metrics, loss_weights, sample_weights_mode = {}, {}, {}, {}
    losses['output_1'] = tf.keras.losses.CategoricalCrossentropy()
    loss_weights['output_1'] = 0.4
    # loss_weights = None
    # loss_weights = [ 1.0 ]
    metrics = [ 'accuracy', get_lr_metric(optimizer) ]
        

    print(' * losses:', losses)
    print(' * loss_weights:', loss_weights)
    if sample_weights_mode == {}: sample_weights_mode = None
    print(' * sample_weights_mode:', sample_weights_mode)

    model.summary(100)

    
    monitor = model_params.get('monitor', 'val_loss')
    print(' * Monitor:', monitor)
    model_chkpt_path = 'ep{epoch:03d}-loss{loss:.5f}-' + monitor + '{' + monitor + ':.5f}.ckpt'
    callbacks = [ 
                    TensorBoard(log_dir = model_params['path_model'], profile_batch=0),
                    ModelCheckpoint(model_params['path_model'] + 'weights/' + model_chkpt_path,
                                             monitor=monitor, save_weights_only=True, 
                                             save_best_only=True, save_freq='epoch'),
                    ReduceLROnPlateau(monitor=monitor, min_delta=0.001, factor=0.1, patience=3, verbose=1, min_lr=1e-7),
                    EarlyStopping(monitor=monitor, min_delta=0.001, patience=6, verbose=1),
                ]



    file_writer = tf.summary.create_file_writer(model_params['path_model'] + "/metrics")
    file_writer.set_as_default()
    
    print(' * metrics:', metrics)
    print(' * sample_weights_mode:', sample_weights_mode)
    
    model.compile(optimizer=optimizer,
                  loss = losses,
                  metrics = metrics,
                  loss_weights = loss_weights,
                  sample_weight_mode=sample_weights_mode
                  )
    
    # Save model
    model.save(model_params['path_model'] + 'model')
    
    train_gen = triplet_data_generator(pose_annotations_file=model_params['train_annotations'], 
                           validation=False, 
                           in_memory_generator=model_params['in_memory_generator_train'],
                           **model_params)
    if model_params['val_annotations'] == '': val_gen = None
    else:
        val_gen = triplet_data_generator(pose_annotations_file=model_params['val_annotations'], 
                           validation=True, 
                           in_memory_generator=model_params['in_memory_generator_val'],
                           **model_params)

    print(train_gen, val_gen)
    
    model.fit(
            train_gen,
            validation_data = val_gen,
            steps_per_epoch = num_train_files//model_params['batch_size'],
            validation_steps = None if num_val_files == 0 else num_val_files//model_params['batch_size'],
            # epochs = 300, 
            epochs = 1,
            # steps_per_epoch = 10,         # num_val_files//model_params['batch_size'],
            # validation_steps = 10,
            # epochs = 50, 
            verbose = train_verbose,
            #callbacks = callbacks,
        )

    del train_gen; del val_gen
    #del callbacks

    model.summary(100)
    
    # Remove suboptimal weights
    remove_path_weights(model_params['path_model'], model_params['monitor'], model_params['min_monitor'])


if __name__ == "__main__":

    print(torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU found")

    model_params = {
        "path_results": "./pretrained_models/",

        # # NTU-120 Data sets to optimize the therapy data
        # "train_annotations": "./ntu_annotations/one_shot_aux_set_train_full8.txt",
        # "val_annotations": "./ntu_annotations/one_shot_aux_set_val_full8.txt",
        "eval_therapies": True,       ### Therapy data needed for its evaluation
        # "eval_therapies_triplets_dataset": "./therapies_annotations/triplets/triplets_dataset.pckl",
        # "eval_therapies_triplets_bgnd_dataset": "./therapies_annotations/triplets/triplets_ther_pat_bgnd_dataset.pckl",
        # "eval_therapies_video_skels": "./therapies_annotations/video_skels.pckl",
        # "h_flip": True,
        # "skip_frames": [2, 3],

        # NTU-120 Data sets to optimize the NTU one-shot benchmark
        "train_annotations": "./ntu_annotations/one_shot_aux_set.txt",
        "val_annotations": "",
        # "eval_therapies": False,
        "h_flip": False,
        "monitor": "ntu_one_shot_acc_euc",
        "min_monitor": False,
        "skip_frames": [2],

        "in_memory_generator_train": False,
        "in_memory_generator_val": True,
        "in_memory_callback": True,

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
        "classification": True, "triplet": False, "decoder": False, "reverse_decoder": False,
        "num_classes": 120,
        "clf_neurons": 0,

        "model_name": "train_TCN",
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

    # Placeholder for train_utils if not properly imported/defined
    if 'train_utils' not in globals() or not hasattr(train_utils, 'create_model_folder'):
        class train_utils_placeholder:
            @staticmethod
            def create_model_folder(path_results, model_name_base):
                from datetime import datetime
                ts = datetime.now().strftime("%m%d_%H%M%S") # Added seconds for more uniqueness
                model_name_folder = f"{str(model_name_base)}_{ts}_run{np.random.randint(1000):03d}"
                path_model = os.path.join(path_results, model_name_folder)
                # os.makedirs(path_model, exist_ok=True) # path_model itself is created later
                return path_model
        train_utils = train_utils_placeholder

    # Placeholder for get_num_feats if not properly imported/defined
    if 'get_num_feats' not in globals():
        def get_num_feats_placeholder(**params):
            num_f = 0
            num_j = params.get('joints_num', 25)
            dim_j = params.get('joints_dim', 3)
            if params.get('use_coords', False): num_f += num_j * dim_j
            if params.get('use_jcd_features', False):
                try: num_f += int(comb(num_j, 2))
                except: num_f += 300 # Fallback
            if params.get('use_bone_angles', False) :
                # This is a very rough estimate for bone angles, actual calculation is complex
                num_f += (num_j - 5) * dim_j if num_j > 5 else num_j * dim_j
            if num_f == 0: # Default to a reasonable number if no features selected
                print("Warning: get_num_feats_placeholder resulted in 0 features. Defaulting to 75 (coords only).")
                num_f = num_j * dim_j
            print(f"Placeholder get_num_feats calculated: {num_f}")
            return num_f
        get_num_feats = get_num_feats_placeholder
        
    main(model_params)
