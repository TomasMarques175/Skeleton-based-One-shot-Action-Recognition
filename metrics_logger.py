import os
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from data_generator import triplet_data_generator_deterministic

class MetricsLogger(tf.keras.callbacks.Callback):
    def __init__(self, validation_steps, pose_annotations_file, metrics_save_dir, 
                in_memory_generator, model_params, validation_generator=None):
        super().__init__()
        self.validation_file = pose_annotations_file
        self.in_memory_generator = in_memory_generator
        self.model_params = model_params
        self.validation_steps = validation_steps
        self.metrics_save_dir = metrics_save_dir
        self.train_losses = []
        self.val_losses = []
        self.val_f1_scores = []
        self.val_auc_scores = []
        self.validation_generator  = validation_generator

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        
        train_loss = logs.get("loss")
        self.train_losses.append(train_loss)

        val_loss = logs.get("val_loss")
        self.val_losses.append(val_loss)

        val_gen_for_metrics = triplet_data_generator_deterministic(
            pose_annotations_file=self.validation_file,
            validation=True,
            in_memory_generator=self.in_memory_generator,
            **self.model_params, 
            repeat=False
        )

        val_data = triplet_data_generator_deterministic(
            pose_annotations_file=self.validation_file,
            validation=True,
            in_memory_generator=self.in_memory_generator,
            **self.model_params, 
            repeat=False
        )

        if val_data is not None:
            print(f"[MetricsLogger] Validation data generator created for {self.validation_steps} steps.")

            # get predictions for the entire validation dataset
            y_pred = self.model.predict(val_gen_for_metrics, steps=None)

            print(f"[MetricsLogger] Raw y_pred type: {type(y_pred)}")
            if isinstance(y_pred, list):
                print(f"[MetricsLogger] y_pred is a list of length {len(y_pred)}")
                for idx, pred in enumerate(y_pred):
                    print(f"   y_pred[{idx}] shape: {pred.shape}")
                y_pred = y_pred[0]
                print(f"[MetricsLogger] Using y_pred[0] with shape: {y_pred.shape}")
            else:
                print(f"[MetricsLogger] y_pred array shape: {y_pred.shape}")

            # collect true labels
            y_true_list = []
            num_batches = 0
            for batch in val_data:
                _, y_batch, *_ = batch
                y_true_list.append(y_batch)
                num_batches += 1
            y_true = np.concatenate(y_true_list, axis=0)

            print(f"[MetricsLogger] Collected y_true from {num_batches} batches with shape: {y_true.shape}")

            # convert to classes
            y_pred_classes = np.argmax(y_pred, axis=1)
            y_true_classes = np.argmax(y_true, axis=1)

            assert y_pred_classes.shape[0] == y_true_classes.shape[0], "Mismatch in number of samples"

            # show debug samples
            print(f"[MetricsLogger] y_pred_classes shape: {y_pred_classes.shape}, "
                f"sample: {y_pred_classes[:10]}")
            print(f"[MetricsLogger] y_true_classes shape: {y_true_classes.shape}, "
                f"sample: {y_true_classes[:10]}")
            
            f1 = f1_score(y_true_classes, y_pred_classes, average="macro")
            self.val_f1_scores.append(f1)
            
            try:
                auc = roc_auc_score(y_true_classes, y_pred, multi_class="ovo")
            except Exception as e:
                print(f"AUC error: {e}")
                auc = np.nan
            self.val_auc_scores.append(auc)

        np.savez(
            os.path.join(self.metrics_save_dir, "tensorflow_train_loss_val_loss_val_f1_val_auc.npz"),
            train_losses=np.array(self.train_losses),
            val_losses=np.array(self.val_losses),
            val_f1_scores=np.array(self.val_f1_scores),
            val_auc_scores=np.array(self.val_auc_scores),
        )
        print(f"[MetricsLogger] Saved metrics to {self.metrics_save_dir}")
    
        if epoch == 0 or (epoch + 1) % 10 == 0:
            # Get directory one level up + 'Conversion comparison'
            current_dir = os.path.dirname(os.path.abspath(__file__))
            parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
            confusion_matrix_dir = os.path.join(parent_dir, 'Conversion comparison')
            os.makedirs(confusion_matrix_dir, exist_ok=True)

            # Compute confusion matrix
            conf_mat = confusion_matrix(y_true_classes, y_pred_classes)

            # Save raw confusion matrix as .npy file
            npy_path = os.path.join(confusion_matrix_dir, f'tensorflow_conf_matrix_epoch_{epoch+1:03d}.npy')
            np.save(npy_path, conf_mat)

            # Save visualisation as PNG file
            plt.figure(figsize=(40, 30))
            ax = sns.heatmap(
                conf_mat,
                annot=True,
                fmt='d',
                cmap='Blues',
                annot_kws={"size": 5},  # small font size for readability
                cbar=True
            )
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.title(f"Confusion Matrix - Epoch {epoch+1}")
            plt.xticks(rotation=90, fontsize=5)
            plt.yticks(fontsize=5)
            plt.tight_layout()
            png_path = os.path.join(confusion_matrix_dir, f'tensorflow_conf_matrix_epoch_{epoch+1:03d}.png')
            plt.savefig(png_path)
            plt.close()

            print(f"[MetricsLogger] Saved confusion matrix (.npy and .png) to {confusion_matrix_dir}")
        print(f"[MetricsLogger] Epoch {epoch+1} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val F1: {f1:.4f}, Val AUC: {auc:.4f}")

