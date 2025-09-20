import os
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from data_generator import triplet_data_generator


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

    def read_annotations(self):
        print('Reading annotations from:', self.validation_file)
        with open(self.validation_file, 'r') as f:
            return [(filename, int(label)) for filename, label in (line.strip().split() for line in f)]
    
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
            **self.model_params
        )

        val_data = triplet_data_generator_deterministic(
            pose_annotations_file=self.validation_file,
            validation=True,
            in_memory_generator=self.in_memory_generator,
            **self.model_params
        )

        if val_data is not None:
            print(f"[MetricsLogger] Validation data generator created for {self.validation_steps} steps.")

            # get predictions for the entire validation dataset
            y_pred = self.model.predict(val_gen_for_metrics, steps=self.validation_steps)

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
            pose_list = self.read_annotations()
            labels_dict = {i: i-1 for i in range(1, 121)}  # same mapping as you printed
            y_true = np.array([labels_dict[label] for (_, label) in pose_list])

            # convert to classes
            y_pred_classes = np.argmax(y_pred, axis=1)
            y_true_classes = y_true

            if y_pred_classes.shape[0] != y_true_classes.shape[0]:
                print(f"Warning: mismatched shapes - y_pred: {y_pred_classes.shape[0]}, y_true: {y_true_classes.shape[0]}")
                min_len = min(y_pred_classes.shape[0], y_true_classes.shape[0])
                y_pred_classes = y_pred_classes[:min_len]
                y_true_classes = y_true_classes[:min_len]

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

""" class MetricsLogger(tf.keras.callbacks.Callback):
    def __init__(self, validation_steps, val_data, metrics_save_dir, 
                 in_memory_generator, model_params, validation_generator=None):
        super().__init__()
        self.in_memory_generator = in_memory_generator
        self.model_params = model_params
        self.validation_steps = validation_steps
        self.metrics_save_dir = metrics_save_dir
        self.val_data = val_data
        self.train_losses = []
        self.val_losses = []
        self.val_f1_scores = []
        self.val_auc_scores = []
        self.validation_generator = validation_generator

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        train_loss = logs.get("loss")
        val_loss = logs.get("val_loss")
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)

        if self.validation_generator is None:
            print("[MetricsLogger] No validation generator provided.")
            return

        print(f"[MetricsLogger] Running validation predictions for {self.validation_steps} steps...")
        y_pred = self.model.predict(self.validation_generator, steps=self.validation_steps)

        if isinstance(y_pred, list):
            print(f"[MetricsLogger] y_pred is list, using y_pred[0] with shape: {y_pred[0].shape}")
            y_pred = y_pred[0]
            print(f"[MetricsLogger] y_pred shape: {y_pred.shape}")

        # Get ground truth from val_data
        labels = self.val_data['action'].values
        # Use the global class list from training
        class_names = sorted(self.val_data['action'].unique())  # Only valid if same as training!
        label_to_idx = {label: idx for idx, label in enumerate(class_names)}
        y_true = np.array([label_to_idx[label] for label in self.val_data['action'].values])
        y_pred_classes = np.argmax(y_pred, axis=1)

        print(f"[MetricsLogger] y_pred shape: {y_pred.shape}, y_true shape: {y_true.shape}")
        print(f"[MetricsLogger] Labels shape: {labels.shape}, sample: {labels[:5]}")
        # print(f"[MetricsLogger] Label to index mapping: {label_to_idx}")
        print(f"[MetricsLogger] y_true shape: {y_true.shape}, sample: {y_true[:5]}")
        print(f"[MetricsLogger] y_pred_classes shape: {y_pred_classes.shape}, sample: {y_pred_classes[:5]}")

        # print("Label to index mapping:", label_to_idx)
        # print("Sample true:", y_true[:5])
        # print("Sample pred:", y_pred_classes[:5])

        # Ensure lengths match
        if len(y_pred_classes) != len(y_true):
            min_len = min(len(y_pred_classes), len(y_true))
            print(f"⚠️ Length mismatch: truncating to {min_len}")
            y_pred_classes = y_pred_classes[:min_len]
            y_true = y_true[:min_len]

        f1 = f1_score(y_true, y_pred_classes, average="macro")
        self.val_f1_scores.append(f1)

        try:
            auc = roc_auc_score(y_true, y_pred, multi_class="ovo")
        except Exception as e:
            print(f"[MetricsLogger] AUC error: {e}")
            auc = np.nan
        self.val_auc_scores.append(auc)

        # Save metrics
        os.makedirs(self.metrics_save_dir, exist_ok=True)
        np.savez(
            os.path.join(self.metrics_save_dir, "tensorflow_train_loss_val_loss_val_f1_val_auc.npz"),
            train_losses=np.array(self.train_losses),
            val_losses=np.array(self.val_losses),
            val_f1_scores=np.array(self.val_f1_scores),
            val_auc_scores=np.array(self.val_auc_scores),
        )

        print(f"[MetricsLogger] Epoch {epoch+1} - Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val F1: {f1:.4f}, Val AUC: {auc:.4f}")

        # Save confusion matrix every 10 epochs
        if epoch == 0 or (epoch + 1) % 10 == 0:
            conf_mat = confusion_matrix(y_true, y_pred_classes)

            current_dir = os.path.dirname(os.path.abspath(__file__))
            parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
            cm_dir = os.path.join(parent_dir, 'Conversion comparison')
            os.makedirs(cm_dir, exist_ok=True)

            np.save(os.path.join(cm_dir, f'conf_matrix_epoch_{epoch+1:03d}.npy'), conf_mat)

            plt.figure(figsize=(40, 30))
            sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues',
                        annot_kws={"size": 5}, cbar=True)
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.title(f"Confusion Matrix - Epoch {epoch+1}")
            plt.xticks(rotation=90, fontsize=5)
            plt.yticks(fontsize=5)
            plt.tight_layout()
            plt.savefig(os.path.join(cm_dir, f'conf_matrix_epoch_{epoch+1:03d}.png'))
            plt.close()

            print(f"[MetricsLogger] Confusion matrix saved to {cm_dir}")
 """