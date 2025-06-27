import os
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


class MetricsLogger(tf.keras.callbacks.Callback):
    def __init__(self, validation_data, metrics_save_dir):
        super().__init__()
        self.validation_data = validation_data
        self.metrics_save_dir = metrics_save_dir
        self.train_losses = []
        self.val_losses = []
        self.val_f1_scores = []
        self.val_auc_scores = []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        
        train_loss = logs.get("loss")
        self.train_losses.append(train_loss)

        val_loss = logs.get("val_loss")
        self.val_losses.append(val_loss)

        val_data = self.validation_data
        if val_data is not None:
            X_val, y_val, *_ = next(iter(val_data))
            y_pred = self.model.predict(X_val)
            y_pred_classes = np.argmax(y_pred, axis=1)
            y_true_classes = np.argmax(y_val, axis=1)
            
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
            png_path = os.path.join(confusion_matrix_dir, f'conf_matrix_epoch_{epoch+1:03d}.png')
            plt.savefig(png_path)
            plt.close()

            print(f"[MetricsLogger] Saved confusion matrix (.npy and .png) to {confusion_matrix_dir}")

