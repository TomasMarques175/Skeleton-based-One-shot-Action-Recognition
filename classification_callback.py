from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import tensorflow as tf
import os
import pandas as pd

class ClassificationMetricsCallback(tf.keras.callbacks.Callback):
    def __init__(self, val_gen, steps, output_dir=".", prefix="val"):
        super(ClassificationMetricsCallback, self).__init__()
        self.val_gen = val_gen
        self.steps = steps
        self.output_dir = output_dir
        self.prefix = prefix

    def on_epoch_end(self, epoch, logs=None):
        y_preds, y_trues = [], []

        for _ in range(self.steps):
            X_val, y_val, _ = next(self.val_gen)
            y_pred_probs = self.model.predict(X_val, verbose=0)[0]
            y_pred = np.argmax(y_pred_probs, axis=1)
            y_true = np.argmax(y_val, axis=1)
            y_preds.extend(y_pred)
            y_trues.extend(y_true)

        report = classification_report(y_trues, y_preds, digits=4)
        print("\n{} Classification Report (Epoch {}):\n{}".format(self.prefix.capitalize(), epoch + 1, report))

        cm = confusion_matrix(y_trues, y_preds)
        cm_df = pd.DataFrame(cm)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        cm_df.to_csv(os.path.join(self.output_dir, "{}_conf_matrix_epoch{}.csv".format(self.prefix, epoch + 1)), index=False)
