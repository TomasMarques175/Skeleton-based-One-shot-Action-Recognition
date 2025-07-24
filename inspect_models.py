import tensorflow as tf
import torch

# ------------------------------
# 1. Load your PyTorch model class
# adjust the import below
from models.TCN_classifier import TCN_clf

# instantiate the PyTorch model
pytorch_model = TCN_clf()
pytorch_model.eval()

# ------------------------------
# 2. Load the TensorFlow SavedModel
tf_model = tf.saved_model.load(
    r".\ntu_benchmark_model"
)

# ------------------------------
# 3. Print TensorFlow variables
print("\n=== TensorFlow variables ===")
for var in tf_model.variables:
    print(f"{var.name} {var.shape}")

# ------------------------------
# 4. Print PyTorch state dict
print("\n=== PyTorch state_dict keys ===")
for k, v in pytorch_model.state_dict().items():
    print(f"{k} {v.shape}")

print("\n✅ Inspection done. Please copy these results so we can map them together.")
