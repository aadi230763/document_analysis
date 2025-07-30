from huggingface_hub import snapshot_download
import shutil
import os

# This will download and extract the actual model files
cache_path = snapshot_download("BAAI/bge-small-en-v1.5")

# Destination where we want to place the model
dest = os.path.join("models", "bge-small-en-v1.5")

# Copy the actual model from cache to your backend/models folder
if os.path.exists(dest):
    shutil.rmtree(dest)

shutil.copytree(cache_path, dest)

print(f"✅ Model downloaded to: {dest}")
