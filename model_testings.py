import os

base_dir = "train_models"

for folder in os.listdir(base_dir):
    parts = folder.split('_')
    model_type = parts[0]
    
    cmd = f"python main.py --mode test --model {model_type} --load_path train_models/{folder}/best_model"

    print(f"Folder name: {folder}")
    print(f"Command: {cmd}\n")
