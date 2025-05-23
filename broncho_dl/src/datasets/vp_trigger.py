#!/usr/bin/env python3
import os
import json
import random
import copy
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

#########################################
# 1. Create Split Indices File          #
#########################################
def create_split_indices(raw_pose_dir, window_size=10, val_ratio=0.2, seed=42, save_path="split_indices.json"):
    random.seed(seed)
    split_indices = {"train": {}, "val": {}}
    traj_list = sorted(os.listdir(raw_pose_dir))
    for traj in traj_list:
        traj_pose_dir = os.path.join(raw_pose_dir, traj)
        if not os.path.isdir(traj_pose_dir):
            continue
        pose_files = sorted([f for f in os.listdir(traj_pose_dir) if f.endswith('.txt')])
        N = len(pose_files)
        if N < 2:
            continue
        valid_indices = list(range(N - 1))
        random.shuffle(valid_indices)
        split_point = int(len(valid_indices) * (1 - val_ratio))
        train_indices = sorted(valid_indices[:split_point])
        val_indices = sorted(valid_indices[split_point:])
        split_indices["train"][traj] = train_indices
        split_indices["val"][traj] = val_indices

    with open(save_path, "w") as f:
        json.dump(split_indices, f, indent=4)
    print(f"Saved split indices to {save_path}")
    return split_indices

###############################################
# 2. Normalizer Class (Consolidated)          #
###############################################
class Normalizer:
    def __init__(self, txt_file=None, raw_image_dir=None, split_indices=None, norm_config_path="normalizer_config.json"):
        self.txt_file = txt_file
        self.raw_image_dir = raw_image_dir
        self.split_indices = split_indices
        if os.path.exists(norm_config_path):
            with open(norm_config_path, "r") as f:
                self.norm_state = json.load(f)
            print(f"Loaded normalization config from {norm_config_path}")
        else:
            self.norm_state = self.compute_overall_norm_state()
            self.save_json(norm_config_path)
    
    def compute_overall_norm_state(self):
        norm_state = {}
        if self.txt_file and self.split_indices:
            norm_state["params"] = self.compute_params_norm_config(self.txt_file, self.split_indices["train"])
        if self.raw_image_dir:
            # Compute image norm config only over training samples:
            norm_state["image"] = self.compute_image_norm_state(self.raw_image_dir)
        return norm_state
    
    def compute_params_norm_config(self, txt_file, train_indices):
        params = {
            "Bend": [], "Rot": [], "Trans": [],
            "Bend_vel": [], "Rot_vel": [], "Trans_vel": []
        }
        with open(txt_file, "r") as f:
            lines = f.readlines()
        for idx in train_indices:
            if idx >= len(lines):
                continue
            line = lines[idx].strip().replace("\t", ", ").split(", ")
            data = {item.split(": ")[0]: item.split(": ")[1] for item in line if ": " in item}
            for key in params:
                try:
                    value = float(data.get(key, "0.0"))
                    params[key].append(value)
                except ValueError:
                    params[key].append(0.0)
        stats = {}
        for key, values in params.items():
            if not values:
                stats[key] = {"mean": 0.0, "std": 1.0, "min": 0.0, "max": 0.0}
                continue
            tensor = torch.tensor(values, dtype=torch.float32)
            stats[key] = {
                "mean": tensor.mean().item(),
                "std": tensor.std().item() if tensor.std() != 0 else 1.0,
                "min": tensor.min().item(),
                "max": tensor.max().item()
            }
        return stats
    
    def compute_image_norm_state(self, image_dir):
        # Use only training samples from txt_file to compute image normalization stats.
        basic_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        all_images = []
        if self.txt_file is None or self.split_indices is None or "train" not in self.split_indices:
            raise ValueError("txt_file and training split indices must be provided for image normalization computation")
        with open(self.txt_file, "r") as f:
            lines = f.readlines()
        for idx in self.split_indices["train"]:
            if idx >= len(lines):
                continue
            line = lines[idx].strip().replace("\t", ", ")
            # Parse line to extract the image file info from "Idx" field
            parts = line.split(", ")
            id_str = None
            for part in parts:
                if ": " in part:
                    key, val = part.split(": ", 1)
                    if key == "Idx":
                        id_str = val
                        break
            if not id_str:
                continue
            id_num_str = id_str.split(".png")[0]
            try:
                id_int = int(id_num_str)
            except ValueError:
                continue
            img_path = os.path.join(image_dir, f"{id_int-1}.png")
            if os.path.exists(img_path):
                try:
                    img = Image.open(img_path).convert("RGB")
                    img_tensor = basic_transform(img)
                    all_images.append(img_tensor)
                except Exception as e:
                    print(f"Skipping image {img_path} due to error: {e}")
                    continue
        if not all_images:
            raise ValueError("No valid training images found for normalization")
        images_cat = torch.stack(all_images, dim=0)
        mean = images_cat.mean(dim=[0, 2, 3])
        std = images_cat.std(dim=[0, 2, 3])
        min_val = images_cat.amin(dim=[0, 2, 3])
        max_val = images_cat.amax(dim=[0, 2, 3])
        return {
            "mean": mean.tolist(),
            "std": std.tolist(),
            "min": min_val.tolist(),
            "max": max_val.tolist()
        }
    
    def save_json(self, save_json_path):
        with open(save_json_path, "w") as f:
            json.dump(self.norm_state, f, indent=4)
        print(f"Saved normalization config to {save_json_path}")
    
    def normalize(self, x, var_name):
        if var_name in ["image", "rgb", "rgb_raw", "rgb_nav"]:
            stats = self.norm_state["image"]
            mean = torch.tensor(stats["mean"], dtype=x.dtype, device=x.device).view(-1, 1, 1)
            std = torch.tensor(stats["std"], dtype=x.dtype, device=x.device).view(-1, 1, 1)
            return (x - mean) / std
        elif var_name in ["Bend", "Rot", "Trans", "Bend_vel", "Rot_vel", "Trans_vel"]:
            stats = self.norm_state["params"][var_name]
            mean = torch.tensor(stats["mean"], dtype=x.dtype, device=x.device)
            std = torch.tensor(stats["std"], dtype=x.dtype, device=x.device)
            return (x - mean) / std
        else:
            raise ValueError(f"Unknown var_name {var_name} in normalization")
    
    def denormalize(self, x, var_name):
        if var_name in ["image", "rgb", "rgb_raw", "rgb_nav"]:
            stats = self.norm_state["image"]
            mean = torch.tensor(stats["mean"], dtype=x.dtype, device=x.device).view(-1, 1, 1)
            std = torch.tensor(stats["std"], dtype=x.dtype, device=x.device).view(-1, 1, 1)
            return x * std + mean
        elif var_name in ["Bend", "Rot", "Trans", "Bend_vel", "Rot_vel", "Trans_vel"]:
            stats = self.norm_state["params"][var_name]
            mean = torch.tensor(stats["mean"], dtype=x.dtype, device=x.device)
            std = torch.tensor(stats["std"], dtype=x.dtype, device=x.device)
            return x * std + mean
        else:
            raise ValueError(f"Unknown var_name {var_name} in denormalization")

###############################################
# 3. Dataset Class                            #
###############################################
class CombinedScoresDataset(Dataset):
    def __init__(self, txt_file, raw_dir, nav_dir, window_size=5, transform=None, nav_transform=None, normalizer=None):
        self.txt_file = txt_file
        self.raw_dir = raw_dir
        self.nav_dir = nav_dir
        self.window_size = window_size
        self.transform = transform if transform else transforms.ToTensor()
        self.nav_transform = nav_transform if nav_transform else transforms.ToTensor()
        self.normalizer = normalizer
        self.area_mapping = {
            'None':0, 'carina': 34, 'RMB': 1, 'RUL': 2, 'RB1': 3, 'back': 4, 'Unknown': 4,
            'RB2': 5, 'RB3': 6, 'BI': 7, 'RML': 8, 'RB4': 9, 'RB5': 10, 'RLL': 11, 
            'RB6': 12, 'RB7-10': 13, 'RB7': 14, 'RB8-10': 15, 'RB8': 16, 'RB9': 17,
            'RB10': 18, 'LBM': 19, 'LUL': 20, 'UBD': 21, 'LB1+2': 22, 'LB3': 23,
            'LI': 24, 'LB4': 25, 'LB5': 26, 'LLL': 27, 'LB6': 28, 'LB8-10': 29,
            'LB8': 30, 'LB9-10': 31, 'LB9': 32, 'LB10': 33, 
        }
        with open(txt_file, "r") as f:
            lines = f.readlines()
        self.samples = []
        for i, line in enumerate(lines):
            line = line.strip().replace("\t", ", ")
            if not line:
                continue
            parts = line.split(", ")
            data = {}
            for part in parts:
                if ": " in part:
                    key, val = part.split(": ", 1)
                    data[key] = val
                    if val == None or val == "None":
                        data[key] = "None"
            idx_str = data.get("Idx", None)
            if not idx_str:
                continue
            id_num_str = idx_str.split(".png")[0]
            try:
                id_int = int(id_num_str)
            except ValueError:
                continue
            
            if self.area_mapping.get(data.get("Target Area")) == None:
                print(data.get("Target Area"))

            self.samples.append({
                "idx": i,
                "file": idx_str,
                "id_int": id_int,
                "current_area": self.area_mapping.get(data.get("Current Area")),
                "target_area": self.area_mapping.get(data.get("Target Area")),
                "bend": float(data.get("Bend", "0")),
                "rot": float(data.get("Rot", "0")),
                "trans": float(data.get("Trans", "0")),
                "trigger": float(data.get("Score", "0")),
                "bend_vel": float(data.get("Bend_vel", "0")),
                "rot_vel": float(data.get("Rot_vel", "0")),
                "trans_vel": float(data.get("Trans_vel", "0"))
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        if index < self.window_size - 1:
            window_indices = [0] * (self.window_size - 1 - index) + list(range(index + 1))
        else:
            window_indices = list(range(index - self.window_size + 1, index + 1))
        
        rgb_raw_list, rgb_nav_list = [], []
        bend_list, rot_list, trans_list = [], [], []
        bend_vel_list, rot_vel_list, trans_vel_list = [], [], []
        current_area_list, target_area_list = [], []
        
        for i in window_indices:
            sample = self.samples[i]
            id_int = sample["id_int"]
            file_name = sample["file"]
            rgb_raw_path = os.path.join(self.raw_dir, f"{id_int-1}.png")
            rgb_nav_path = os.path.join(self.nav_dir, file_name)
            rgb_raw_img = Image.open(rgb_raw_path).convert("RGB") if os.path.exists(rgb_raw_path) else Image.new("RGB", (224, 224))
            rgb_nav_img = Image.open(rgb_nav_path).convert("RGB") if os.path.exists(rgb_nav_path) else Image.new("RGB", (224, 224))
            rgb_raw_list.append(self.transform(rgb_raw_img))
            rgb_nav_list.append(self.nav_transform(rgb_nav_img))
            bend_list.append(sample["bend"])
            rot_list.append(sample["rot"])
            trans_list.append(sample["trans"])
            bend_vel_list.append(sample["bend_vel"])
            rot_vel_list.append(sample["rot_vel"])
            trans_vel_list.append(sample["trans_vel"])
            current_area_list.append(sample["current_area"])
            target_area_list.append(sample["target_area"])
        try:
            torch.tensor(target_area_list, dtype=torch.int32)
        except:
            print(target_area_list)
        output = {
            "rgb_raw": torch.stack(rgb_raw_list, dim=0),
            "rgb_nav": torch.stack(rgb_nav_list, dim=0),
            "bend": torch.tensor(bend_list, dtype=torch.float32),
            "rot": torch.tensor(rot_list, dtype=torch.float32),
            "trans": torch.tensor(trans_list, dtype=torch.float32),
            "bend_vel": torch.tensor(bend_vel_list, dtype=torch.float32),
            "rot_vel": torch.tensor(rot_vel_list, dtype=torch.float32),
            "trans_vel": torch.tensor(trans_vel_list, dtype=torch.float32),
            "trigger": torch.tensor(self.samples[index]["trigger"], dtype=torch.float32),
            "time_step": index,
            "traj_idx": 0,
            "current_area_seq": torch.tensor(current_area_list, dtype=torch.int32),
            "target_area_seq": torch.tensor(target_area_list, dtype=torch.int32),
        }
        return output

###############################################
# 4. Helper Functions                         #
###############################################
def create_txt_split_indices(txt_file, val_ratio=0.2, seed=42, save_path="txt_split_indices.json"):
    with open(txt_file, "r") as f:
        lines = f.readlines()
    total = len(lines)
    indices = list(range(total))
    random.seed(seed)
    random.shuffle(indices)
    split_point = int(total * (1 - val_ratio))
    split_indices = {"train": sorted(indices[:split_point]), "val": sorted(indices[split_point:])}
    with open(save_path, "w") as f:
        json.dump(split_indices, f, indent=4)
    print(f"Saved txt split indices to {save_path}")
    return split_indices

def get_loaders(batch_size: int, txt_file: str, raw_dir: str, nav_dir: str, drop_last: bool, shuffle: bool,
                num_workers: int = 4, val_ratio: float = 0.2, seed: int = 42, window_size: int = 5,
                transform=None, nav_transform=None, **kwargs):
    # Create or load the txt split indices.
    split_indices_path = os.path.join(os.path.dirname(txt_file), "txt_split_indices.json")
    if os.path.exists(split_indices_path):
        with open(split_indices_path, "r") as f:
            split_indices = json.load(f)
        print(f"Loaded txt split indices from {split_indices_path}")
    else:
        split_indices = create_txt_split_indices(txt_file, val_ratio=val_ratio, seed=seed, save_path=split_indices_path)

    # Build the normalizer (which computes image stats from training samples only).
    norm_config_file = os.path.join(os.path.dirname(txt_file), "normalizer_config_combined.json")
    normalizer = Normalizer(txt_file=txt_file, raw_image_dir=raw_dir, split_indices=split_indices, norm_config_path=norm_config_file)

    # Define separate image augmentation transforms for training and validation.
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor()
    ])
    nav_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # Create separate dataset instances with their respective transforms.
    train_dataset = Subset(
        CombinedScoresDataset(txt_file, raw_dir, nav_dir, window_size=window_size,
                                transform=train_transform, nav_transform=nav_transform, normalizer=normalizer),
        split_indices["train"]
    )
    val_dataset = Subset(
        CombinedScoresDataset(txt_file, raw_dir, nav_dir, window_size=window_size,
                                transform=val_transform, nav_transform=nav_transform, normalizer=normalizer),
        split_indices["val"]
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers)
    
    # Create test dataset using the entire dataset (all images and labels in original order)
    test_dataset = CombinedScoresDataset(txt_file, raw_dir, nav_dir, window_size=window_size,
                                         transform=val_transform, nav_transform=nav_transform, normalizer=normalizer)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers)

    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Train loader length: {len(train_loader)}")
    print(f"Number of validation samples: {len(val_dataset)}")
    print(f"Validation loader length: {len(val_loader)}")
    print(f"Number of test samples (full dataset): {len(test_dataset)}")
    return train_loader, val_loader, test_loader, normalizer, train_dataset, val_dataset

def visualize_dataloader_output(dataloader, num_samples=2):
    for i in range(len(dataloader)):
        batch = next(iter(dataloader))
        sample = {key: value[0] for key, value in batch.items()}
        print("Trigger:", sample["trigger"].item())
        print("Time step:", sample["time_step"])
        print("Bend sequence:", sample["bend"])
        print("Rot sequence:", sample["rot"])
        print("Trans sequence:", sample["trans"])
        print("Bend_vel sequence:", sample["bend_vel"])
        print("Rot_vel sequence:", sample["rot_vel"])
        print("Trans_vel sequence:", sample["trans_vel"])
        print("Current_area sequence:", sample["current_area_seq"])
        print("Target_area sequence:", sample["target_area_seq"])
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        img_raw = sample["rgb_raw"][-1].permute(1, 2, 0).cpu().numpy()
        img_nav = sample["rgb_nav"][-1].permute(1, 2, 0).cpu().numpy()
        axes[0].imshow(img_raw)
        axes[0].set_title("RGB Raw (last in sequence)")
        axes[0].axis("off")
        axes[1].imshow(img_nav)
        axes[1].set_title("RGB Nav (last in sequence)")
        axes[1].axis("off")
        plt.suptitle("Combined Scores Dataset Sample")
        plt.show()

if __name__ == "__main__":
    txt_file = "/home/haoj/0/dataset.txt"  # Replace with your actual path
    raw_dir = "/home/haoj/0/raw"           # Replace with your actual path
    nav_dir = "/home/haoj/0/nav"           # Replace with your actual path
    batch_size = 2

    train_loader, val_loader, test_loader, normalizer, train_dataset, val_dataset = get_loaders(
        batch_size=batch_size,
        txt_file=txt_file,
        raw_dir=raw_dir,
        nav_dir=nav_dir,
        drop_last=True,
        shuffle=True,
        num_workers=0,  # Set to 0 for simplicity in testing
        val_ratio=0.2,
        seed=42,
        window_size=5
    )

    print("Visualizing training dataloader output:")
    visualize_dataloader_output(test_loader)

    # Example normalization usage
    batch = next(iter(train_loader))
    rgb = normalizer.normalize(batch["rgb_raw"], "rgb_raw")
    bend_vel = normalizer.normalize(batch["bend_vel"].unsqueeze(-1), "Bend_vel")
    rot_vel = normalizer.normalize(batch["rot_vel"].unsqueeze(-1), "Rot_vel")
    trans_vel = normalizer.normalize(batch["trans_vel"].unsqueeze(-1), "Trans_vel")
    print("Normalized RGB shape:", rgb.shape)
    print("Normalized Bend_vel shape:", bend_vel.shape)
