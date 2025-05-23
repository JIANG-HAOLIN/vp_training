#!/usr/bin/env python3
import os
import sys
import argparse
import numpy as np
import torch
import hydra
from hydra.core.hydra_config import HydraConfig
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2  # For video creation
import json
import textwrap

# Import your get_loaders function. Adjust the module path as needed.
from broncho_dl.src.datasets.vp_trigger import get_loaders   # <-- Replace with the actual module name

def inference_classification_visualization(cfg: DictConfig, args: argparse.Namespace):
    # ------------------------------------------------------------------ #
    #  Load VLM predictions → dict[id_int] = instruction string          #
    # ------------------------------------------------------------------ #
    with open(args.pred_path, "r") as f:
        pred_lines = [json.loads(l) for l in f]
    vlm_instr = {int(p["id"]): p["prediction"] for p in pred_lines}

    device = args.device

    # Force no shuffling for deterministic ordering on test set.
    cfg.datasets.dataloader.shuffle = False

    # Load train, validation, and test loaders along with the normalizer.
    train_loader, val_loader, test_loader, normalizer, train_dataset, val_dataset = get_loaders(
        **cfg.datasets.dataloader,
    )
    test_dataset = test_loader.dataset
    # ——— Report total inference steps ———
    total_batches = len(test_loader)
    # If you also want the raw number of samples:
    try:
        total_samples = len(test_loader.dataset)
    except AttributeError:
        total_samples = sum(len(dl.dataset) for dl in test_loader)  # for multi-loader setups
    print(f"Total inference steps (batches): {total_batches}")
    print(f"Total inference samples     : {total_samples}")


    # Instantiate the model using Hydra.
    model = hydra.utils.instantiate(cfg.models.model, _recursive_=False).to(device)

    # Locate and load the pre-trained checkpoint.
    cfgs = HydraConfig.get()
    cfg_path = cfgs.runtime['config_sources'][1]['path']
    checkpoints_folder_path = os.path.abspath(os.path.join(cfg_path, 'checkpoints'))
    ckpt_path = args.ckpt_path  # Provided or default value.
    for p in os.listdir(checkpoints_folder_path):
        if "best" in p and p.split('.')[-1] == 'ckpt':
            ckpt_path = p
    checkpoint_path = os.path.join(checkpoints_folder_path, ckpt_path)
    if os.path.isfile(checkpoint_path):
        print(f"Loading pre-trained model from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint['state_dict']
        # Remove the 'mdl.' prefix if present.
        clone_state_dict = {key[4:]: state_dict[key] for key in state_dict.keys() if key.startswith('mdl.')}
        model.load_state_dict(clone_state_dict)
        model.eval()
    else:
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    # For video visualization, we use a fixed threshold (here 0.8)
    video_thresh = 0.9
    instruction_cooldown = 100  # min steps between two displayed instructions
    last_instr_step = -instruction_cooldown
    current_instr_txt = ""
    instruction_pause_frames = 0
    # Lists to accumulate results for final visualizations.
    all_time_steps = []
    all_pred_scores = []
    all_gt_scores = []
    video_frames = []

    # Process the test dataset.
    with torch.no_grad():
        global_step = 0
        for batch in tqdm(test_loader, total=len(test_loader), desc="Processing Test Set"):
            # Expecting rgb_raw shape: [B, seq_len, C, H, W]
            rgb = batch["rgb_raw"].to(device)
            rgb = normalizer.normalize(rgb, "image") 
            cur_area = batch["current_area_seq"].to(device)
            target_area = batch["target_area_seq"].to(device)
            # Normalize velocity values (each originally a scalar sequence).
            bend_vel = normalizer.normalize(batch["bend_vel"].unsqueeze(-1).to(device), "Bend_vel")
            rot_vel  = normalizer.normalize(batch["rot_vel"].unsqueeze(-1).to(device), "Rot_vel")
            trans_vel = normalizer.normalize(batch["trans_vel"].unsqueeze(-1).to(device), "Trans_vel")
            # Stack normalized velocities: expected shape [B, seq_len, 3].
            vel = torch.cat([bend_vel, rot_vel, trans_vel], dim=-1)
            gt = batch["trigger"].unsqueeze(-1).to(device)

            # Forward pass: model returns scores with shape [B, 1].
            out = model(rgb, cur_area, target_area, vel)

            # Process each sample in the batch.
            batch_size = rgb.size(0)
            # Get time steps from the batch (assumed to be numeric).
            time_steps = batch["time_step"].cpu().tolist()
            for i in range(batch_size):
                pred_score = out[i].item()
                gt_score = gt[i].item()
                time_step = time_steps[i]
                all_time_steps.append(time_step)
                all_pred_scores.append(pred_score)
                all_gt_scores.append(gt_score)

                # ------------------------------------------------------ #
                #   Update instruction text if:                         #
                #   (a) trigger predicted AND (b) 45 steps since last   #
                # ------------------------------------------------------ #
                id_int = test_dataset.samples[time_step]["id_int"]
                if (
                    pred_score > video_thresh
                    and (time_step - last_instr_step) >= instruction_cooldown
                    and id_int in vlm_instr
                ):
                    current_instr_txt = vlm_instr[id_int]
                    last_instr_step = time_step
                    # schedule a pause proportional to the text length (3 frames per word = 0.2s at 15 fps)
                    n_words = len(current_instr_txt.split())
                    instruction_pause_frames = n_words * 3

                # Extract the last raw and navigation images from the sequence.
                raw_img_tensor = batch["rgb_raw"][i, -1]  # shape: [C, H, W]
                nav_img_tensor = batch["rgb_nav"][i, -1]
                # Convert tensor to numpy image (H, W, C) in RGB.
                raw_img = raw_img_tensor.permute(1, 2, 0).cpu().numpy()
                nav_img = nav_img_tensor.permute(1, 2, 0).cpu().numpy()

                # Scale images to [0, 255] and convert to uint8.
                raw_img_disp = cv2.resize((raw_img * 255).astype(np.uint8), (600, 600))
                nav_img_disp = cv2.resize((nav_img * 255).astype(np.uint8), (600, 600))


                # Convert RGB to BGR for OpenCV.
                raw_img_disp = cv2.cvtColor(raw_img_disp, cv2.COLOR_RGB2BGR)
                nav_img_disp = cv2.cvtColor(nav_img_disp, cv2.COLOR_RGB2BGR)

                # Overlay text on images.
                cv2.putText(raw_img_disp, f"Pred: {pred_score:.2f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
                cv2.putText(nav_img_disp, f"GT: {gt_score:.2f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)

                # If predicted score is higher than video_thresh, outline raw image with a yellow box.
                if pred_score > video_thresh:
                    # Draw a rectangle covering the entire image.
                    cv2.rectangle(raw_img_disp, (0, 0),
                                  (raw_img_disp.shape[1]-1, raw_img_disp.shape[0]-1),
                                  (0, 255, 255), 2)
                # If ground truth score is higher than video_thresh, outline nav image with a yellow box.
                if gt_score > video_thresh:
                    cv2.rectangle(nav_img_disp, (0, 0),
                                  (nav_img_disp.shape[1]-1, nav_img_disp.shape[0]-1),
                                  (0, 255, 255), 2)

                # Concatenate the raw (left) and nav (right) images.
                composite_frame = cv2.hconcat([raw_img_disp, nav_img_disp])
                # ---------------------- WHITE BANNER ------------------ #
                banner = np.ones((200, 1200, 3), dtype=np.uint8) * 255
                if current_instr_txt:
                    wrapped = textwrap.wrap(current_instr_txt, width=90)
                    y0 = 40
                    for line in wrapped[:4]:  # max 4 lines
                        cv2.putText(banner, line, (10, y0),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
                        y0 += 40

                full_frame = cv2.vconcat([composite_frame, banner])

                # normal frame
                video_frames.append(full_frame)
                # if we just hit a trigger, append extra copies to pause
                if instruction_pause_frames > 0:
                    for _ in range(instruction_pause_frames):
                        video_frames.append(full_frame)
                    instruction_pause_frames = 0
                global_step += 1

    # --- Compute overall classification accuracy for multiple thresholds ---
    threshold_values = [0.5, 0.6, 0.7, 0.8, 0.9]
    accuracy_dict = {}
    pred_scores_np = np.array(all_pred_scores)
    gt_scores_np = np.array(all_gt_scores)
    for th in threshold_values:
        pred_binary = (pred_scores_np >= th).astype(int)
        gt_binary = (gt_scores_np >= th).astype(int)
        acc = np.mean(pred_binary == gt_binary)
        accuracy_dict[th] = acc
        print(f"Overall classification accuracy at threshold {th}: {acc:.3f}")

    # Plot overall accuracy versus threshold.
    plt.figure(figsize=(8, 5))
    plt.bar([str(th) for th in threshold_values],
            [accuracy_dict[th] for th in threshold_values],
            color='skyblue')
    plt.xlabel("Threshold")
    plt.ylabel("Accuracy")
    plt.title("Overall Classification Accuracy vs Threshold")
    plt.ylim([0, 1])
    plt.tight_layout()
    plt.show()

    # --- Visualization 1: Plot scores over time ---
    plt.figure(figsize=(10, 5))
    plt.plot(all_time_steps, all_pred_scores, marker="o", label="Predicted Score")
    plt.plot(all_time_steps, all_gt_scores, marker="x", label="Ground Truth Score")
    plt.axhline(video_thresh, color="gray", linestyle="--", label=f"Threshold = {video_thresh}")
    plt.xlabel("Time Step")
    plt.ylabel("Score")
    plt.title("Predicted vs Ground Truth Scores over Time (Test Dataset)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --- Visualization 2: Create and store a video visualization ---
    if len(video_frames) > 0:
        # Use the size of the first frame for video dimensions.
        frame_height, frame_width, _ = video_frames[0].shape
        fps = 15  # 30 Hz
        video_filename = "test_visualization_video.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(video_filename, fourcc, fps, (frame_width, frame_height))

        for frame in video_frames:
            video_writer.write(frame)
        video_writer.release()
        print(f"Saved visualization video to {video_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str,
                        default='../../../vp_trigger/vp_training_trigger_predictor/dummy/baseline_debug/broncho_real_trigger_predictor_res_vit_predictor_coswarmup/repeat_trial=1_04-15-11:21:31',
                        help="Path to the Hydra configuration directory")
    parser.add_argument('--device', type=str, default='cpu', help="Device to run on (cpu or cuda)")
    parser.add_argument('--ckpt_path', type=str, default="not needed anymore",
                        help='Checkpoint filename or path override')
    parser.add_argument('--pred_path', type=str,
                        default='/home/haoj/0/idefics3_my/checkpoint-190/predictions.jsonl',
                        help='Path to VLM jsonl predictions')
    args = parser.parse_args()

    # Initialize Hydra and compose the configuration.
    initialize(config_path=args.config_path, version_base=None)
    cfg = compose(config_name='config', return_hydra_config=True)
    OmegaConf.resolve(cfg)
    HydraConfig().cfg = cfg  # (Optional) Set global configuration if needed

    inference_classification_visualization(cfg, args)
    # Clear Hydra global state.
    HydraConfig().clear()
