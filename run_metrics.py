import os
import glob
import re
import numpy as np
import shutil
import sys
from omegaconf import OmegaConf
import torch
from rich.progress import track

# --- Import logic from fid_metrics ---
from fid_metrics import (
    build_loaders,
    build_model,
    calculate_fid,
    postprocess_i2d_pred,
)

# --- Constants ---
REAL_DATA_ROOT = 'video_data_sample'
GEN_DATA_ROOT = 'video_data_gen'
LOG_FILE = 'metrics_log.txt'
NUM_VIDEOS_TO_PROCESS = 5 # Limit number of videos for faster debugging

def parse_metrics_from_string(output):
    """Parses FID and FVD from a string."""
    fid_match = re.search(r'FID: ([\d\.]+)', output)
    fvd_match = re.search(r'FVD: ([\d\.]+)', output)
    fid = float(fid_match.group(1)) if fid_match else None
    fvd = float(fvd_match.group(1)) if fvd_match else None
    return fid, fvd

def get_features(dl, model, metric_type, device, model_subtype="styleganv"):
    """Extracts features for a given dataloader and model."""
    feats = []
    for x in track(dl, description=f'{metric_type} feature extraction'):
        x = x.to(device)
        if metric_type == 'fid' and x.dim() == 5:
            x = x.squeeze(0).transpose(0, 1)
        elif metric_type == 'fvd':
            x = x * 2 - 1  # [-1, 1]
        
        with torch.no_grad():
            if metric_type == 'fid':
                pred = model(x)
                pred = postprocess_i2d_pred(pred)
            elif metric_type == 'fvd':
                if model_subtype == 'styleganv':
                    pred = model(x, return_features=True)
                else:
                    pred = model(x)
        feats.append(pred.cpu().numpy())
    return np.concatenate(feats, axis=0)


def run():
    """Main function to generate videos, compute metrics, and log results."""
    # --- Setup ---
    os.makedirs(GEN_DATA_ROOT, exist_ok=True)
    with open(LOG_FILE, 'w') as log:
        log.write("--- Metrics Calculation Log ---\n\n")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # --- Load Models Once ---
    print("Loading FID and FVD models...")
    fid_model_cfg = OmegaConf.create({"dims": 2048})
    fvd_model_cfg = OmegaConf.create({
        "type": "styleganv",
        "path": "i3d_torchscript.pt" 
    })
    fid_model = build_model('fid', fid_model_cfg).to(device).eval()
    fvd_model = build_model('fvd', fvd_model_cfg).to(device).eval()
    print("Models loaded.")

    # --- Data and Configs ---
    all_results = {}
    datasets = [d for d in os.listdir(REAL_DATA_ROOT) if os.path.isdir(os.path.join(REAL_DATA_ROOT, d))]
    
    fid_data_cfg = OmegaConf.create({
        "dataset": {"resize_shape": [256, 512]},
        "batch_size": 64, "num_workers": 16
    })
    fvd_data_cfg = OmegaConf.create({
        "dataset": {"sequence_length": 16, "resize_shape": [224, 224]},
        "batch_size": 4, "num_workers": 16
    })

    for dataset in datasets:
        print(f"\nProcessing dataset: {dataset}")
        all_results[dataset] = {'fid': [], 'fvd': []}
        
        real_dataset_path = os.path.join(REAL_DATA_ROOT, dataset)
        gen_dataset_path = os.path.join(GEN_DATA_ROOT, dataset)
        os.makedirs(gen_dataset_path, exist_ok=True)
        
        video_files = sorted(glob.glob(os.path.join(real_dataset_path, '*.mp4')))
        video_files = video_files[:NUM_VIDEOS_TO_PROCESS]
        print(f"  Processing {len(video_files)} videos in this dataset...")

        for real_video_path in video_files:
            video_filename = os.path.basename(real_video_path)
            gen_video_path = os.path.join(gen_dataset_path, video_filename)

            print(f"  Processing video: {video_filename}")
            # --- Task 1: Generate Video (Simulated) ---
            shutil.copy(real_video_path, gen_video_path)

            try:
                paths = [real_video_path, gen_video_path]
                
                # --- Calculate FID ---
                fid_loaders = build_loaders('fid', paths, fid_data_cfg)
                real_fid_feats = get_features(fid_loaders[0], fid_model, 'fid', device)
                gen_fid_feats = get_features(fid_loaders[1], fid_model, 'fid', device)
                fid_score = calculate_fid(real_fid_feats, gen_fid_feats)
                
                # --- Calculate FVD ---
                fvd_loaders = build_loaders('fvd', paths, fvd_data_cfg)
                real_fvd_feats = get_features(fvd_loaders[0], fvd_model, 'fvd', device, model_subtype=fvd_model_cfg.type)
                gen_fvd_feats = get_features(fvd_loaders[1], fvd_model, 'fvd', device, model_subtype=fvd_model_cfg.type)
                fvd_score = calculate_fid(real_fvd_feats, gen_fvd_feats) # FID function is reused for FVD calculation

                # --- Task 3: Log Results ---
                log_entry = f"Dataset: {dataset}, Video: {video_filename}, FID: {fid_score:.4f}, FVD: {fvd_score:.4f}\n"
                print(f"    FID: {fid_score:.4f}, FVD: {fvd_score:.4f}")
                with open(LOG_FILE, 'a') as log_file:
                    log_file.write(log_entry)
                
                all_results[dataset]['fid'].append(fid_score)
                all_results[dataset]['fvd'].append(fvd_score)

            except Exception as e:
                print(f"      Error calculating metrics for {video_filename}: {e}")
                import traceback
                with open(LOG_FILE, 'a') as log_file:
                    log_file.write(f"Dataset: {dataset}, Video: {video_filename}, ERROR: Exception occurred.\n")
                    traceback.print_exc(file=log_file)
    
    # --- Task 3: Final Analysis ---
    print("\n--- Final Statistics ---")
    final_summary = "\n\n--- Final Statistics ---\n"
    for dataset, results in all_results.items():
        if results['fid'] and results['fvd']:
            fid_mean = np.mean(results['fid'])
            fid_std = np.std(results['fid'])
            fvd_mean = np.mean(results['fvd'])
            fvd_std = np.std(results['fvd'])
            
            summary = f"\nDataset: {dataset}\n"
            summary += f"  FID: Mean={fid_mean:.2f}, Std={fid_std:.2f} (from {len(results['fid'])} videos)\n"
            summary += f"  FVD: Mean={fvd_mean:.2f}, Std={fvd_std:.2f} (from {len(results['fvd'])} videos)\n"
            
            print(summary)
            final_summary += summary
        else:
            summary = f"\nDataset: {dataset}\n  No valid results to analyze.\n"
            print(summary)
            final_summary += summary

    with open(LOG_FILE, 'a') as log:
        log.write(final_summary)

if __name__ == '__main__':
    run()