import os
import glob
import re
import numpy as np
import shutil
import sys
from omegaconf import OmegaConf
import torch
from rich.progress import track
import cv2
import argparse
import json

# --- Import logic from local files ---
from fid_metrics import (
    build_loaders,
    build_model,
    calculate_fid,
    postprocess_i2d_pred,
)
from gen_video import VideoGenerator

# --- Constants ---
REAL_DATA_ROOT = 'video_data_sample'
# GEN_DATA_ROOT and LOG_FILE will be handled by argparse
NUM_VIDEOS_TO_PROCESS = 5 # Limit number of videos for faster debugging

def parse_metrics_from_string(output):
    """Parses FID and FVD from a string."""
    fid_match = re.search(r'FID: ([\d\.]+)', output)
    fvd_match = re.search(r'FVD: ([\d\.]+)', output)
    fid = float(fid_match.group(1)) if fid_match else None
    fvd = float(fvd_match.group(1)) if fvd_match else None
    return fid, fvd

def get_first_frame(video_path):
    """Extracts the first frame of a video and returns it as a numpy array."""
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print(f"Warning: Could not read first frame from {video_path}")
        return None
    return frame

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

def truncate_video(input_path, output_path, frame_limit):
    """Reads a video, writes the first `frame_limit` frames to a new video file."""
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"    Error: Could not open video file for reading: {input_path}")
        return False
    
    # Get video properties to create a writer
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not out.isOpened():
        print(f"    Error: Could not open video file for writing: {output_path}")
        cap.release()
        return False

    frame_num = 0
    while frame_num < frame_limit:
        ret, frame = cap.read()
        if not ret:
            break # Reached end of video before frame_limit
        out.write(frame)
        frame_num += 1
    
    cap.release()
    out.release()
    return True

def sample_video_frames(input_path, output_path, num_frames):
    """
    Reads a video and writes a new video file with `num_frames` uniformly sampled frames.
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"    Error: Could not open video file for reading: {input_path}")
        return False
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        print(f"    Warning: Video has no frames or invalid metadata: {input_path}")
        return False
        
    # Get video properties for the writer
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not out.isOpened():
        print(f"    Error: Could not open video file for writing: {output_path}")
        cap.release()
        return False

    # Generate indices of frames to sample
    sample_indices = np.linspace(0, total_frames - 1, num=num_frames, dtype=int)
    
    for frame_idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            out.write(frame)
        else:
            # If we fail to read a frame, it might be the end of a corrupted file
            print(f"    Warning: Could not read frame {frame_idx} from {input_path}. Output may have fewer frames.")
            continue
    
    cap.release()
    out.release()
    return True

def run(args):
    """Main function to generate videos, compute metrics, and log results."""
    # --- Setup from args ---
    GEN_DATA_ROOT = args.gen_data_root
    LOG_FILE = args.log_file
    
    os.makedirs(GEN_DATA_ROOT, exist_ok=True)

    # --- Checkpoint Loading ---
    completed_videos = set()
    if os.path.exists(LOG_FILE):
        print(f"Resuming from existing log file: {LOG_FILE}")
        with open(LOG_FILE, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if data.get('status') == 'completed':
                        completed_videos.add((data['dataset'], data['video_filename']))
                except json.JSONDecodeError:
                    continue # Skip corrupted lines
        print(f"Found {len(completed_videos)} completed videos to skip.")

    # Open log file in append mode
    log_handle = open(LOG_FILE, 'a')

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

    # --- Initialize Video Generator ---
    video_generator = VideoGenerator(ip_address=args.ip_address, ports=args.ports)

    # --- Data and Configs ---
    all_results = {}
    datasets = [d for d in os.listdir(REAL_DATA_ROOT) if os.path.isdir(os.path.join(REAL_DATA_ROOT, d))]
    
    fid_data_cfg = OmegaConf.create({
        "dataset": {"resize_shape": [256, 512]},
        "batch_size": 4, "num_workers": 1
    })
    fvd_data_cfg = OmegaConf.create({
        "dataset": {
            "sequence_length": 16, 
            "resize_shape": [224, 224]
        },
        "batch_size": 4, "num_workers": 1
    })

    for dataset in datasets:
        print(f"\nProcessing dataset: {dataset}")
        all_results[dataset] = {'fid': [], 'fvd': []}
        
        real_dataset_path = os.path.join(REAL_DATA_ROOT, dataset)
        gen_dataset_path = os.path.join(GEN_DATA_ROOT, dataset)
        os.makedirs(gen_dataset_path, exist_ok=True)
        
        video_files = sorted(glob.glob(os.path.join(real_dataset_path, '*.mp4')))
        video_files = video_files[:]
        print(f"  Processing {len(video_files)} videos in this dataset...")

        for real_video_path in video_files:
            video_filename = os.path.basename(real_video_path)
            
            # --- Checkpoint Skip ---
            if (dataset, video_filename) in completed_videos:
                print(f"  Skipping already processed video: {video_filename}")
                continue

            gen_video_path = os.path.join(gen_dataset_path, video_filename)
            
            # Associated text file
            text_path = os.path.splitext(real_video_path)[0] + '.txt'
            prompt_text = "N/A"
            if os.path.exists(text_path):
                with open(text_path, 'r') as f:
                    prompt_text = f.read().strip()

            print(f"  Processing video: {video_filename} with prompt: '{prompt_text[:30]}...'")
            
            # --- Task 1: Generate Video (if it doesn't exist) ---
            if not os.path.exists(gen_video_path):
                first_frame = get_first_frame(real_video_path)
                generation_successful = False
                if first_frame is not None:
                    generation_successful = video_generator.generate(first_frame, prompt_text, gen_video_path)

                if not generation_successful:
                    print("    Video generation failed. Falling back to copying original video.")
                    shutil.copy(real_video_path, gen_video_path)
            else:
                print(f"    Found existing generated video, skipping generation: {gen_video_path}")

            temp_files_to_clean = []
            try:
                # --- Step 1: Uniformly sample longer video to match shorter one ---
                real_cap = cv2.VideoCapture(real_video_path)
                real_frame_count = int(real_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                real_cap.release()

                gen_cap = cv2.VideoCapture(gen_video_path)
                gen_frame_count = int(gen_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                gen_cap.release()
                
                if real_frame_count == 0 or gen_frame_count == 0:
                    raise ValueError("One of the videos has 0 frames.")

                target_frame_count = min(real_frame_count, gen_frame_count)

                path_for_real_sampled = real_video_path
                path_for_gen_sampled = gen_video_path

                if real_frame_count > target_frame_count:
                    print(f"    Sampling real video from {real_frame_count} to {target_frame_count} frames.")
                    path_for_real_sampled = os.path.join(gen_dataset_path, f"temp_sampled_real_{video_filename}")
                    temp_files_to_clean.append(path_for_real_sampled)
                    if not sample_video_frames(real_video_path, path_for_real_sampled, target_frame_count):
                        raise IOError(f"Failed to sample real video: {real_video_path}")
                
                if gen_frame_count > target_frame_count:
                    print(f"    Sampling generated video from {gen_frame_count} to {target_frame_count} frames.")
                    path_for_gen_sampled = os.path.join(gen_dataset_path, f"temp_sampled_gen_{video_filename}")
                    temp_files_to_clean.append(path_for_gen_sampled)
                    if not sample_video_frames(gen_video_path, path_for_gen_sampled, target_frame_count):
                        raise IOError(f"Failed to sample generated video: {gen_video_path}")

                # --- Step 2: Truncate the sampled videos to their first 16 frames ---
                print("    Creating 16-frame clips from sampled videos for metrics.")
                temp_real_final = os.path.join(gen_dataset_path, f"temp_final_real_{video_filename}")
                temp_gen_final = os.path.join(gen_dataset_path, f"temp_final_gen_{video_filename}")
                temp_files_to_clean.extend([temp_real_final, temp_gen_final])

                if not truncate_video(path_for_real_sampled, temp_real_final, 16):
                    raise IOError(f"Failed to truncate real video: {path_for_real_sampled}")
                if not truncate_video(path_for_gen_sampled, temp_gen_final, 16):
                    raise IOError(f"Failed to truncate generated video: {path_for_gen_sampled}")
                
                paths = [temp_real_final, temp_gen_final]

                # FID feature shape
                print(f"    FID feature shape: {real_fid_feats.shape}")
                print(f"    FVD feature shape: {real_fvd_feats.shape}")
                
                # --- Calculate FID (first 16 frames of sampled videos) ---
                fid_loaders = build_loaders('fid', paths, fid_data_cfg)
                real_fid_feats = get_features(fid_loaders[0], fid_model, 'fid', device)
                gen_fid_feats = get_features(fid_loaders[1], fid_model, 'fid', device)
                fid_score = calculate_fid(real_fid_feats, gen_fid_feats)
                
                # --- Calculate FVD (first 16 frames of sampled videos) ---
                fvd_loaders = build_loaders('fvd', paths, fvd_data_cfg)
                real_fvd_feats = get_features(fvd_loaders[0], fvd_model, 'fvd', device, model_subtype=fvd_model_cfg.type)
                gen_fvd_feats = get_features(fvd_loaders[1], fvd_model, 'fvd', device, model_subtype=fvd_model_cfg.type)
                fvd_score = calculate_fid(real_fvd_feats, gen_fvd_feats)

                # --- Task 3: Log Results (Structured JSON) ---
                log_entry = {
                    "dataset": dataset,
                    "video_filename": video_filename,
                    "fid_score": fid_score,
                    "fvd_score": fvd_score,
                    "status": "completed"
                }
                log_handle.write(json.dumps(log_entry) + '\n')
                log_handle.flush() # Ensure it's written immediately
                print(f"    FID: {fid_score:.4f}, FVD: {fvd_score:.4f}")

            except Exception as e:
                print(f"      Error calculating metrics for {video_filename}: {e}")
                log_entry = {
                    "dataset": dataset,
                    "video_filename": video_filename,
                    "error": str(e),
                    "status": "failed"
                }
                log_handle.write(json.dumps(log_entry) + '\n')
                log_handle.flush()
            
            finally:
                # --- Cleanup ---
                for temp_file in temp_files_to_clean:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
    
    log_handle.close()

    # --- Task 3: Final Analysis (from log file) ---
    print("\n--- Final Statistics ---")
    all_results = {}
    with open(LOG_FILE, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                if data.get('status') == 'completed':
                    dset = data['dataset']
                    if dset not in all_results:
                        all_results[dset] = {'fid': [], 'fvd': []}
                    all_results[dset]['fid'].append(data['fid_score'])
                    all_results[dset]['fvd'].append(data['fvd_score'])
            except (json.JSONDecodeError, KeyError):
                continue
    
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
    parser = argparse.ArgumentParser(description="Run FID and FVD metrics on generated videos.")
    parser.add_argument(
        '--gen_data_root', 
        type=str, 
        default='video_data_gen_finetune',
        help='Directory to save generated videos. Defaults to video_data_gen_finetune.'
    )
    parser.add_argument(
        '--log_file',
        type=str,
        default='metrics_log.jsonl',
        help='File to write logs and results to (JSON Lines format). Defaults to metrics_log.jsonl.'
    )
    parser.add_argument(
        '--ip_address',
        type=str,
        default='172.16.204.187',
        help='IP address of the video generation server.'
    )
    parser.add_argument(
        '--ports',
        nargs='+',
        type=int,
        default=[23991],
        help='List of ports for the video generation server.'
    )
    args = parser.parse_args()
    run(args)