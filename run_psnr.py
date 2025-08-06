import os
import glob
import re
import numpy as np
import shutil
import sys
import cv2
import argparse
import json

# --- Constants ---
REAL_DATA_ROOT = 'video_data_sample'
NUM_VIDEOS_TO_PROCESS = -1 # Process all videos by default, -1 for all

# --- Helper Functions ---

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
        cap.release()
        return False
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not out.isOpened():
        print(f"    Error: Could not open video file for writing: {output_path}")
        cap.release()
        return False

    sample_indices = np.linspace(0, total_frames - 1, num=num_frames, dtype=int)
    
    for frame_idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            out.write(frame)
        else:
            print(f"    Warning: Could not read frame {frame_idx} from {input_path}.")
            continue
    
    cap.release()
    out.release()
    return True

def calculate_psnr_for_videos(video_path1, video_path2):
    """Calculates the average PSNR between two videos, frame by frame."""
    cap1 = cv2.VideoCapture(video_path1)
    cap2 = cv2.VideoCapture(video_path2)

    if not cap1.isOpened() or not cap2.isOpened():
        print("    Error: Could not open one or both video files for PSNR calculation.")
        if cap1.isOpened(): cap1.release()
        if cap2.isOpened(): cap2.release()
        return 0.0

    psnr_values = []
    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if not ret1 or not ret2:
            break

        psnr = cv2.PSNR(frame1, frame2)
        # PSNR can be inf if frames are identical, handle this case if necessary
        if not np.isinf(psnr):
            psnr_values.append(psnr)
    
    cap1.release()
    cap2.release()

    if not psnr_values:
        return 0.0
    
    return np.mean(psnr_values)

# --- Main Execution Logic ---

def run(args):
    """Main function to compute PSNR and log results."""
    GEN_DATA_ROOT = args.gen_data_root
    LOG_FILE = args.log_file
    
    os.makedirs(GEN_DATA_ROOT, exist_ok=True)

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
                    continue
        print(f"Found {len(completed_videos)} completed videos to skip.")

    log_handle = open(LOG_FILE, 'a')

    datasets = [d for d in os.listdir(REAL_DATA_ROOT) if os.path.isdir(os.path.join(REAL_DATA_ROOT, d))]
    
    for dataset in datasets:
        print(f"\nProcessing dataset: {dataset}")
        real_dataset_path = os.path.join(REAL_DATA_ROOT, dataset)
        gen_dataset_path = os.path.join(GEN_DATA_ROOT, dataset)
        os.makedirs(gen_dataset_path, exist_ok=True)
        
        video_files = sorted(glob.glob(os.path.join(real_dataset_path, '*.mp4')))
        if NUM_VIDEOS_TO_PROCESS > 0:
            video_files = video_files[:NUM_VIDEOS_TO_PROCESS]
        print(f"  Processing {len(video_files)} videos in this dataset...")

        for real_video_path in video_files:
            video_filename = os.path.basename(real_video_path)
            
            if (dataset, video_filename) in completed_videos:
                print(f"  Skipping already processed video: {video_filename}")
                continue

            gen_video_path = os.path.join(gen_dataset_path, video_filename)
            
            if not os.path.exists(gen_video_path):
                print(f"  Skipping {video_filename}: Corresponding generated video not found.")
                continue

            print(f"  Processing video: {video_filename}")

            temp_files_to_clean = []
            try:
                real_cap = cv2.VideoCapture(real_video_path)
                real_frame_count = int(real_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                real_cap.release()

                gen_cap = cv2.VideoCapture(gen_video_path)
                gen_frame_count = int(gen_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                gen_cap.release()
                
                target_frame_count = min(real_frame_count, gen_frame_count)

                path_for_real = real_video_path
                path_for_gen = gen_video_path

                if real_frame_count > target_frame_count:
                    print(f"    Sampling real video from {real_frame_count} to {target_frame_count} frames.")
                    path_for_real = os.path.join(gen_dataset_path, f"temp_psnr_real_{video_filename}")
                    temp_files_to_clean.append(path_for_real)
                    if not sample_video_frames(real_video_path, path_for_real, target_frame_count):
                        raise IOError(f"Failed to sample real video: {real_video_path}")
                
                if gen_frame_count > target_frame_count:
                    print(f"    Sampling generated video from {gen_frame_count} to {target_frame_count} frames.")
                    path_for_gen = os.path.join(gen_dataset_path, f"temp_psnr_gen_{video_filename}")
                    temp_files_to_clean.append(path_for_gen)
                    if not sample_video_frames(gen_video_path, path_for_gen, target_frame_count):
                        raise IOError(f"Failed to sample generated video: {gen_video_path}")

                psnr_score = calculate_psnr_for_videos(path_for_real, path_for_gen)
                
                log_entry = {
                    "dataset": dataset,
                    "video_filename": video_filename,
                    "psnr_score": psnr_score,
                    "status": "completed"
                }
                log_handle.write(json.dumps(log_entry) + '\n')
                log_handle.flush()
                print(f"    Average PSNR: {psnr_score:.4f}")

            except Exception as e:
                print(f"      Error calculating PSNR for {video_filename}: {e}")
                log_entry = {"dataset": dataset, "video_filename": video_filename, "error": str(e), "status": "failed"}
                log_handle.write(json.dumps(log_entry) + '\n')
                log_handle.flush()
            
            finally:
                for temp_file in temp_files_to_clean:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
    
    log_handle.close()

    print("\n--- Final Statistics ---")
    all_results = {}
    with open(LOG_FILE, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                if data.get('status') == 'completed':
                    dset = data['dataset']
                    if dset not in all_results:
                        all_results[dset] = {'psnr': []}
                    all_results[dset]['psnr'].append(data['psnr_score'])
            except (json.JSONDecodeError, KeyError):
                continue
    
    final_summary = "\n\n--- Final PSNR Statistics ---\n"
    for dataset, results in all_results.items():
        if results['psnr']:
            psnr_mean = np.mean(results['psnr'])
            psnr_std = np.std(results['psnr'])
            
            summary = f"\nDataset: {dataset}\n"
            summary += f"  PSNR: Mean={psnr_mean:.2f}, Std={psnr_std:.2f} (from {len(results['psnr'])} videos)\n"
            print(summary)
            final_summary += summary
        else:
            summary = f"\nDataset: {dataset}\n  No valid PSNR results to analyze.\n"
            print(summary)
            final_summary += summary

    with open(LOG_FILE, 'a') as log_file_append:
        log_file_append.write(final_summary)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run PSNR metrics on generated videos.")
    parser.add_argument(
        '--gen_data_root', 
        type=str, 
        default='video_data_gen_finetune',
        help='Directory where generated videos are located.'
    )
    parser.add_argument(
        '--log_file',
        type=str,
        default='psnr_log.jsonl',
        help='File to write logs and results to (JSON Lines format).'
    )
    args = parser.parse_args()
    run(args) 