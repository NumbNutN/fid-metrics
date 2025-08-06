import os
import glob
import argparse
from multiprocessing import Process, Queue, Lock, set_start_method
from itertools import cycle
import queue
import cv2

from gen_video import VideoGenerator

REAL_DATA_ROOT = 'video_data_sample'

def get_first_frame(video_path):
    """Extracts the first frame of a video and returns it as a numpy array."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Warning: Could not open video for reading first frame: {video_path}")
        return None
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print(f"Warning: Could not read first frame from {video_path}")
        return None
    return frame

def worker(task_queue, args, log_lock):
    """A worker process that generates a video for each task in the queue."""
    # Each worker gets its own video generator
    video_generator = VideoGenerator(ip_address=args.ip_address)
    
    while True:
        try:
            task = task_queue.get(timeout=3)
        except queue.Empty:
            print(f"Worker on port {args.ports[os.getpid() % len(args.ports)]} finished.") # Just for info
            break

        dataset, real_video_path, prompt_text, port = task
        video_filename = os.path.basename(real_video_path)
        gen_dataset_path = os.path.join(args.gen_data_root, dataset)
        os.makedirs(gen_dataset_path, exist_ok=True)
        gen_video_path = os.path.join(gen_dataset_path, video_filename)
        
        print(f"  [Port {port}] Attempting to generate video: {video_filename}")

        # This check is redundant due to checkpointing but provides an extra layer of safety
        if os.path.exists(gen_video_path):
            print(f"    [Port {port}] Video already exists, skipping: {gen_video_path}")
            continue

        first_frame = get_first_frame(real_video_path)
        generation_successful = False
        if first_frame is not None:
            generation_successful = video_generator.generate(port, first_frame, prompt_text, gen_video_path)

        # Log the result
        with log_lock:
            with open(args.log_file, 'a') as log_handle:
                status = "completed" if generation_successful else "failed"
                log_handle.write(f"{dataset},{video_filename},{status}\n")

        if generation_successful:
            print(f"    [Port {port}] Successfully generated: {gen_video_path}")
        else:
            print(f"    [Port {port}] Failed to generate: {video_filename}")

def run_generation(args):
    """Main function to set up and run the video generation processes."""
    os.makedirs(args.gen_data_root, exist_ok=True)

    # --- Checkpoint Loading and Log Cleanup ---
    completed_videos = set()
    latest_statuses = {}
    if os.path.exists(args.log_file):
        print(f"Resuming from existing log file: {args.log_file}")
        with open(args.log_file, 'r') as f:
            for line in f:
                try:
                    parts = line.strip().split(',')
                    if len(parts) == 3:
                        # Overwrite with the latest status found in the log
                        latest_statuses[(parts[0], parts[1])] = parts[2]
                except (IndexError, ValueError):
                    continue # Skip corrupted lines
        
        # Now, rebuild the set of completed videos and clean the log
        if latest_statuses:
            with open(args.log_file, 'w') as f: # Overwrite with clean log
                for (dataset, video_filename), status in latest_statuses.items():
                    f.write(f"{dataset},{video_filename},{status}\n")
                    if status == 'completed':
                        completed_videos.add((dataset, video_filename))

        print(f"Found {len(completed_videos)} completed videos to skip.")

    # --- Create Task Queue ---
    task_queue = Queue()
    ports_cycle = cycle(args.ports)
    
    total_tasks = 0
    datasets = [d for d in os.listdir(REAL_DATA_ROOT) if os.path.isdir(os.path.join(REAL_DATA_ROOT, d))]
    # reverse the datasets
    for dataset in reversed(datasets):
        real_dataset_path = os.path.join(REAL_DATA_ROOT, dataset)
        video_files = sorted(glob.glob(os.path.join(real_dataset_path, '*.mp4')))

        print(f"Queueing {len(video_files)} videos from dataset: {dataset}...")

        for real_video_path in video_files:
            video_filename = os.path.basename(real_video_path)
            
            if (dataset, video_filename) in completed_videos:
                continue

            text_path = os.path.splitext(real_video_path)[0] + '.txt'
            prompt_text = "N/A"
            if os.path.exists(text_path):
                with open(text_path, 'r') as f:
                    prompt_text = f.read().strip()

            port = next(ports_cycle)
            task_queue.put((dataset, real_video_path, prompt_text, port))
            total_tasks += 1

    if total_tasks == 0:
        print("No new videos to generate. Exiting.")
        return

    print(f"\nTotal new tasks to process: {total_tasks}")
    print(f"Starting {len(args.ports)} worker processes...")

    # --- Create and Start Worker Processes ---
    log_lock = Lock()
    processes = []
    for _ in range(len(args.ports)):
        p = Process(target=worker, args=(task_queue, args, log_lock))
        processes.append(p)
        p.start()
        
    for p in processes:
        p.join()
        
    print("\nAll generation workers have finished.")

if __name__ == '__main__':
    # Set start method for CUDA compatibility in forked processes
    set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser(description="Generate videos in parallel using a video generation service.")
    parser.add_argument(
        '--gen_data_root', 
        type=str, 
        default='video_data_gen_only',
        help='Directory to save generated videos. Defaults to video_data_gen_only.'
    )
    parser.add_argument(
        '--log_file',
        type=str,
        default='generation_log.csv',
        help='File to write generation status logs to (CSV format). Defaults to generation_log.csv.'
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
    run_generation(args) 