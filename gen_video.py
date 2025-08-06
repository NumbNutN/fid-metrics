import cv2
import numpy as np
import subprocess
import multiprocessing
import requests
import json
from base64 import b64encode, b64decode
import os
import logging
import urllib3

# Setup basic logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Helper Functions (from original file) ---

def save_video(ffmpeg_cmd, images):
    try:
        proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        for image in images:
            img = cv2.imdecode(np.frombuffer(b64decode(image), np.uint8), cv2.IMREAD_COLOR)
            proc.stdin.write(img.tobytes())
        proc.stdin.close()
        proc.wait()
        if proc.returncode != 0:
            error_msg = proc.stderr.read().decode()
            logger.error(f"FFMPEG Error: {error_msg}")
    except Exception as e:
        logger.error(f"Failed to save video: {e}")

def save_videos(videos, width, height, fps=8):
    workers = []
    for k, v in videos.items():
        os.makedirs(os.path.dirname(k), exist_ok=True)
        ffmpeg_cmd = [
            'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
            '-s', f'{width}x{height}', '-pix_fmt', 'bgr24', '-r', str(fps),
            '-i', '-', '-c:v', 'libx264', '-preset', 'veryslow',
            '-crf', '10', '-threads', '1', '-pix_fmt', 'yuv420p',
            '-loglevel', 'error', k
        ]
        worker = multiprocessing.Process(target=save_video, args=(ffmpeg_cmd, v))
        worker.start()
        workers.append(worker)
    for worker in workers:
        worker.join()

def worker(ip_address, port, headers, data, verify, result_queue):
    try:
        logger.info(f"Waiting for response from {ip_address}:{port}")
        response = requests.post(f"https://{ip_address}:{port}", headers=headers, data=json.dumps(data), verify=verify)
        response.raise_for_status()
        result_queue.put(response.json())
        logger.info(f"Response from port {port} got")
    except requests.exceptions.RequestException as e:
        logger.error(f"Request to port {port} failed: {e}")
        result_queue.put(None)


# --- Main Generator Class ---

class VideoGenerator:
    def __init__(self, ip_address='172.16.204.187', ports=[23991]):
        self.ip_address = ip_address
        self.ports = ports
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    def generate(self, first_frame_image, prompt_text, output_path):
        """
        Generates a video by calling an external service.
        Returns True on success, False on failure.
        """
        logger.info(f"Attempting to generate video for prompt: '{prompt_text}'")
        
        headers = {"Content-Type": "application/json"}
        # Use a multiprocessing queue to get results from worker processes
        result_queue = multiprocessing.Manager().Queue()
        jobs = []

        for port in self.ports:
            data = {
                "prompt": prompt_text, 
                "img": b64encode(cv2.imencode(".jpg", first_frame_image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])[1].tobytes()).decode("utf-8"), 
                "seed": 1234, # Using a fixed seed for consistency
                "password": "r49h8fieuwK"
            }
            p = multiprocessing.Process(target=worker, args=(self.ip_address, port, headers, data, False, result_queue))
            jobs.append(p)
            p.start()

        for p in jobs:
            p.join()

        responses = []
        while not result_queue.empty():
            res = result_queue.get()
            if res:
                responses.append(res)

        if not responses or not responses[0]:
            logger.error("Video generation failed. No valid responses from server.")
            return False

        try:
            videos_to_save = {}
            sample_image = responses[0][0]
            height, width, _ = cv2.imdecode(np.frombuffer(b64decode(sample_image), np.uint8), cv2.IMREAD_COLOR).shape
            
            # Save the first successful response to the specified output path
            videos_to_save[output_path] = responses[0]
            save_videos(videos_to_save, width, height)
            
            logger.info(f"Video policy generated at: {output_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to process server response and save video: {e}")
            return False