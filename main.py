from flask import Flask, request, jsonify, send_file, Response, render_template_string
import threading
import os
import sys
import time 
import random
import cv2
import requests
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union
import io

import draccus
import numpy as np
import tqdm
from libero.libero import benchmark
import wandb
import pdb
# Append current directory so that interpreter can find experiments.robot
sys.path.append("../..")
from openvla.experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    get_eye_image,
    quat2axisangle,
    save_rollout_video,
)
from openvla.experiments.robot.openvla_utils import get_processor
from openvla.experiments.robot.robot_utils import (
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)
# Other imports remain unchanged

app = Flask(__name__)
instruction = "default"  # Default instruction
instruction_lock = threading.Lock()
latest_frame = np.zeros((224, 224, 3), dtype=np.uint8)
latest_wrist = np.zeros((224, 224, 3), dtype=np.uint8)
first_frame = np.zeros((224, 224, 3), dtype=np.uint8)
first_wrist = np.zeros((224, 224, 3), dtype=np.uint8)

frame_lock = threading.Lock()  # Lock to handle concurrent access to the frame
start = False

@dataclass
class GenerateConfig:
    # Model and environment config (unchanged)
    model_family: str = "openvla"
    task: str = "spatial"
    model: str = None
    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization
    center_crop: bool = True    
    num_steps_wait: int = 10
    num_trials_per_task: int = 2
    run_id_note: Optional[str] = None                # Extra note to add in run ID for logging
    local_log_dir: str = "./experiments/logs"
    use_wandb: bool = False
    wandb_project: str = "YOUR_WANDB_PROJECT"
    wandb_entity: str = "YOUR_WANDB_ENTITY"
    seed: int = 7

# Flask route to update instructions
@app.route('/update_instruction', methods=['POST'])
def update_instruction():
    global instruction
    global start
    data = request.get_json()
    new_instruction = data.get("instruction", "")
    with instruction_lock:
        start = True
        instruction = new_instruction
    return jsonify({"status": "success", "new_instruction": new_instruction})

@app.route('/get_instruction', methods=['GET'])
def get_instruction():
    global instruction
    with instruction_lock:
        return jsonify({"status": "success", "instruction": instruction})

# Flask route to get the latest frame
@app.route('/get_latest_frame', methods=['GET'])
def get_latest_frame():
    global latest_frame
    with frame_lock:
        # Convert the frame to a JPEG image and send it as a response
        _, img_encoded = cv2.imencode('.jpg', latest_frame)
        return send_file(io.BytesIO(img_encoded.tobytes()), mimetype='image/jpeg')

# HTML template

@app.route('/')
def index():
    html_template = open('html.txt', 'r').read()
    return render_template_string(html_template), 200, {'Content-Type': 'text/html'}


@app.route('/video_feed')
def video_feed():
    def generate(frame_type):
        while True:
            time.sleep(0.3)
            with frame_lock:
                frame = latest_frame if frame_type == 'latest' else latest_wrist
                frame = cv2.resize(frame, (512, 512))  # Ensure frames are square
                _, buffer = cv2.imencode('.jpg', frame)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    frame_type = request.args.get('frame_type', 'latest')
    return Response(generate(frame_type), mimetype='multipart/x-mixed-replace; boundary=frame')

def start_flask_server():
    app.run(host='127.0.0.1', port=5000)

def store_frame(obs, step):
    global latest_frame
    global latest_wrist
    global first_frame
    global first_wrist
    try:
        im = obs['agentview_image']
        im = cv2.rotate(im, cv2.ROTATE_180)
        im2 = obs['robot0_eye_in_hand_image']
        im2 = cv2.rotate(im2, cv2.ROTATE_180)
        with frame_lock:
            if step == 0:
                first_frame = im
                first_wrist = im2
            latest_wrist = im2
            latest_frame = im  # Store the most recent frame
    except Exception as e:
        print(f"Failed to store frame: {e}")

@draccus.wrap()
def main(cfg: GenerateConfig) -> None:
    # Initialization remains unchanged
    server_thread = threading.Thread(target=start_flask_server, daemon=True)
    server_thread.start()
    cfg.task_suite_name = f"libero_{cfg.task}"
    cfg.unnorm_key = f"libero_{cfg.model}" if cfg.model != 'base' else "bridge_orig"
    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    print(f"Task suite: {cfg.task_suite_name}")

    def get_instruction():
        global instruction
        with instruction_lock:
            return instruction
    
    if cfg.model == None:
        cfg.pretrained_checkpoint = f"openvla/openvla-7b-finetuned-libero-{cfg.task}"
    elif cfg.model == "base":
        cfg.pretrained_checkpoint = f"openvla/openvla-7b"
    else:
        cfg.pretrained_checkpoint = f"openvla/openvla-7b-finetuned-libero-{cfg.task}"

    assert cfg.pretrained_checkpoint is not None, "cfg.pretrained_checkpoint must not be None!"
    if "image_aug" in cfg.pretrained_checkpoint:
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"
    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"

    # Set random seed
    set_seed_everywhere(cfg.seed)

    # Load model
    model = get_model(cfg)

    # [OpenVLA] Check that the model contains the action un-normalization key
    if cfg.model_family == "openvla":
        # In some cases, the key must be manually modified (e.g. after training on a modified version of the dataset
        # with the suffix "_no_noops" in the dataset name)
        if cfg.unnorm_key not in model.norm_stats and f"{cfg.unnorm_key}_no_noops" in model.norm_stats:
            cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"
        assert cfg.unnorm_key in model.norm_stats, f"Action un-norm key {cfg.unnorm_key} not found in VLA {model.norm_stats.keys()}!"

    # [OpenVLA] Get Hugging Face processor
    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)

    run_id = f"EVAL-{cfg.task_suite_name}-{cfg.model_family}-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    log_file.write(f"Task suite: {cfg.task_suite_name}\n")

    if cfg.use_wandb:
        wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=run_id,
        )

    finished = False
    while not finished:
        task_id = random.sample(range(task_suite.n_tasks), 1)[0]
        task = task_suite.get_task(task_id)

        env, original_task = get_libero_env(task, cfg.model_family, resolution=512)
        print(f'Starting task {task_id}: {original_task}')
        global instruction
        global start
        with instruction_lock:
            instruction = original_task
            start = False
        
        env.reset()
        obs = env.set_init_state(task_suite.get_task_init_states(task_id)[0])
        store_frame(obs, 0)
        t = 0
        replay_images = []
        starting_state = None

        while True:
            try:
                # Initial steps to wait for stabilization (unchanged)
                print(f'Step {t}, current instruction: {get_instruction()}')
                if t < cfg.num_steps_wait:
                    obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                    store_frame(obs, t)
                    if t == 0:
                        print(obs.keys())
                        starting_state = env.get_sim_state()[1:10]
                    t += 1
                    continue

                while True:
                    with instruction_lock:
                        if start:
                            break
                    time.sleep(0.2)
                # Handle instructions
                current_instruction = get_instruction()
                if current_instruction == "reset":
                    curr_state = env.get_sim_state()
                    curr_state[1:10] = starting_state
                    env.set_state(curr_state)
                    action = np.array([0, 0, 0, 0, 0, 0, -1])
                    time.sleep(0.3)
                elif current_instruction == "stop":
                    time.sleep(0.3)
                    action = np.array([0, 0, 0, 0, 0, 0, -1])
                elif current_instruction == "next":
                    break
                elif current_instruction == "quit":
                    finished = True
                    break

                # Default behavior (running previous instruction)
                else:
                    action = get_action(
                        cfg, model,
                        {
                            "full_image": get_libero_image(obs, get_image_resize_size(cfg)),
                            "state": np.concatenate((obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"]))
                        },
                        current_instruction,
                        processor=processor
                    )
                    action = normalize_gripper_action(action, binarize=True)
                    action = invert_gripper_action(action)

                obs, reward, done, info = env.step(action.tolist())
                if done:
                    break
                # Store the frame
                store_frame(obs, t)

                t += 1
            except DeprecationWarning as e:
                print(f"Error: {e}")
                break

        # After-task handling (unchanged)

if __name__ == "__main__":
    main()
