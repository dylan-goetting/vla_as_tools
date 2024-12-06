import os
import sys
import time 
import random
import cv2
import requests
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

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
from experiments.robot.robot_utils import (
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)


@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    # pretrained_checkpoint: Union[str, Path] = ""     # Pretrained checkpoint path
    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task: str = "spatial"
    # task_suite_name: str = "libero_spatial"          # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 2                    # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add in run ID for logging
    local_log_dir: str = "./experiments/logs"        # Local directory for eval logs

    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_project: str = "YOUR_WANDB_PROJECT"        # Name of W&B project to log to (use default!)
    wandb_entity: str = "YOUR_WANDB_ENTITY"          # Name of entity to log under

    seed: int = 7                                    # Random Seed (for reproducibility)
    test = False
    max_tasks = 100
    # fmt: on

def send_frame_to_server(obs, step):
    try:
        im = obs['agentview_image']
        im = cv2.rotate(im, cv2.ROTATE_180)
        cv2.putText(im, f"Step {step}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        _, img_encoded = cv2.imencode('.jpg', im)
        response = requests.post('http://localhost:5000/update_frame', data=img_encoded.tobytes())
        if response.status_code != 200:
            print(f"Error updating frame: {response.status_code}")
    except Exception as e:
        print(f"Failed to send frame: {e}")

@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> None:
    cfg.pretrained_checkpoint = f"openvla/openvla-7b-finetuned-libero-{cfg.task}"
    assert cfg.pretrained_checkpoint is not None, "cfg.pretrained_checkpoint must not be None!"
    if "image_aug" in cfg.pretrained_checkpoint:
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"
    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"

    # Set random seed
    set_seed_everywhere(cfg.seed)

    # [OpenVLA] Set action un-normalization key
    cfg.task_suite_name = f"libero_{cfg.task}"
    cfg.unnorm_key = cfg.task_suite_name
    
    # Load model
    model = get_model(cfg)

    # [OpenVLA] Check that the model contains the action un-normalization key
    if cfg.model_family == "openvla":
        # In some cases, the key must be manually modified (e.g. after training on a modified version of the dataset
        # with the suffix "_no_noops" in the dataset name)
        if cfg.unnorm_key not in model.norm_stats and f"{cfg.unnorm_key}_no_noops" in model.norm_stats:
            cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"
        assert cfg.unnorm_key in model.norm_stats, f"Action un-norm key {cfg.unnorm_key} not found in VLA `norm_stats`!"

    # [OpenVLA] Get Hugging Face processor
    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)

    # Initialize local logging
    run_id = f"EVAL-{cfg.task_suite_name}-{cfg.model_family}-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")

    # Initialize Weights & Biases logging as well
    if cfg.use_wandb:
        wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=run_id,
        )

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    print(f"Task suite: {cfg.task_suite_name}")
    log_file.write(f"Task suite: {cfg.task_suite_name}\n")

    # Get expected image dimensions
    resize_size = get_image_resize_size(cfg)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    finished = False
    while not finished:
        # Get task
        task_id = random.sample(range(num_tasks_in_suite), 1)[0]
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, original_task = get_libero_env(task, cfg.model_family, resolution=256)
        print(original_task)
        env.reset()

        # Set initial states
        obs = env.set_init_state(initial_states[0])

        # Setup
        t = 0
        replay_images = []
        max_steps = 10000
        task_description = None
        prev_state = None
        prev_action = None
        next_input = 0
        while t < max_steps + cfg.num_steps_wait:
            try:
                # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                # and we need to wait for them to fall
                if t < cfg.num_steps_wait:
                    obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                    if t == 0:
                        starting_state = env.get_sim_state()[1:10]
                        send_frame_to_server(obs, t)
                    t += 1
                    continue
                
                # Get preprocessed image
                img = get_libero_image(obs, resize_size)

                # Save preprocessed image for replay video
                replay_images.append(img)
                # Prepare observations dict
                # Note: OpenVLA does not take proprio state as input
                observation = {
                    "full_image": img,
                    "state": np.concatenate(
                        (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                    ),
                }
                if t > next_input:
                    task_description = input('instruction: \n')
                    if task_description == 'n':
                        break
                    if task_description == 'q':
                        finished = True
                        break
                    else:
                        num_steps = input('How many steps to run? \n')
                        next_input = int(num_steps) + t
                # Query model to get action
                if task_description == 'r': 
                    curr_state = env.get_sim_state()
                    curr_state[1:10] = starting_state
                    env.set_state(curr_state)
                    action = np.array([0, 0, 0, 0, 0, 0, -1])

                else:
                    action = get_action(
                        cfg,
                        model,
                        observation,
                        task_description,
                        processor=processor,
                    )
                    # Normalize gripper action [0,1] -> [-1,+1] because the environment expects the latter
                    action = normalize_gripper_action(action, binarize=True)

                    # [OpenVLA] The dataloader flips the sign of the gripper action to align with other datasets
                    # (0 = close, 1 = open), so flip it back (-1 = open, +1 = close) before executing the action
                    if cfg.model_family == "openvla":
                        action = invert_gripper_action(action)
                print('step', t)

                send_frame_to_server(obs, t)

                # Execute action in environment
                # reset logic

                obs, reward, done, info = env.step(action.tolist())
                if done:
                    break
                t += 1

            except DeprecationWarning as e:
                print(f"Caught exception: {e}")
                log_file.write(f"Caught exception: {e}\n")
                break

        # # Save a replay video of the episode
        # save_rollout_video(
        #     replay_images, total_episodes, success=done, task_description=task_description, log_file=log_file
        # )

        if cfg.use_wandb:
            wandb.log(
                {
                    f"success_rate/{task_description}": float(task_successes) / float(task_episodes),
                    f"num_episodes/{task_description}": task_episodes,
                }
            )

    # Save local log file
    log_file.close()

    # Push total metrics and local log file to wandb
    if cfg.use_wandb:
        wandb.log(
            {
                "success_rate/total": float(total_successes) / float(total_episodes),
                "num_episodes/total": total_episodes,
            }
        )
        wandb.save(local_log_filepath)


if __name__ == "__main__":
    eval_libero()
