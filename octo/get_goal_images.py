import argparse
import numpy as np
import os
import tensorflow as tf
import json
import pickle
import jax
import time

import simpler_env
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
import sapien.core as sapien
import jax.numpy as jnp

import mediapy
import gymnasium as gym
import matplotlib.pyplot as plt

# prevent a single jax process from taking up all the GPU memory
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
gpus = tf.config.list_physical_devices("GPU")
if len(gpus) > 0:
    # prevent a single tf process from taking up all the GPU memory
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=3072)],
    )

RT_1_CHECKPOINTS = {
    "rt_1_x": "rt_1_x_tf_trained_for_002272480_step",
    "rt_1_400k": "rt_1_tf_trained_for_000400120",
    "rt_1_58k": "rt_1_tf_trained_for_000058240",
    "rt_1_1k": "rt_1_tf_trained_for_000001120",
}


def get_rt_1_checkpoint(name, ckpt_dir="/user/octo/finetune_saves/rtx"):
    assert name in RT_1_CHECKPOINTS, name
    ckpt_name = RT_1_CHECKPOINTS[name]
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    if not os.path.exists(ckpt_path):
        if name == "rt_1_x":
            os.system(f'gsutil -m cp -r gs://gdm-robotics-open-x-embodiment/open_x_embodiment_and_rt_x_oss/{ckpt_name}.zip {ckpt_dir}')
            os.system(f'unzip {ckpt_dir}/{ckpt_name}.zip -d {ckpt_dir}')
        else:
            os.system(f'gsutil -m cp -r gs://gdm-robotics-open-x-embodiment/open_x_embodiment_and_rt_x_oss/{ckpt_name} {ckpt_dir}')
    return ckpt_path


def load_model(model_name, model_path, base_model, policy_setup, input_rng=0, step=None, action_ensemble=False, action_horizon=4, image_horizon=2, crop=False, save_attention_map=False, image_size=256, padded_resize=False):
    if "rt_1" in model_name:
        from simpler_env.policies.rt1.rt1_model import RT1Inference
        ckpt_path = get_rt_1_checkpoint(model_name)
        model = RT1Inference(saved_model_path=ckpt_path, policy_setup=policy_setup)
    return model

def evaluate(model_name, model_path, base_model, tasks, seed=0, checkpoint_step=None, action_ensemble=True, image_horizon=2, save_video=False, save_trajectory=False, recompute=False, crop=False, save_attention_map=False, EMA_coefficient=None):
    assert model_name in ["rt_1_x", "rt_1_400k", "rt_1_58k", "rt_1_1k"], "This evaluation is restricted to RT-X policies only."

    previous_policy_setup = ''
    eval_path = f'/user/octo/eval_results/rtx_goal_data/{model_name}/{seed}'
    os.makedirs(eval_path, exist_ok=True)

    # Load existing info if available
    success_episode_info_path = os.path.join(eval_path, "success_episode_info.json")
    if os.path.exists(success_episode_info_path):
        with open(success_episode_info_path, 'r') as f:
            success_episode_info = json.load(f)
    else:
        success_episode_info = {}

    # Initialize failure_episode_info dictionary to keep track of failures
    failure_episode_info = {}

    for task_name in tasks:
        print(f"\n===== {task_name} =====")
        video_path = os.path.join(eval_path, "video", task_name)
        os.makedirs(video_path, exist_ok=True)

        if not recompute and task_name in success_episode_info:
            print(f"Skipping {task_name}, already computed. Use --recompute to overwrite.")
            continue

        if "google" in task_name:
            policy_setup = "google_robot"
        else:
            policy_setup = "widowx_bridge"

        if policy_setup != previous_policy_setup:
            model = load_model(
                model_name,
                model_path,
                base_model,
                policy_setup,
                seed,
                step=checkpoint_step,
                action_ensemble=action_ensemble,
                action_horizon=4,
                image_horizon=image_horizon,
                crop=crop,
                save_attention_map=save_attention_map,
                image_size=256,
                padded_resize=False,
            )
        previous_policy_setup = policy_setup

        if 'env' in locals():
            print("Closing existing env")
            env.close()
            del env

        env_name, total_runs, options = tasks[task_name]
        if options is not None:
            total_runs = min(total_runs, len(options))
            print(f"Total runs: {total_runs}, Available reset options: {len(options)}")

        env = simpler_env.make(task_name)

        obs, _ = env.reset(seed=seed)
        success_runs = []
        failure_runs = []  # List to track failure runs
        goal_images_dict = {}

        for run in range(total_runs):
            if options is not None:
                obs, reset_info = env.reset(options=options[run])
            else:
                obs, reset_info = env.reset()

            instruction = env.get_language_instruction()
            image = get_image_from_maniskill2_obs_dict(env, obs)

            model.reset(instruction)

            images = []
            success, truncated = False, False

            while not (truncated or success):
                raw_action, action = model.step(image)
                action = np.concatenate([action["world_vector"], action["rot_axangle"], action["gripper"]])
                obs, reward, success, truncated, info = env.step(action)
                image = get_image_from_maniskill2_obs_dict(env, obs)
                images.append(image.copy())

            if success:
                print(f"Run {run}: SUCCESS")
                success_runs.append(run)
                goal_images = images[-3:] if len(images) >= 3 else images
                goal_images_dict[str(run)] = [img.tolist() for img in goal_images]

                if save_video:
                    mediapy.write_video(f'{video_path}/run_{run}_success.mp4', images, fps=10)

            else:
                print(f"Run {run}: FAIL")
                failure_runs.append(run)  # Track failure runs

        env.close()

        # Save both success and failure information
        success_episode_info[task_name] = {
            "success_runs": success_runs,
            "goal_images": goal_images_dict
        }
        failure_episode_info[task_name] = {
            "failure_runs": failure_runs
        }

        with open(success_episode_info_path, 'w') as f:
            json.dump(success_episode_info, f, indent=2)

        # Save failure info as a separate file
        failure_episode_info_path = os.path.join(eval_path, "failure_episode_info.json")
        with open(failure_episode_info_path, 'w') as f:
            json.dump(failure_episode_info, f, indent=2)

    print("Evaluation complete. Success and failure run info saved.")


if __name__ == '__main__':

    # Add arguments
    parser = argparse.ArgumentParser(description="A simple example of argparse")
    parser.add_argument("--model", choices=["octo-base", "rt_1_x", "rt_1_400k", "openvla", "hypervla", "base_net"], default="rt_1_x", help="The model used for evaluation")
    parser.add_argument("--model_path", type=str, default='', help="The path of the custom model (only useful for octo-custom?)")
    parser.add_argument("--seeds", type=str, default='0+1+2+3', help="seeds for policy and env")
    parser.add_argument("--step", type=int, default=None, help="checkpoint step to evaluate")
    parser.add_argument("--action_ensemble", action='store_true', help="use action ensemble or not")
    parser.add_argument("--save_video", action='store_true', help="save evaluation video or not")
    parser.add_argument("--save_trajectory", action='store_true', help="save eval trajectory or not")
    parser.add_argument("--recompute", action='store_true', help="whether to overwrite existing eval results")
    parser.add_argument("--window_size", type=int, default=2, help="window size of historical observations")
    parser.add_argument("--crop", action='store_true', help="whether to crop the resized image or not")
    parser.add_argument("--save_attention_map", action='store_true', help="whether to save attention map of DINOv2 or not")
    parser.add_argument("--EMA", type=float, default=None, help="evaluate with EMA of model parameters during training")
    # Parse the arguments
    args = parser.parse_args()

    # define reproducible reset options
    pick_object_options = [{"obj_init_options": {"episode_id": i}} for i in range(200)]
    pick_coke_can_options = [{"obj_init_options": {"episode_id": i}} for i in range(200)]
    move_near_options = [{"obj_init_options": {"episode_id": i}} for i in range(60)]
    drawer_task_options = [{"obj_init_options": {"episode_id": i}} for i in range(150)]
    # drawer_middle_options = [{"obj_init_options": {"episode_id": i}} for i in range(20)]
    # drawer_bottom_options = [{"obj_init_options": {"episode_id": i}} for i in range(20)]
    widowx_task_options = [{"obj_init_options": {"episode_id": i}} for i in range(200)]
    # widowx_carrot_options = [{"obj_init_options": {"episode_id": i}} for i in range(20)]
    # widowx_stack_options = [{"obj_init_options": {"episode_id": i}} for i in range(20)]
    # widowx_eggplant_options = [{"obj_init_options": {"episode_id": i}} for i in range(20)]

    tasks = {
        # "google_robot_pick_object": (None, 200, pick_object_options),
        # "google_robot_pick_coke_can": (None, 200, pick_coke_can_options),
        # "google_robot_close_top_drawer": (None, 150, drawer_task_options),
        # "google_robot_close_middle_drawer": (None, 20, drawer_task_options),
        #"google_robot_close_bottom_drawer": (None, 150, drawer_task_options),
        # "google_robot_move_near": (None, len(move_near_options), move_near_options),
        "widowx_spoon_on_towel": (None, 200, widowx_task_options),
        "widowx_carrot_on_plate": (None, 200, widowx_task_options),
        # "widowx_stack_cube": (None, 80, widowx_task_options),
        # "widowx_put_eggplant_in_basket": (None, 80, widowx_task_options),
    }

    base_model = None

    seeds = [eval(seed) for seed in args.seeds.split('+')]
    for seed in seeds:
        evaluate(args.model, args.model_path, base_model, tasks, seed=seed, checkpoint_step=args.step, action_ensemble=args.action_ensemble, save_video=args.save_video, save_trajectory=args.save_trajectory, recompute=args.recompute, image_horizon=args.window_size, crop=args.crop, save_attention_map=args.save_attention_map, EMA_coefficient=args.EMA)
