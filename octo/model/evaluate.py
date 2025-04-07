import argparse
import numpy as np
import os
import tensorflow as tf
import json
import pickle
import time
import simpler_env
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
import sapien.core as sapien
import mediapy
import gymnasium as gym
from visualize_utils import visualize_attention_episodes, generate_videos_for_heads

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
gpus = tf.config.list_physical_devices("GPU")
if len(gpus) > 0:
    tf.config.set_logical_device_configuration(
        gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=3072)]
    )

RT_1_CHECKPOINTS = {
    "rt_1_x": "rt_1_x_tf_trained_for_002272480_step",
    "rt_1_400k": "rt_1_tf_trained_for_000400120",
    "rt_1_58k": "rt_1_tf_trained_for_000058240",
    "rt_1_1k": "rt_1_tf_trained_for_000001120",
}

def get_rt_1_checkpoint(name, ckpt_dir="/user/hypervla/finetune_saves/rtx"):
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

def load_model(policy_name, ckpt_path, policy_setup, input_rng=0, step=None, image_horizon=2):
    if "rt_1" in policy_name:
        from simpler_env.policies.rt1.rt1_model import RT1Inference
        ckpt_path = get_rt_1_checkpoint(policy_name)
        model = RT1Inference(saved_model_path=ckpt_path, policy_setup=policy_setup)
    elif "octo" in policy_name:
        from octo_inference import OctoInference
        from octo.model.octo_model import OctoModel
        if os.path.exists(ckpt_path):
            temp_model = OctoModel.load_pretrained(ckpt_path, step=step)
            model = OctoInference(model=temp_model, policy_setup=policy_setup, init_rng=input_rng, horizon=image_horizon)
        else:
            model = OctoInference(model_type=ckpt_path, policy_setup=policy_setup, init_rng=input_rng, horizon=image_horizon)
    else:
        raise NotImplementedError(f"Unknown policy type: {policy_name}")
    return model

def evaluate(policy_name, ckpt_path, tasks, seed=0, checkpoint_step=None, image_horizon=2, save_video=False, save_trajectory=False, recompute=False):
    if ckpt_path in [None, "None"]:
        ckpt_path = policy_name
    if ckpt_path[-1] == "/":
        ckpt_path = ckpt_path[:-1]

    if not os.path.exists(ckpt_path):
        eval_path = f'eval_results/google_robot/{policy_name}/{seed}'
    else:
        save_dir = ckpt_path.replace('finetune_saves', 'eval_results')
        eval_path = f'{save_dir}/eval_step_{checkpoint_step}/{seed}'
    os.makedirs(eval_path, exist_ok=True)

    save_file_name = f'success_rate_horizon_{image_horizon}'
    if os.path.exists(f'{eval_path}/{save_file_name}.json'):
        with open(f'{eval_path}/{save_file_name}.json', 'r') as f:
            all_tasks_success_rate = json.load(f)
    else:
        all_tasks_success_rate = dict()

    previous_policy_setup = ''
    for task_name in tasks:
        if not recompute and not save_video:
            if task_name in all_tasks_success_rate and len(all_tasks_success_rate[task_name][1]) == tasks[task_name][1]:
                continue

        video_path = f"{eval_path}/video/{task_name}"
        os.makedirs(video_path, exist_ok=True)
        if not recompute and len(os.listdir(video_path)) >= 10:
            continue

        policy_setup = "google_robot" if "google" in task_name else "widowx_bridge"

        if policy_setup != previous_policy_setup:
            model = load_model(
                policy_name,
                ckpt_path,
                policy_setup,
                input_rng=seed,
                step=checkpoint_step,
                image_horizon=image_horizon,
            )
        previous_policy_setup = policy_setup

        if 'env' in locals():
            env.close()
            del env

        env_name, total_runs, options = tasks[task_name]
        env = gym.make(env_name, obs_mode="rgbd", prepackaged_config=True) if env_name else simpler_env.make(task_name)

        if save_video:
            total_runs = 50

        sapien.render_config.rt_use_denoiser = False
        print(f'===== {task_name} =====')

        success_count = 0
        episode_results = []

        for run in range(total_runs):
            obs, _ = env.reset(options=options[run] if options else None)
            instruction = env.get_language_instruction()
            model.reset(instruction)
            image = get_image_from_maniskill2_obs_dict(env, obs)

            images, action_sequence = [], []
            attention_maps_per_step = []
            success = False
            truncated = False

            while not (truncated or success):
                result = model.step(image)
                if len(result) == 3:
                    raw_action, action, attention_maps = result
                    attention_maps_per_step.append(attention_maps)
                else:
                    raw_action, action = result

                action = np.concatenate([action["world_vector"], action["rot_axangle"], action["gripper"]])
                images.append(image.squeeze())
                action_sequence.append(raw_action)

                obs, reward, success, truncated, _ = env.step(action)
                image = get_image_from_maniskill2_obs_dict(env, obs)

            success_count += int(success)
            episode_results.append(success)
            print(f"Episode {run+1}: {'Success' if success else 'Fail'}")

            if save_video:
                mediapy.write_video(f'{video_path}/{run + 1}_success_{success}.mp4', images, fps=10)
                attention_output_dir = f'{video_path}/attention_episode_{run + 1}_success_{success}'
                visualize_attention_episodes([attention_maps_per_step], [images], attention_output_dir, last_n_layers=12)
                generate_videos_for_heads(attention_output_dir)

            if save_trajectory:
                traj_path = f"{eval_path}/trajectory/{task_name}"
                os.makedirs(traj_path, exist_ok=True)
                with open(f"{traj_path}/{run}_{instruction}_success_{success}.pkl", "wb") as f:
                    pickle.dump([images, action_sequence], f)

        env.close()
        all_tasks_success_rate[task_name] = [success_count / total_runs, episode_results]
        print(f"Success Rate Summary: {all_tasks_success_rate}")

        if not save_video:
            with open(f'{eval_path}/{save_file_name}.json', 'w') as f:
                json.dump(all_tasks_success_rate, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="octo-base", choices=["octo-base", "octo-small", "rt_1_x", "rt_1_400k"])
    parser.add_argument("--ckpt-path", type=str, default=None)
    parser.add_argument("--seeds", type=str, default='0')
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--save_video", action='store_true')
    parser.add_argument("--save_trajectory", action='store_true')
    parser.add_argument("--recompute", action='store_true')
    parser.add_argument("--window_size", type=int, default=2)
    args = parser.parse_args()

    if args.policy in ["octo-base", "octo-small"]:
        if args.ckpt_path in [None, "None"] or "rt_1_x" in str(args.ckpt_path):
            args.ckpt_path = args.policy
    if args.ckpt_path and args.ckpt_path.endswith("/"):
        args.ckpt_path = args.ckpt_path[:-1]

    move_task_options = [{"obj_init_options": {"episode_id": i}} for i in range(60)]
    tasks = {
        "google_robot_close_middle_drawer": (None, 50, None),
        "google_robot_close_bottom_drawer": (None, 50, None),
        "google_robot_open_top_drawer": (None, 20, None),
        "google_robot_open_middle_drawer": (None, 20, None),
        "google_robot_open_bottom_drawer": (None, 20, None),
        "google_robot_place_apple_in_closed_top_drawer": (None, 10, None),
        "widowx_spoon_on_towel": (None, 20, None)
    }

    seeds = [int(seed) for seed in args.seeds.split('+')]
    for seed in seeds:
        evaluate(
            args.policy,
            args.ckpt_path,
            tasks,
            seed=seed,
            checkpoint_step=args.step,
            image_horizon=args.window_size,
            save_video=args.save_video,
            save_trajectory=args.save_trajectory,
            recompute=args.recompute,
        )
