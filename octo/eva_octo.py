import argparse
import os
import json
import numpy as np
import tensorflow as tf
import jax
import simpler_env
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
import mediapy

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
gpus = tf.config.list_physical_devices("GPU")
if gpus:
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

def get_rt_1_checkpoint(name, ckpt_dir="/workspace/Octo_visualization/finetune_saves/rtx"):
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
        return RT1Inference(saved_model_path=ckpt_path, policy_setup=policy_setup)
    else:
        from octo.model.octo_inference import OctoInference
        from octo.model.octo_model import OctoModel
        if os.path.exists(model_path):
            temp_model = OctoModel.load_pretrained(model_path, step=step)
            model = OctoInference(model=temp_model, policy_setup=policy_setup, init_rng=input_rng,
                                  horizon=image_horizon)
        else:
            model = OctoInference(model_type=model_name, policy_setup=policy_setup, init_rng=input_rng,
                                  horizon=image_horizon)
        return model

def evaluate_rt1(model_name, model_path, base_model, tasks, seed=0, checkpoint_step=None, image_horizon=2, crop=False, save_video=False, recompute=False):
    eval_path = f'/mnt/auriga/vis24xl/eval_results/rtx_goal_data/{model_name}/{seed}'
    os.makedirs(eval_path, exist_ok=True)
    success_episode_info_path = os.path.join(eval_path, "success_episode_info.json")

    success_episode_info = {}

    for task_name in tasks:
        print(f"\n===== Evaluating RT-1 on {task_name} =====")
        if not recompute and os.path.exists(success_episode_info_path):
            with open(success_episode_info_path, "r") as f:
                existing = json.load(f)
            if task_name in existing:
                print(f"Skipping {task_name}, already computed.")
                continue

        policy_setup = "google_robot" if "google" in task_name else "widowx_bridge"
        model = load_model(model_name, model_path, base_model, policy_setup, seed, checkpoint_step, image_horizon=image_horizon, crop=crop)

        env_name, total_runs, options = tasks[task_name]
        env = simpler_env.make(task_name)
        obs, _ = env.reset(seed=seed)
        success_runs, goal_images_dict = [], {}

        for run in range(total_runs):
            obs, _ = env.reset(options=options[run]) if options else env.reset()
            instruction = env.get_language_instruction()
            image = get_image_from_maniskill2_obs_dict(env, obs)
            model.reset(instruction)

            images, success = [], False
            while not success:
                _, action = model.step(image)
                action = np.concatenate([action["world_vector"], action["rot_axangle"], action["gripper"]])
                obs, _, success, truncated, _ = env.step(action)
                image = get_image_from_maniskill2_obs_dict(env, obs)
                images.append(image.copy())
                if truncated:
                    break

            if success:
                print(f"RT-1 Run {run}: SUCCESS")
                success_runs.append(run)
                goal_images_dict[str(run)] = [img.tolist() for img in images[-3:]]
                if save_video:
                    video_path = os.path.join(eval_path, "video", task_name)
                    os.makedirs(video_path, exist_ok=True)
                    mediapy.write_video(f"{video_path}/run_{run}_success.mp4", images, fps=10)
            else:
                print(f"RT-1 Run {run}: FAIL")

        env.close()
        success_episode_info[task_name] = {
            "success_runs": success_runs,
            "goal_images": goal_images_dict
        }

        with open(success_episode_info_path, 'w') as f:
            json.dump(success_episode_info, f, indent=2)

def evaluate_octo_on_success_runs_image_goal_only(
    model_name,
    model_path,
    base_model,
    tasks,
    seed=0,
    checkpoint_step=None,
    image_horizon=2,
    crop=False,
    save_video=False,
    use_last_n_goal_images=3,
):
    # eval_path = f'/mnt/leo/vis24xl/eval_results/rtx_goal_data/{model_name}/{seed}'
    # rt1_success_path = f'/mnt/leo/vis24xl/eval_results/rtx_goal_data/rt_1_x/{seed}/success_episode_info.json'
    eval_path = f'/user/octo/eval_results/rtx_goal_data_2/{model_name}/{seed}'
    rt1_success_path = f'/user/octo/eval_results/rtx_goal_data_2/rt_1_x/{seed}/success_episode_info.json'

    with open(rt1_success_path, 'r') as f:
        rt_success_info = json.load(f)

    total_eval, total_success = 0, 0

    for task_name in tasks:
        if task_name not in rt_success_info:
            continue

        print(f"\n=== Octo eval (image-goal-only) on {task_name} using RT-1 successes ===")
        success_runs = rt_success_info[task_name]["success_runs"]
        goal_images_dict = rt_success_info[task_name]["goal_images"]

        policy_setup = "google_robot" if "google" in task_name else "widowx_bridge"

        # Use OctoInference directly to access reset(goal_image=...)
        model = load_model(model_name, model_path, base_model, policy_setup, seed, checkpoint_step, image_horizon=image_horizon, crop=crop)


        env_name, total_runs, options = tasks[task_name]
        env = simpler_env.make(task_name)
        obs, _ = env.reset(seed=seed)

        for run in range(total_runs):
            if run not in success_runs:
                continue

            goal_imgs_np = np.array(goal_images_dict[str(run)], dtype=np.uint8)
            found_success = False
            for goal_idx in range(min(use_last_n_goal_images, len(goal_imgs_np))):
                goal_image = goal_imgs_np[goal_idx]  # Try last, then second last, etc.
                obs, _ = env.reset(options=options[run]) if options else env.reset() #
                image = get_image_from_maniskill2_obs_dict(env, obs)

                model.reset(task_description=None, goal_image=goal_image)

                images = []
                success = False
                while not success:
                    _, action, _ = model.step(image, goal_image=goal_image)
                    action = np.concatenate([action["world_vector"], action["rot_axangle"], action["gripper"]])
                    obs, _, success, truncated, _ = env.step(action)
                    image = get_image_from_maniskill2_obs_dict(env, obs)
                    images.append(image.copy())
                    if truncated:
                        break

                total_eval += 1
                if success:
                    total_success += 1
                    found_success = True
                    print(f"Octo Run {run} (goal image idx: {goal_idx}): SUCCESS")
                    if save_video:
                        video_path = os.path.join(eval_path, "video", f"octo_imggoal_{task_name}")
                        os.makedirs(video_path, exist_ok=True)
                        mediapy.write_video(f'{video_path}/run_{run}_imggoal_success.mp4', images, fps=10)
                else:
                    print(f"Octo Run {run} (goal image idx: {goal_idx}): FAIL")

        env.close()

    print("\n=== Octo (image-goal-only) Evaluation Summary ===")
    print(f"Evaluated: {total_eval}, Successes: {total_success}, Image Goal Success Rate: {100 * total_success / total_eval:.2f}%" if total_eval else "No episodes evaluated.")
    # Save evaluation summary
    summary_path = os.path.join(eval_path, "image_goal_success.json")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)

    with open(summary_path, "w") as f:
        json.dump({
            "evaluated": total_eval,
            "successes": total_success,
            "success_rate": float(total_success) / total_eval if total_eval else 0.0
        }, f, indent=2)


def evaluate_octo_on_success_runs_lang_goal_only(model_name, model_path, base_model, tasks, seed=0, checkpoint_step=None, image_horizon=2, crop=False, save_video=False):
    eval_path = f'/user/octo/eval_results/rtx_goal_data_2/{model_name}/{seed}'
    rt1_success_path = f'/user/octo/eval_results/rtx_goal_data_2/rt_1_x/{seed}/success_episode_info.json'

    with open(rt1_success_path, 'r') as f:
        rt_success_info = json.load(f)

    total_eval, total_success = 0, 0

    for task_name in tasks:
        if task_name not in rt_success_info:
            continue

        print(f"\n=== Octo eval on {task_name} using RT-1 successes ===")
        success_runs = rt_success_info[task_name]["success_runs"]
        policy_setup = "google_robot" if "google" in task_name else "widowx_bridge"
        model = load_model(model_name, model_path, base_model, policy_setup, seed, checkpoint_step, image_horizon=image_horizon, crop=crop)

        env_name, total_runs, options = tasks[task_name]
        env = simpler_env.make(task_name)
        obs, _ = env.reset(seed=seed)

        for run in range(total_runs):
            if run not in success_runs:
                continue

            obs, _ = env.reset(options=options[run]) if options else env.reset()
            instruction = env.get_language_instruction()
            image = get_image_from_maniskill2_obs_dict(env, obs)
            model.reset(instruction) # for simpler_env given policy inference, this is only for language goal.

            images, success = [], False
            while not success:
                _, action, _ = model.step(image, task_description=instruction)
                action = np.concatenate([action["world_vector"], action["rot_axangle"], action["gripper"]])
                obs, _, success, truncated, _ = env.step(action)
                image = get_image_from_maniskill2_obs_dict(env, obs)
                images.append(image.copy())
                if truncated:
                    break

            total_eval += 1
            if success:
                total_success += 1
                print(f"Octo Run {run}: SUCCESS")
                if save_video:
                    video_path = os.path.join(eval_path, "video", f"octo_{task_name}")
                    os.makedirs(video_path, exist_ok=True)
                    mediapy.write_video(f'{video_path}/run_{run}_success.mp4', images, fps=10)
            else:
                print(f"Octo Run {run}: FAIL")

        env.close()

    print("\n=== Octo Evaluation Summary ===")
    print(f"Evaluated: {total_eval}, Successes: {total_success}, Language Goal Success Rate: {100 * total_success / total_eval:.2f}%" if total_eval else "No episodes evaluated.")
    # Save evaluation summary
    summary_path = os.path.join(eval_path, "language_goal_success.json")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump({
            "evaluated": total_eval,
            "successes": total_success,
            "success_rate": float(total_success) / total_eval if total_eval else 0.0
        }, f, indent=2)

def evaluate_octo_on_both(
    model_name,
    model_path,
    base_model,
    tasks,
    seed=0,
    checkpoint_step=None,
    image_horizon=2,
    crop=False,
    save_video=False,
    use_last_n_goal_images=3,
):
    eval_path = f'/user/octo/eval_results/rtx_goal_data/{model_name}/{seed}'
    rt1_success_path = f'/user/octo/eval_results/rtx_goal_data/rt_1_x/{seed}/success_episode_info.json'

    with open(rt1_success_path, 'r') as f:
        rt_success_info = json.load(f)

    total_eval, total_success = 0, 0

    for task_name in tasks:
        if task_name not in rt_success_info:
            continue

        print(f"\n=== Octo eval on {task_name} using RT-1 successes ===")
        success_runs = rt_success_info[task_name]["success_runs"]
        goal_images_dict = rt_success_info[task_name]["goal_images"]

        policy_setup = "google_robot" if "google" in task_name else "widowx_bridge"

        # Use OctoInference directly to access reset(goal_image=...)
        model = load_model(model_name, model_path, base_model, policy_setup, seed, checkpoint_step, image_horizon=image_horizon, crop=crop)

        env_name, total_runs, options = tasks[task_name]
        env = simpler_env.make(task_name)
        obs, _ = env.reset(seed=seed)

        for run in range(total_runs):
            if run not in success_runs:
                continue

            goal_imgs_np = np.array(goal_images_dict[str(run)], dtype=np.uint8)
            found_success = False
            instruction = env.get_language_instruction()

            for goal_idx in range(min(use_last_n_goal_images, len(goal_imgs_np))):
                goal_image = goal_imgs_np[-(goal_idx + 1)]  # Try last, then second last, etc.
                obs, _ = env.reset(options=options[run]) if options else env.reset()
                image = get_image_from_maniskill2_obs_dict(env, obs)
                model.reset(task_description=instruction, goal_image=goal_image)

                images = []
                success = False
                while not success:
                    _, action, _ = model.step(image, goal_image=goal_image)
                    action = np.concatenate([action["world_vector"], action["rot_axangle"], action["gripper"]])
                    obs, _, success, truncated, _ = env.step(action)
                    image = get_image_from_maniskill2_obs_dict(env, obs)
                    images.append(image.copy())
                    if truncated:
                        break

                total_eval += 1
                if success:
                    total_success += 1
                    found_success = True
                    print(f"Octo Run {run} (goal image idx: {goal_idx}): SUCCESS")
                    if save_video:
                        video_path = os.path.join(eval_path, "video", f"octo_imggoal_{task_name}")
                        os.makedirs(video_path, exist_ok=True)
                        mediapy.write_video(f'{video_path}/run_{run}_imggoal_success.mp4', images, fps=10)
                    break
                else:
                    print(f"Octo Run {run} (goal image idx: {goal_idx}): FAIL")

        env.close()

    print("\n=== Octo Evaluation Summary ===")
    print(f"Evaluated: {total_eval}, Successes: {total_success}, Image Goal Success Rate: {100 * total_success / total_eval:.2f}%" if total_eval else "No episodes evaluated.")
    summary_path = os.path.join(eval_path, "image_goal_success.json")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)

    with open(summary_path, "w") as f:
        json.dump({
            "evaluated": total_eval,
            "successes": total_success,
            "success_rate": float(total_success) / total_eval if total_eval else 0.0
        }, f, indent=2)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model_path", type=str, default='')
    parser.add_argument("--seeds", type=str, default='0')
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--save_video", action="store_true")
    parser.add_argument("--recompute", action="store_true")
    parser.add_argument("--window_size", type=int, default=2)
    parser.add_argument("--crop", action="store_true")
    args = parser.parse_args()

    pick_object_options = [{"obj_init_options": {"episode_id": i}} for i in range(200)]
    pick_coke_can_options = [{"obj_init_options": {"episode_id": i}} for i in range(50)]
    widowx_task_options = [{"obj_init_options": {"episode_id": i}} for i in range(200)]

    tasks = {
        # "google_robot_pick_object": (None, 50, pick_object_options),
        "google_robot_pick_coke_can": (None, 50, pick_coke_can_options),
        #  "widowx_spoon_on_towel": (None, 200, widowx_task_options),
        # "widowx_carrot_on_plate": (None, 200, widowx_task_options),
    }

    base_model = None
    seeds = [int(s) for s in args.seeds.split('+')]

    for seed in seeds:
       # evaluate_rt1("rt_x", args.model_path, base_model, tasks, seed=seed, checkpoint_step=args.step, image_horizon=args.window_size, crop=args.crop, save_video=args.save_video, recompute=args.recompute)
       #  evaluate_octo_on_success_runs_lang_goal_only(args.model, args.model_path, base_model, tasks, seed=seed, checkpoint_step=args.step, image_horizon=args.window_size, crop=args.crop, save_video=args.save_video)
       #  evaluate_octo_on_success_runs_image_goal_only(
       #          args.model,
       #          args.model_path,
       #          base_model,
       #          tasks,
       #          seed=seed,
       #          checkpoint_step=args.step,
       #          image_horizon=args.window_size,
       #          crop=args.crop,
       #          save_video=args.save_video,
       #          use_last_n_goal_images=3,
       #      )

        evaluate_octo_on_both(
                args.model,
                args.model_path,
                base_model,
                tasks,
                seed=seed,
                checkpoint_step=args.step,
                image_horizon=args.window_size,
                crop=args.crop,
                save_video=args.save_video,
                use_last_n_goal_images=5)
