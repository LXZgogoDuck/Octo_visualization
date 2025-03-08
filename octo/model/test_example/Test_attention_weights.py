import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from octo.model.octo_model import OctoModel
model = OctoModel.load_pretrained("hf://rail-berkeley/octo-small-1.5")

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow_datasets as tfds
import cv2
import jax
from PIL import Image
import mediapy as mp
import tensorflow as tf
import tqdm

import imageio
from scipy.ndimage import zoom


### Load the BRIDGE Dataset
print("Loading BRIDGE dataset...")
builder = tfds.builder_from_directory(builder_dir="gs://gresearch/robotics/bridge/0.1.0/")
ds = builder.as_dataset(split="train[:1]")  # Load first episode

# Extract a single episode
episode = next(iter(ds))
steps = list(episode["steps"])
images = [cv2.resize(np.array(step["observation"]["image"]), (256, 256)) for step in steps]

# Extract goal image (last frame) & language instruction
goal_image = images[-1]
language_instruction = steps[0]["observation"]["natural_language_instruction"].numpy().decode()
print(f"Instruction: {language_instruction}")

for img in images:
    cv2.imshow("Episode Frame", img)
    cv2.waitKey(100)  # Wait 100ms per frame
cv2.destroyAllWindows()

### Load the Pretrained Octo Model
print("Loading Octo model checkpoint...")
model = OctoModel.load_pretrained("hf://rail-berkeley/octo-base-1.5")

WINDOW_SIZE = 2
task = model.create_tasks(goals={"image_primary": goal_image[None]})   # for goal-conditioned
task = model.create_tasks(texts=[language_instruction])                # for language conditioned

pred_actions, true_actions, attention_maps_per_step = [], [], []
for step in tqdm.trange(len(images) - (WINDOW_SIZE - 1)):
    input_images = np.stack(images[step:step+WINDOW_SIZE])[None]
    observation = {
        'image_primary': input_images,
        'timestep_pad_mask': np.full((1, input_images.shape[1]), True, dtype=bool)
    }

    # Get both predicted actions and attention maps
    actions, attention_maps = model.sample_actions(
        observation,
        task,
        unnormalization_statistics=model.dataset_statistics["bridge_dataset"]["action"],
        rng=jax.random.PRNGKey(0)
    )

    # Store predicted actions
    pred_actions.append(actions[0])

    # Store attention maps (for all layers & heads at this step)
    attention_maps_per_step.append(attention_maps)

    # Store true actions
    final_window_step = step + WINDOW_SIZE - 1
    true_actions.append(np.concatenate(
        (
            steps[final_window_step]['action']['world_vector'],
            steps[final_window_step]['action']['rotation_delta'],
            np.array(steps[final_window_step]['action']['open_gripper']).astype(np.float32)[None]
        ), axis=-1
    ))

for step in range(len(attention_maps_per_step)):
    for layer_idx in range(len(attention_maps_per_step[step]["all"])):  # Iterate over all layers
        attn_map = attention_maps_per_step[step]["all"][layer_idx]
        print(f"Step {step} - Layer {layer_idx} Attention Map Shape: {attn_map.shape}")
        print(
            f"Step {step} - Layer {layer_idx} Sample Attention Values: {attn_map[:2, :5]}")  # Print first 2 heads, first 5 tokens
