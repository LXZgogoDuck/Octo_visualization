# **OctoModel Attention Visualization**

## **📌 Project Overview**
This project extracts and visualizes **Transformer attention maps** from the **OctoModel**, a robotics-focused multimodal model. The goal is to understand **where the model focuses** when processing robotic tasks.

## **🚀 Features Implemented**
### **1. Extracting Attention Maps**
- Modified `OctoModel.run_transformer()` to store **all attention weights** across **all layers and heads**.
- Extracted attention values from `states['intermediates']`, specifically from:
  ```python
  attention_weights['encoderblock_X']['MultiHeadDotProductAttention_0']['attention_weights']
  ```
- Ensured **correct indexing** for **all Transformer layers and heads**.

### **2. Debugging Shape Issues**
- Initial shape issue: `attention_maps` had an **extra batch dimension** `(1, 12, 256)`.
- Fixed it using `[0]` to remove batch size, ensuring shape **(num_heads, image_token_num)**.

### **3. Attention Map Visualization**
- Created a function to **overlay attention heatmaps** onto original images.
- Scaled **attention tokens** to image size using OpenCV and SciPy.
- Displayed **attention per layer & head** and saved outputs in `attention_results/`.

## **🛠 How to Run the Code**
### **1️⃣ Install Dependencies**
```bash
pip install numpy matplotlib opencv-python scipy jax tensorflow_datasets
```

### **2️⃣ Load and Run OctoModel**
```python
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from octo.model.octo_model import OctoModel
model = OctoModel.load_pretrained("hf://rail-berkeley/octo-base-1.5")
```

### **3️⃣ Run Attention Visualization**
```python
visualize_attention_maps(attention_maps_per_step, images, save_path="attention_results")
```

## **📂 Output Structure**
- `attention_results/`
  - `step_0_layer_0.png` → Attention at step 0, layer 0.
  - `step_0_layer_1.png` → Attention at step 0, layer 1.
  - `step_1_layer_0.png` → Attention at step 1, layer 0.
  - ... (One image per timestep and layer)

## **📊 Next Steps**
- Convert visualizations into **GIFs or videos**.
- Improve clarity (e.g., **bounding boxes**, **labels**).
- Analyze attention trends across different robotic tasks.

---
