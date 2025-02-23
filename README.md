# **OctoModel Attention Visualization**

## **Week 3**
### **1. Extracting Attention Maps**
- Modified `OctoModel.run_transformer()` to store **all attention weights** across **all layers and heads**.
- Extracted attention values from `states['intermediates']`, specifically from:
  ```python
  attention_weights['encoderblock_X']['MultiHeadDotProductAttention_0']['attention_weights']
  ```
- Ensured **correct indexing** for **all Transformer layers and heads**.

### **2. Attention Map Visualization**
- Created a function to **overlay attention heatmaps** onto original images.
- Scaled **attention tokens** to image size using OpenCV and SciPy.
- Displayed **attention per layer & head** and saved outputs in `attention_results/`.

## **Run the Code**

See model/Test_visualization.ipynb file for visualization.

## **Output Structure**
- `output/attention_results/`
  - `step_0_layer_0.png` → Attention at step 0, layer 0.
  - `step_0_layer_1.png` → Attention at step 0, layer 1.
  - `step_1_layer_0.png` → Attention at step 1, layer 0.
  - ... (One image per timestep and layer)
- `output/attention_video.mp4`

## **Next Steps**
- Analyze attention trends across different robotic tasks.
- 

---
