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

## Docker installation
1. Edit the docker name, i.e., the `TAG` variable in `docker/build_docker.sh`. It is usually in the format of `your_user_name/docker_name`.
2. Under the root directory of the repo, run `bash docker/build_docker.sh`.
3. Start the docker with command `docker run --gpus '"device=0"' --rm --network host --ipc=host --user $(id -u) -v source_folder_path:/user/hypervla -it docker_name /bin/bash`

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

## **CNN Visualization** 
### To-do:
1. need to know the structure of the whole token_matrix (attention_weights)
2. extract (task) language tokens & image tokens separately & visualize images’ attention map attending to tasks
3. In tokenization process, how task tokens evolved into encoding? How FiLM layers are used? What features image tokens encode? 
4. How does ViT actually process images?
5. Could we simply don’t embed irrelevant features like background/edges… in tokenization part?
6. CNN layers visualizations (only 3 CNN layers? Any advantages?): Which method to choose and what information can it give us?

## Convolutional Neural Network Filter Visualization
CNN filters can be visualized when we optimize the input image with respect to output of the specific convolution operation. For this example I used a pre-trained **VGG16**. Visualizations of layers start with basic color and direction filters at lower levels. As we approach towards the final layer the complexity of the filters also increase. If you employ external techniques like blurring, gradient clipping etc. you will probably produce better images.

<table border=0 align=center>
	<tbody> 
		<tr>
			<td width="19%" align="center"> Layer 2 <br /> (Conv 1-2)</td>
			<td width="27%" align="center"> <img src="https://raw.githubusercontent.com/utkuozbulak/pytorch-cnn-visualizations/master/results/layer_visualizations/layer_vis_l2_f1.jpg"> </td>
			<td width="27%" align="center"> <img src="https://raw.githubusercontent.com/utkuozbulak/pytorch-cnn-visualizations/master/results/layer_visualizations/layer_vis_l2_f21.jpg"> </td>
			<td width="27%" align="center"> <img src="https://raw.githubusercontent.com/utkuozbulak/pytorch-cnn-visualizations/master/results/layer_visualizations/layer_vis_l2_f54.jpg"> </td>
		</tr>
		<tr>
			<td width="19%" align="center"> Layer 10 <br /> (Conv 2-1)</td>
			<td width="27%" align="center"> <img src="https://raw.githubusercontent.com/utkuozbulak/pytorch-cnn-visualizations/master/results/layer_visualizations/layer_vis_l10_f7.jpg"> </td>
			<td width="27%" align="center"> <img src="https://raw.githubusercontent.com/utkuozbulak/pytorch-cnn-visualizations/master/results/layer_visualizations/layer_vis_l10_f10.jpg"> </td>
			<td width="27%" align="center"> <img src="https://raw.githubusercontent.com/utkuozbulak/pytorch-cnn-visualizations/master/results/layer_visualizations/layer_vis_l10_f69.jpg"> </td>
		</tr>
		<tr>
			<td width="19%" align="center"> Layer 17 <br /> (Conv 3-1)</td>
			<td width="27%" align="center"> <img src="https://raw.githubusercontent.com/utkuozbulak/pytorch-cnn-visualizations/master/results/layer_visualizations/layer_vis_l17_f4.jpg"> </td>
			<td width="27%" align="center"> <img src="https://raw.githubusercontent.com/utkuozbulak/pytorch-cnn-visualizations/master/results/layer_visualizations/layer_vis_l17_f8.jpg"> </td>
			<td width="27%" align="center"> <img src="https://raw.githubusercontent.com/utkuozbulak/pytorch-cnn-visualizations/master/results/layer_visualizations/layer_vis_l17_f9.jpg"> </td>
		</tr>
		<tr>
			<td width="19%" align="center"> Layer 24 <br /> (Conv 4-1)</td>
			<td width="27%" align="center"> <img src="https://raw.githubusercontent.com/utkuozbulak/pytorch-cnn-visualizations/master/results/layer_visualizations/layer_vis_l24_f4.jpg"> </td>
			<td width="27%" align="center"> <img src="https://raw.githubusercontent.com/utkuozbulak/pytorch-cnn-visualizations/master/results/layer_visualizations/layer_vis_l24_f17.jpg"> </td>
			<td width="27%" align="center"> <img src="https://raw.githubusercontent.com/utkuozbulak/pytorch-cnn-visualizations/master/results/layer_visualizations/layer_vis_l24_f22.jpg"> </td>
		</tr>
	</tbody>
</table>

Another way to visualize CNN layers is to to visualize activations for a specific input on a specific layer and filter. This was done in [1] Figure 3. Below example is obtained from layers/filters of VGG16 for the first image using guided backpropagation. The code for this opeations is in *layer_activation_with_guided_backprop.py*. The method is quite similar to guided backpropagation but instead of guiding the signal from the last layer and a specific target, it guides the signal from a specific layer and filter. 

<table border=0 align=center>
	<tbody> 
    <tr>		<td width="27%" align="center"> Input Image </td>
			<td width="27%" align="center"> Layer Vis. (Filter=0)</td>
			<td width="27%" align="center"> Filter Vis. (Layer=29)</td>
		</tr>
<tr>
			<td width="27%"> <img src="https://raw.githubusercontent.com/utkuozbulak/pytorch-cnn-visualizations/master/input_images/spider.png"> </td>
			<td width="27%"> <img src="https://raw.githubusercontent.com/utkuozbulak/cnn-gifs/master/spider_layer_graph.gif"> </td>
			<td width="27%" align="center"> <img src="https://raw.githubusercontent.com/utkuozbulak/cnn-gifs/master/spider_filter_graph.gif"> </td>
		</tr>
	</tbody>
</table>

---

### obs 
layer 9
- head 10 focus on the desk edge constantly
- h1: focus on the target object, and lastly both pot and grippers+object are given attention
- h4 focuses on eggplant explicitly

layer 10
- head 10: at first gripper, eggplants are given attention, then attention switches from ep to pot.
- h6 initially has spread attention over background info, and then focuses on pot.
- h2, h5-8, h12 focuses explicitly on grippers

layer 11
- head 1 focuses on eggplant, at some internal images, attention given to background.
- h2, 11, 12 focuses specifically on robot hands and sensors?
- h8 has attention spread over background knowledge like desk edges
- h9: objects are given most attention score


- sudden change of attention focus in last step? e.g. L11 eggplants are given most attention at first, then grippers are given attention at last step suddenly; h4 at last focus sudden change to the drawer
- (e.g. result1-ep3: failed) at 7-9 layers, specific attn is given to objects/grippers, but at last layers, some heads give attention to background/desk edges information
- how to cope with the situation when language instructions are inconsistent with the image observation?
- r1-ep5 in last layer, only one head focuses on cloth, other heads on irrelavant objects/background


Observation
1. sharp attention at mid-depth layers (like 8–9) but then blurry, distracted attention at deeper layers 
2. 