import sys
sys.path.append('/content/drive/MyDrive/Code_BrainTumorSeg_Conf/')

import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from utils import post_trans, inference

# Define a custom colormap with different colors for each label
label_colors = ['red','yellow', 'green']

# RGB values for the colors
color_values = {'red': (255, 0, 0, 255),  'yellow': (255, 255, 0, 255),'green': (0, 255, 0, 255)}
background_color = (0, 0, 0, 255)

def visualize_results(model, val_loader, model_file_out, munber_images, device):

    model.load_state_dict(torch.load(os.path.join(model_file_out)))
    model.eval()

    stop=1
    for val_val in val_loader:
        stop+=1
        with torch.no_grad():
            # select one image to evaluate and visualize the model output
            val_input = val_val["image"].to(device)
            roi_size = (128, 128, 128)
            sw_batch_size = 4
            val_output = inference(val_input, model)
            print("val_output: ", val_output.shape)
            val_output = post_trans(val_output[0])
            print("val_output: ", val_output.shape)


            image_sample_np =  val_val["image"].numpy()
            z_slice =  image_sample_np.shape[2] // 2


            plt.figure("image", (24, 6))
            for i in range(4):
                plt.subplot(1, 4, i + 1)
                plt.title(f"image channel {i}")
                plt.imshow(val_val["image"][0, i, z_slice].detach().cpu(), cmap="gray")
            plt.show()
            # visualize the 3 channels label corresponding to this image
            plt.figure("label", (18, 6))
            for i in range(3):
                plt.subplot(1, 3, i + 1)
                plt.title(f"label channel {i}")
                plt.imshow(val_val["label"][0, i, z_slice].detach().cpu())
            plt.show()

            # visualize the 3 channels model output corresponding to this image
            plt.figure("output", (18, 6))
            for i in range(3):
                plt.subplot(1, 3, i + 1)
                plt.title(f"output channel {i}")
                plt.imshow(val_output[i, z_slice].detach().cpu())
            plt.show()

            ## Combine label
            label_sample_np = val_val['label'].numpy()
            image_sample_np = np.full((label_sample_np.shape[3], label_sample_np.shape[4], 4), background_color, dtype=np.uint8)

            num_channels_labels = label_sample_np.shape[1]
            for channel in range(num_channels_labels-1, -1, -1):
                label_channel = label_sample_np[0, channel, z_slice]

                # Overlay the label with a unique color
                label_color = label_colors[channel % len(label_colors)]
                color_value = color_values[label_color]

                # Create a mask for the current label channel
                label_mask = label_channel > 0

                # Apply the color with alpha channel to the corresponding pixels in the composite label
                image_sample_np[label_mask] = color_value

            ### predict combine
            print("val_output: ", val_output.shape)
            val_output =  val_output.cpu().numpy()
            image_sample_np_pre = np.full((val_output.shape[2], val_output.shape[3], 4), background_color, dtype=np.uint8)
            num_channels_labels_pre = val_output.shape[0]

            for channel in range(num_channels_labels_pre-1, -1, -1):
                label_channel = val_output[channel, z_slice]
                label_color = label_colors[channel % len(label_colors)]
                color_value = color_values[label_color]
                label_mask = label_channel > 0
                image_sample_np_pre[label_mask] = color_value
            plt.show()


            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes[0].imshow(image_sample_np)
            axes[0].set_title("Composite Label")
            axes[1].imshow(image_sample_np_pre)
            axes[1].set_title("Composite Label Predict")
            plt.show()

            if stop == munber_images:
                break