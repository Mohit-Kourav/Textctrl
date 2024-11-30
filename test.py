import pickle
import matplotlib.pyplot as plt
import numpy as np

# Dump the tensor
output_path = "/disk512gb/TextCtr/TextCtrl/latent_features.pkl"

with open(output_path, "rb") as f:
    output = pickle.load(f)

# Assuming `output` is your encoder output with shape [4, 320, 32, 32]
feature_map = output[0, 0, :, :]  # Select batch 0, channel 0
feature_map = feature_map.cpu().numpy()


# List of colormaps to use
colormaps = ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'cool', 'hot', 'gray']

# Save images for each colormap
for cmap in colormaps:
    plt.figure(figsize=(6, 6))
    plt.imshow(feature_map, cmap=cmap)
    plt.colorbar()
    plt.title(f"Feature Map with {cmap} colormap")
    
    # Save the plot as an image
    output_image_path = f"feature_map_{cmap}.png"
    plt.savefig(output_image_path, dpi=300)  # Save with high resolution
    plt.close()  # Close the figure to free up memory
    print(f"Feature map saved with {cmap} colormap as {output_image_path}")
# # Plot the feature map
# plt.figure(figsize=(6, 6))
# plt.imshow(feature_map, cmap='viridis')  # Visualize with a colormap
# plt.colorbar()
# plt.title("Feature Map: Batch 0, Channel 0")

# # Save the plot as an image
# output_image_path = "feature_map_batch0_channel0.png"
# plt.savefig(output_image_path, dpi=300)  # Save with high resolution
# plt.close()  # Close the figure to free up memory