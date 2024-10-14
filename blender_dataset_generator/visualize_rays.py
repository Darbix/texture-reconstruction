import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# Define universal image prefix, identifier, and view number
image_prefix = "image_"
image_id = "0000"
view_number = "view_00"

# Load the first image (camera view)
image_path_camera = f'camera_images/{image_prefix}{image_id}_{view_number}.png'
image_camera = Image.open(image_path_camera)
image_camera_np = np.array(image_camera)

# Get dimensions of the camera view image
img_height_camera, img_width_camera, _ = image_camera_np.shape
print("Camera resolution:", img_height_camera, img_width_camera)

# Load the second image (texture image)
image_path_texture = f'texture_images/{image_prefix}{image_id}.jpg'
image_texture = Image.open(image_path_texture)
image_texture_np = np.array(image_texture)

# Get dimensions of the texture image
img_height_texture, img_width_texture, _ = image_texture_np.shape
print("Texture resolution:", img_height_texture, img_width_texture)

# Load the .txt data
txt_file_path = f'camera_images/{image_prefix}{image_id}_{view_number}.txt'
data = []
with open(txt_file_path, 'r') as file:
    for line in file:
        values = [float(x) for x in line.split()]
        data.append(values)

# Extract columns 0 and 1 (normalized coordinates) for the camera image
x_coords_camera = np.array([row[0] for row in data])  # Column 0 (camera)
y_coords_camera = np.array([row[1] for row in data])  # Column 1 (camera)

# Convert normalized coordinates to pixel coordinates
x_pixel_coords_camera = (x_coords_camera * img_width_camera).astype(int)
y_pixel_coords_camera = (y_coords_camera * img_height_camera).astype(int)

# Extract columns 2 and 3 (normalized coordinates) for the texture image
x_coords_texture = np.array([row[2] for row in data])  # Column 2 (texture)
y_coords_texture = np.array([row[3] for row in data])  # Column 3 (texture)

# Convert normalized coordinates to pixel coordinates
x_pixel_coords_texture = (x_coords_texture * img_width_texture).astype(int)
y_pixel_coords_texture = (y_coords_texture * img_height_texture).astype(int)

# Define colors to iterate over
colors = np.array(['red', 'green', 'blue', 'orange'])
num_colors = len(colors)

# Create color indices for the scatter plot
color_indices_camera = np.arange(len(x_pixel_coords_camera)) % num_colors
color_indices_texture = np.arange(len(x_pixel_coords_texture)) % num_colors

plt.figure(figsize=(14, 7))

# Plot camera view with points in different colors
plt.subplot(1, 2, 1)
plt.imshow(image_camera_np)

# Use a single scatter plot call
plt.scatter(x_pixel_coords_camera, y_pixel_coords_camera, 
            color=colors[color_indices_camera], s=5)

plt.title('Camera View with [0], [1] Points')

# Plot texture image with points in different colors
plt.subplot(1, 2, 2)
plt.imshow(image_texture_np)

# Use a single scatter plot call
plt.scatter(x_pixel_coords_texture, y_pixel_coords_texture, 
            color=colors[color_indices_texture], s=5)

plt.title('Texture Image with [2], [3] Points')

plt.show()

