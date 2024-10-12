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
x_coords_camera = [row[0] for row in data]  # Column 2 (camera)
y_coords_camera = [row[1] for row in data]  # Column 3 (camera)

# Convert normalized coordinates to pixel coordinates
x_pixel_coords_camera = [int(x * img_width_camera) for x in x_coords_camera]
y_pixel_coords_camera = [int(y * img_height_camera) for y in y_coords_camera]

# Extract columns 2 and 3 (normalized coordinates) for the texture image
x_coords_texture = [row[2] for row in data]  # Column 0 (texture)
y_coords_texture = [row[3] for row in data]  # Column 1 (texture)

# Convert normalized coordinates to pixel coordinates
x_pixel_coords_texture = [int(x * img_width_texture) for x in x_coords_texture]
y_pixel_coords_texture = [int(y * img_height_texture) for y in y_coords_texture]


plt.figure(figsize=(14, 7))
plt.subplot(1, 2, 1)
plt.imshow(image_camera_np)
plt.scatter(x_pixel_coords_camera, y_pixel_coords_camera, color='red', s=5)
plt.title('Camera View with [0], [1] Points')

plt.subplot(1, 2, 2)
plt.imshow(image_texture_np)
plt.scatter(x_pixel_coords_texture, y_pixel_coords_texture, color='blue', s=5)
plt.title('Texture Image with [2], [3] Points')

plt.show()

