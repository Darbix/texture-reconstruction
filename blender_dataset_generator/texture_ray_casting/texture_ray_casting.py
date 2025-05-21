# texture_ray_casting.py

import bpy
from mathutils import Vector, Quaternion
import numpy as np
import bmesh
import os


def convert_local_to_texture(obj, face_index, local_coords, mesh):
    """
    Convert local object's coordinates to UV texture coordinates for the specified face.
    Given a 3D point, it gets 3D and UV coordinates of the triangle’s corners, computes
    barycentric weights of the point within the triangle and uses them to find the UVs of the point.
    """
    # Get the active UV layer
    uv_layer = mesh.uv_layers.active.data

    # Retrieve the polygon corresponding to the given face index
    closest_poly = mesh.polygons[face_index]
    loop_indices = closest_poly.loop_indices

    # Get the UV coordinates and vertex positions for the specified polygon
    uv_coords = [uv_layer[loop_idx].uv for loop_idx in loop_indices]
    vertex_positions = [mesh.vertices[vert_idx].co for vert_idx in closest_poly.vertices]


    def get_uv_from_triangle(v0, v1, v2, uv0, uv1, uv2):
        """Calculates UV coordinates from local coordinates using barycentric interpolation."""
        # Vectors for the triangle
        v0_v1 = v1 - v0
        v0_v2 = v2 - v0
        v0_p = local_coords - v0 # Vector to the target point local_coords
        
        # Compute dot products
        d00 = v0_v1.dot(v0_v1)
        d01 = v0_v1.dot(v0_v2)
        d11 = v0_v2.dot(v0_v2)
        d20 = v0_p.dot(v0_v1)
        d21 = v0_p.dot(v0_v2)

        # Calculate the denominator
        denom = d00 * d11 - d01 * d01
        if denom == 0:
            # Collinear points
            return None  

        # Compute barycentric coordinates
        v = (d11 * d20 - d01 * d21) / denom
        w = (d00 * d21 - d01 * d20) / denom
        u = 1 - v - w

        # Check if the local coordinates are within the triangle
        if 0.0 <= u <= 1.0 and 0.0 <= v <= 1.0 and u + v <= 1.0:
            return (u * uv0 + v * uv1 + w * uv2)

        # Limit the values to not return None (the point must be in the triangle anyway) 
        v = max(0.0, min(1.0, v))
        w = max(0.0, min(1.0, w))
        u = max(0.0, min(1.0, u))

        return u * uv0 + v * uv1 + w * uv2


    # Calculate the UV point in the triangle face
    uv = get_uv_from_triangle(vertex_positions[0], vertex_positions[1], vertex_positions[2],
                              uv_coords[0], uv_coords[1], uv_coords[2])
    return uv


def draw_point_in_local_space(target_object, location):
    """Add a 3D point at the hitpoint in the target_object local space location"""
    # Create a new mesh and object
    mesh = bpy.data.meshes.new("Point_mesh")
    point_object = bpy.data.objects.new("Point_object", mesh)
    
    # Link the object to the scene
    bpy.context.collection.objects.link(point_object)

    # Set the location to the local coordinates
    point_object.location = location

    # Create a single vertex for the point
    bm = bmesh.new()
    bmesh.ops.create_vert(bm, co=(0, 0, 0))  # Create a single vertex at the origin

    # Update the mesh with the vertex data
    bm.to_mesh(mesh)
    bm.free()

    # Optionally, set the scale of the point to make it visible
    scale = 0.1
    point_object.scale = (scale, scale, scale)  # Scale it down to represent a point

    # Set the point object's parent to the target object for local space
    point_object.parent = target_object


def cast_rays_to_texture(cam, target_object, res_coef=1.0, flip_uv=False, visualize=False):
    """
    Cast rays from the camera through the view plane pixels to the texture object. The result
    values (array of UV coordinates) can be interpolated if a small number of rays is used.
    Number of rays will be res_coef * resolution_x * resolution_y. Returns a 2D array
    of tuples (cam_x, cam_y, uv_x, uv_y) with normalized camera frame pixel locations.
    """

    # Get the camera plane frame
    # Relative to the camera center in world coordinates [m]
    top_r, bottom_r, bottom_l, top_l = cam.data.view_frame(scene=bpy.context.scene)
    # Camera view plane relative z coord in the world space
    cam_view_rel_z = top_l[2]

    # Resolution in pixels
    cam_res_x = int(bpy.context.scene.render.resolution_x * (bpy.context.scene.render.resolution_percentage / 100))
    cam_res_y = int(bpy.context.scene.render.resolution_y * (bpy.context.scene.render.resolution_percentage / 100))

    # Resolution for transmitting rays (number of them)
    num_rays_x = round(cam_res_x * res_coef)
    num_rays_y = round(cam_res_y * res_coef)

    # Pixel points for rays on the camera plane (relative to the camera center)
    # Samples relative to the camera center
    ray_range_x = np.linspace(top_l[0], top_r[0], num_rays_x)
    ray_range_y = np.linspace(top_l[1], bottom_l[1], num_rays_y)

    # 2D array of mappings camera-pixel to texture-pixel
    values = np.empty((ray_range_x.size, ray_range_y.size), dtype=object)
    outputs = np.empty((ray_range_x.size, ray_range_y.size), dtype=object)
    z_vals = np.empty((ray_range_x.size, ray_range_y.size), dtype=object)

    for ix, _ in enumerate(ray_range_x):
        for iy, _ in enumerate(ray_range_y):
            values[ix, iy] = None
            outputs[ix, iy] = None
            z_vals[ix, iy] = None
            
    # Target object local space to world space matrix
    matrix_world = target_object.matrix_world
    # World space to target object local space
    matrix_world_inverted = matrix_world.inverted()
    matrix_cam_quat = cam.matrix_world.to_quaternion()
    # The origin of rays (the camera center) relative to the target local space scale
    origin = matrix_world_inverted @ cam.matrix_world.translation

    # Flips the UV coordination coord -> 1-coord if needed
    get_uv_coord = (lambda c: 1 - c[1]) if flip_uv else (lambda c: c)

    # Variables for convert_local_to_texture()
    # Access the mesh data in its current evaluated state for mesh
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated_obj = target_object.evaluated_get(depsgraph)
    mesh = evaluated_obj.data

    # iterate over all x, y coordinates (vectors from the camera center to its plane pixels) 
    for ix, ray_rel_x in enumerate(ray_range_x):
        for iy, ray_rel_y in enumerate(ray_range_y):
            # Get the current pixel vector
            pixel_vector = Vector((ray_rel_x, ray_rel_y, cam_view_rel_z))
            # Rotate that vector according to a camera rotation
            pixel_vector.rotate(matrix_cam_quat)
            # Convert the world space ray vector to the target object space
            destination = matrix_world_inverted @ (pixel_vector + cam.matrix_world.translation) 
            direction = (destination - origin).normalized()
            
            # Ray casting (in the target object space) limited to max distance 10 m 
            # Note: takes the second most execution time
            hit, location, norm, face_index = target_object.ray_cast(origin, direction, distance=10.0)

            if hit:
                # Normalized texture coordinates (averaged by polygon vertices)
                # Note: takes the most execution time
                texture_coords = convert_local_to_texture(target_object, face_index, location, mesh)

                # Store world space hits for ray visualization
                world_coords = (matrix_world @ location)
                values[ix,iy] = world_coords

                # Remove '1 -' to shift 0,0 from the top left to the bottom left
                texture_coords = (texture_coords[0], get_uv_coord(texture_coords[1]))

                # Normalized coordinates on the rendered camera image
                cam_x = ray_rel_x + 0.5
                # The camera sensor has aspect 1x1, so the normalization relates to that size
                cam_y = (1 - ((ray_rel_y * (cam_res_x / cam_res_y)) + 0.5))
                
                outputs[ix,iy] = (cam_x, cam_y, *texture_coords)

                # Z-buffer values
                z_vals[ix,iy] = (location - pixel_vector).length

    if(visualize):
        # Draw points
        for row in values:
            for w_loc in row:
                if w_loc is not None:
                    loc = matrix_world_inverted @ w_loc
                    draw_point_in_local_space(target_object, loc)       
        
        visualize_rays(values, cam)
    
    return outputs, z_vals


def export_outputs(file_path, outputs):
    # If the file_path is relative, convert it to an absolute path
    if not os.path.isabs(file_path):
        # Use the directory of the currently opened Blender file
        current_dir = os.path.dirname(bpy.data.filepath) if bpy.data.filepath else os.getcwd()
        # Concatenate with the relative file_path
        file_path = os.path.join(current_dir, file_path)

    # Open the file and write the outputs
    with open(file_path, 'w') as file:
        for row in outputs:
            for item in row:
                if item is not None:
                    file.write(' '.join(map(str, item)) + '\n')


def visualize_rays(values, cam):
    """
    Visualizes rays and hits for input 2D array of hit points (in the world space)
    """
    # Create a new mesh for hit points and lines
    mesh = bpy.data.meshes.new(name='Rays_mesh')
    bm = bmesh.new()

    # Camera center BMVert vertex
    bmv_cam = bm.verts.new(cam.matrix_world.translation)

    for location in values.flat:
        if location is not None:
            bmv_loc = bm.verts.new(location)
            bm.edges.new([bmv_loc, bmv_cam]) 

    # Complete the mesh
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()
    mesh.validate()

    # Create and link the new object to the scene
    obj = bpy.data.objects.new('Camera_texture_rays', mesh)
    bpy.context.scene.collection.objects.link(obj)


def set_render_resolution(rx, ry):
    # Set the resolution width and height
    bpy.context.scene.render.resolution_x = rx
    bpy.context.scene.render.resolution_y = ry

    # Set the resolution percentage (scaling)
    # Keep render at 100% of the specified resolution
    bpy.context.scene.render.resolution_percentage = 100


def upsample_ray_data(data_matrix, res_x, res_y, res_coef):
    """
    Upsamples numpy array matrix to res_x * res_y size
    (the source has sizes res_coef * res_<x,y>)
    """
    # Convert z_vals to a numpy array for easier manipulation
    w, h = data_matrix.shape

    # Precompute the indices map (upsample the ray grid to the render resolution)
    x_indices = np.clip((np.arange(res_x) * res_coef).astype(int), 0, w - 1)
    y_indices = h - 1 - np.clip((np.arange(res_y) * res_coef).astype(int), 0, h - 1)

    # Create a grid of y indices for all x at once
    # Repeats the indexing (substitutes loops over all rows and columns)
    y_indices_grid = np.tile(y_indices, (res_x, 1)).T
    x_indices_grid = np.tile(x_indices, (res_y, 1))

    # Retrieve z_values for all pixels at once (2D dimension of res_x*res_y)
    upsampled_data_matrix = data_matrix[x_indices_grid, y_indices_grid]
    
    return upsampled_data_matrix


def setup_and_save_image(file_path, pixels, color_depth_bits_str):
    """
    Set render properties and create image from pixels
    and save to a specific format given by file_path extension
    """
    # Determine file extension and set image format
    _, file_extension = os.path.splitext(file_path)
    file_extension = file_extension.lower()

    image = None
    scene = bpy.context.scene
    prev_format = scene.render.image_settings.file_format
    prev_codec = scene.render.image_settings.exr_codec
    prev_color_depth = scene.render.image_settings.color_depth

    res_x = pixels.shape[1]
    res_y = pixels.shape[0]

    if file_extension in ['.exr']:
        image = bpy.data.images.new("UV_coord_map", width=res_x, height=res_y, alpha=True, float_buffer=True, is_data=True)
        # image.file_format = 'OPEN_EXR'
        scene.render.image_settings.file_format = 'OPEN_EXR'
        scene.render.image_settings.exr_codec = 'ZIP'
        scene.render.image_settings.color_depth = color_depth_bits_str
    elif file_extension in ['.png']:
        image = bpy.data.images.new("UV_coord_map", width=res_x, height=res_y, alpha=True, float_buffer=False, is_data=True)
        # image.file_format = 'PNG'
        scene.render.image_settings.file_format = 'PNG'
        scene.render.image_settings.color_depth = color_depth_bits_str
        # Converts float to PNG itself
    else:
        print(f"Unsupported file extension: {file_extension}. Defaulting to PNG.")
        return None

    image.pixels = pixels.ravel()
    image.filepath_raw = file_path
    image.save_render(file_path)

    # Restore the previous render settings
    scene.render.image_settings.file_format = prev_format
    scene.render.image_settings.exr_codec = prev_codec
    scene.render.image_settings.color_depth = prev_color_depth

    # Remove image data from the memory
    image.user_clear()
    bpy.data.images.remove(image)

    return image


def get_uv_coords_map(coords_matrix, res_x, res_y, res_coef, file_path, color_depth=np.float32):
    """
    Saves the UV map of the texture from coords_matrix and upsamples to res_x*res_y if res_coef < 1
    """
    # Convert the ray data array to upsampled render size array
    coords_data = np.array(coords_matrix, dtype=object)
    upsampled_data = upsample_ray_data(coords_data, res_x, res_y, res_coef)

    pixels = np.zeros((res_y, res_x, 4), dtype=color_depth)

    # Mask for valid not None values
    mask = np.array([[v is not None for v in y] for y in upsampled_data])

    # Extract valid values where the mask is True
    # RGB (assign 0 if None)
    pixels[mask, 0] = np.array([[v[2] if v is not None else 0.0 for v in y] for y in upsampled_data])[mask]
    pixels[mask, 1] = np.array([[v[3] if v is not None else 0.0 for v in y] for y in upsampled_data])[mask]
    # Alpha channel, 1 for valid pixels
    pixels[mask, 3] = 1.0

    color_depth_bits_str = str(np.dtype(color_depth).itemsize * 8)
    image = setup_and_save_image(file_path, pixels, color_depth_bits_str)

    return image


def get_z_value_map(z_vals_matrix, res_x, res_y, res_coef, file_path, color_depth=np.float32):
    """Saves the Z depth map of the texture from z_vals_matrix and upsamples to res_x*res_y if res_coef < 1"""
    data_matrix = np.array(z_vals_matrix, dtype=float)

    z_values = upsample_ray_data(data_matrix, res_x, res_y, res_coef)

    # Find the max value in the array, ignoring None
    max_z_value = np.nanmax(data_matrix)

    # Normalize the non-NaN values (NaN stay NaN)
    normalized_values = np.clip(z_values / max_z_value, 0.0, 1.0)

    # Create pixel array and fill with normalized RGB values
    pixels = np.zeros((res_y, res_x, 4), dtype=color_depth)
    # RGB
    pixels[:, :, :3] = normalized_values[:, :, np.newaxis]
    # Alpha
    pixels[:, :, 3] = 1.0
    pixels[np.isnan(z_values)] = 0.0  # Where NaN

    color_depth_bits_str = str(np.dtype(color_depth).itemsize * 8)
    image = setup_and_save_image(file_path, pixels, color_depth_bits_str)

    return image
