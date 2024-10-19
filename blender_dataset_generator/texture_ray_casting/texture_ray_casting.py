import bpy
from mathutils import Vector, Quaternion
import numpy as np
import bmesh
import os


def convert_local_to_texture(obj, face_index, local_coords):
    """
    Converts local object's coordinates to UV texture coordinates for the specified face.
    Weightens UV face by local_coords position in the vertex face square.  
    """
    # Access the mesh data in its current evaluated state
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated_obj = obj.evaluated_get(depsgraph)

    # Get the deformed mesh data and the active UV layer
    mesh = evaluated_obj.data
    uv_layer = mesh.uv_layers.active.data

    # Retrieve the polygon corresponding to the given face index
    closest_poly = mesh.polygons[face_index]
    loop_indices = closest_poly.loop_indices

    # Get the UV coordinates and vertex positions for the specified polygon (quad)
    uv_coords = [uv_layer[loop_idx].uv for loop_idx in loop_indices]
    vertex_positions = [mesh.vertices[vert_idx].co for vert_idx in closest_poly.vertices]


    def get_uv_from_triangle(v0, v1, v2, uv0, uv1, uv2):
        """Calculates UV coordinates from local coordinates using barycentric interpolation."""
        
        # Vectors for the triangle
        v0_v1 = v1 - v0
        v0_v2 = v2 - v0
        v0_p = local_coords - v0
        
        # Compute dot products
        d00 = v0_v1.dot(v0_v1)
        d01 = v0_v1.dot(v0_v2)
        d11 = v0_v2.dot(v0_v2)
        d20 = v0_p.dot(v0_v1)
        d21 = v0_p.dot(v0_v2)

        # Calculate the denominator
        denom = d00 * d11 - d01 * d01
        if denom == 0:
            return None  # Collinear points

        # Compute barycentric coordinates
        v = (d11 * d20 - d01 * d21) / denom
        w = (d00 * d21 - d01 * d20) / denom
        u = 1 - v - w

        # Check if the local coordinates are within the triangle
        if 0.0 <= u <= 1.0 and 0.0 <= v <= 1.0 and u + v <= 1.0:
            return (u * uv0 + v * uv1 + w * uv2)

        # Limit the values to not return None (the point must be in the face triangle anyway) 
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


def cast_rays_to_texture(cam, target_object, res_coef=1.0, visualize=False):
    """
    Cast rays from the camera through the view plane pixels to the texture object
    Args:
        res_coef: Resolution coeficient 0.0 to 1.0.
            Number of rays will be res_coef * resolution_x * resolution_y
    Return:
        outputs: 2D array of tuples with normalized camera pixels 
            mapped to texture coords (cam_x, cam_y, uv_x, uv_y)
    """
    # Save the current view mode
    mode = bpy.context.area.type

    # Set view mode to 3D (makes variables available)
    bpy.context.area.type = "VIEW_3D"

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
    # The origin of rays (the camera center) relative to the target local space scale
    origin = matrix_world_inverted @ cam.matrix_world.translation

    # iterate over all x, y coordinates (vectors from the camera center to its plane pixels) 
    for ix, ray_rel_x in enumerate(ray_range_x):
        for iy, ray_rel_y in enumerate(ray_range_y):
            # Get the current pixel vector
            pixel_vector = Vector((ray_rel_x, ray_rel_y, cam_view_rel_z))
            # Rotate that vector according to a camera rotation
            pixel_vector.rotate(cam.matrix_world.to_quaternion())
            # Convert the world space ray vector to the target object space
            destination = matrix_world_inverted @ (pixel_vector + cam.matrix_world.translation) 
            direction = (destination - origin).normalized()
            
            # Ray casting (in the target object space)
            hit, location, norm, face_index =  target_object.ray_cast(origin, direction)
            
            if hit:
                # Normalized texture coordinates (averaged by polygon vertices)
                texture_coords = convert_local_to_texture(target_object, face_index, location)
                # if(texture_coords is None):
                #     continue

                # Store world space hits for ray visualization
                world_coords = (matrix_world @ location)
                values[ix,iy] = world_coords

                # Remove '1 -' to shift 0,0 from the top left to the bottom left
                texture_coords = (texture_coords[0], 1 - texture_coords[1])

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
        
    # Reset a view mode
    bpy.context.area.type = mode
    
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
    Visualize rays and hits
    Args:
        values: 2D matrix of hit points in the world space
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
    """numpy array as input TODO"""
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


def get_uv_coords_map(coords_matrix, res_x, res_y, res_coef, file_path):
    # Convert the ray data array to upsampled render size array
    coords_data = np.array(coords_matrix, dtype=object)
    upsampled_data = upsample_ray_data(coords_data, res_x, res_y, res_coef)

    pixels = np.zeros((res_y, res_x, 4), dtype=np.float32)

    # Mask for valid not None values
    mask = np.array([[v is not None for v in y] for y in upsampled_data])

    # Extract valid values where the mask is True
    # RGB
    pixels[mask, 0] = np.array([[v[2] if v is not None else 0 for v in y] for y in upsampled_data])[mask]
    pixels[mask, 1] = np.array([[v[3] if v is not None else 0 for v in y] for y in upsampled_data])[mask]
    # Alpha
    pixels[mask, 3] = 1.0

    # Create and save the image
    image = bpy.data.images.new("UV_coord_map", width=res_x, height=res_y, alpha=True)
    image.pixels = pixels.ravel()
    image.filepath_raw = file_path
    image.file_format = 'PNG'
    image.save()


def get_z_value_map(z_vals_matrix, res_x, res_y, res_coef, file_path):

    data_matrix = np.array(z_vals_matrix, dtype=float)

    z_values = upsample_ray_data(data_matrix, res_x, res_y, res_coef)

    # Find the max value in the array, ignoring None
    max_z_value = np.nanmax(data_matrix)

    # Normalize the non-NaN values (NaN stay NaN)
    normalized_values = np.clip(z_values / max_z_value, 0.0, 1.0)

    # Create pixel array and fill with normalized RGB values
    pixels = np.zeros((res_y, res_x, 4), dtype=np.float32)
    # RGB
    pixels[:, :, :3] = normalized_values[:, :, np.newaxis]
    # Alpha
    pixels[:, :, 3] = 1.0
    pixels[np.isnan(z_values)] = 0.0  # Where NaN

    # Create an image for the map
    image = bpy.data.images.new("", width=res_x, height=res_y, alpha=True)
    image.pixels = pixels.ravel()
    image.filepath_raw = file_path
    image.file_format = 'PNG'
    image.save()


    
# Only for testing
if __name__ == '__main__':
    # Target object with the texture
    target_object = bpy.data.objects['Target_object']

    # Camera which is the ray source
    cam = bpy.data.objects['Main_camera']

    set_render_resolution(1920, 1080)
    
    # Cast rays from the camera to the texture for res_coef * <n_pixels> rays
    outputs = cast_rays_to_texture(cam, target_object, res_coef=0.01, visualize=True)
    
    export_outputs('rays.txt', outputs)
