import bpy
from mathutils import Vector, Quaternion
import numpy as np
import bmesh
import os


def convert_local_to_texture(obj, local_coords):
    """Converts local coordinates local_coords of object obj to UV texture coordinates"""
    # Ensure the object is a mesh and is in the correct context
    if obj.type != 'MESH':
        print("The object must be a mesh")
        return None

    # Access the mesh data (dependency graph) in its current evaluated state
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated_obj = obj.evaluated_get(depsgraph)
    # Get the deformed mesh data
    mesh = evaluated_obj.data

    uv_layer = mesh.uv_layers.active.data
    closest_face_index = -1
    closest_distance = float('inf')

    # Transform local coordinates to world coordinates
    world_coords = obj.matrix_world @ local_coords

    # Iterate through all the polygons to find the closest face
    for poly in mesh.polygons:
        # Calculate the polygon's world-space center (average of polygon's vertices)
        world_vertices = (evaluated_obj.matrix_world @ mesh.vertices[vert].co for vert in poly.vertices)
        world_center = sum(world_vertices, Vector()) / len(poly.vertices)

        # Calculate the distance to the world coordinate
        distance = (world_coords - world_center).length
        
        if distance < closest_distance:
            closest_distance = distance
            closest_face_index = poly.index

    # If a closest face is found, retrieve its UV coordinates
    if closest_face_index != -1:
        closest_poly = mesh.polygons[closest_face_index]
        
        # Collect UV coordinates for the closest face
        uv_coords = [uv_layer[loop_index].uv for loop_index in closest_poly.loop_indices]
        
        if uv_coords:
            # Return the average UV coordinate of the face
            return sum(uv_coords, Vector((0.0, 0.0))) / len(uv_coords)
    return None


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
        outputs: Normalized camera pixels mapped to texture normalized coordinates
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

    for ix, _ in enumerate(ray_range_x):
        for iy, _ in enumerate(ray_range_y):
            values[ix, iy] = None
            outputs[ix, iy] = None
            
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
            hit, location, norm, face =  target_object.ray_cast(origin, direction)
            
            if hit:
                # Store world space hits for ray visualization
                values[ix,iy] = (matrix_world @ location)

                # Normalized local space coordinates of the target object hit point
                # "1 -" inverts coordinates so that the 0,0 is top left not bottom left
                # texture_coords = (location[0] + 0.5,
                #                  1.0 - (location[1] + 0.5)) #  "1 -" to invert y=0 to the top left
                
                # Normalized texture coordinates (averaged by polygon vertices)
                texture_coords = convert_local_to_texture(target_object, location)
                # Remove '1 -' to shift 0,0 from the top left to the bottom left
                texture_coords = (texture_coords[0], 1 - texture_coords[1])

                # Normalized coordinates on the rendered camera image
                cam_x = ray_rel_x + 0.5
                # The camera sensor has aspect 1x1, so the normalization relates to that size
                cam_y = (1 - ((ray_rel_y * (cam_res_x / cam_res_y)) + 0.5))
                
                outputs[ix,iy] = (cam_x, cam_y, *texture_coords)
    
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
    
    return outputs


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
