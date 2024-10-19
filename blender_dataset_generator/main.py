import os
import sys
import random
import bpy
import math
import bmesh
import importlib
import mathutils
import numpy as np

def force_reload_module(module_name, path):
    sys.path.insert(0, path)  # Add the path to the module
    module = __import__(module_name)  # Import the module
    importlib.reload(module)  # Reload the module
    return module

# Absolute path to the project directory (can be set manually)
abs_curr_dir = os.path.dirname(bpy.data.filepath) if bpy.data.filepath else os.getcwd()

sys.path.insert(0, abs_curr_dir + '/scene_init')
import scene_init
importlib.reload(scene_init)

sys.path.insert(0, abs_curr_dir + '/texture_ray_casting')
import texture_ray_casting
importlib.reload(texture_ray_casting)

sys.path.insert(0, abs_curr_dir + '/data_generation')
import data_generation
importlib.reload(data_generation)


TARGET_OBJECT = 'Target_object'
MAIN_CAMERA = 'Main_camera'
SURFACE_OBJECT = 'Surface_object'

textures_dir = abs_curr_dir + '/texture_images' 
renders_dir = abs_curr_dir + '/camera_images'
surfaces_dir = abs_curr_dir + '/surface_images'


def initialize_scene(surface_size):
    # Delete all objects
    scene_init.delete_all_objects()
    
    # Background object
    size = surface_size # Size in meters
    background_object = scene_init.create_background_object(SURFACE_OBJECT,
        dimensions=(size, size, 0), location=(0, 0, 0))
    scene_init.set_background_object_modifiers(background_object)
    
    # Camera
    scene_init.create_camera(name=MAIN_CAMERA, location=(0.0, -5.0, 5.0),
        rotation_deg=(45, 0, 0))


def cast_rays(file_path_name, res_x=1920, res_y=1080, visualize=False, res_coef=1.0):
    # Target object with the texture
    target_object = bpy.data.objects[TARGET_OBJECT]

    # Camera which is the ray source
    cam = bpy.data.objects[MAIN_CAMERA]

    texture_ray_casting.set_render_resolution(res_x, res_y)
    
    # Cast rays from the camera to the texture for res_coef * <n_pixels> rays
    outputs, z_vals = texture_ray_casting.cast_rays_to_texture(cam, target_object,
        res_coef=res_coef, visualize=visualize)
    
    # If file path to the general name is set, save data
    if(file_path_name):
        # Save UV map image
        texture_ray_casting.get_uv_coords_map(outputs, res_x, res_y, res_coef, file_path_name + '_uv.png')
        # Save Z-buffer image
        texture_ray_casting.get_z_value_map(z_vals, res_x, res_y, res_coef, file_path_name + '_z.png')
        # Save the render image
        texture_ray_casting.export_outputs(file_path_name + '.txt', outputs)


def setup_collision_removal_handler(obj, removal_frame, hide, last_frame):
    """
    Set up a frame change handler to remove the collision modifier from the object
    at a specific frame.
    Args:
    - obj: The object with a collision modifier to be removed.
    - removal_frame: The frame at which to remove the collision modifier.
    - hide: Whether to hide the object after removing the modifier.
    - last_frame: The total number of frames in the animation.
    """
    if removal_frame > 0 and removal_frame < last_frame:
        # Define the frame change handler function
        def frame_change_handler(scene):
            # Check if the object still exists in the scene
            if obj and obj.name in scene.objects:
                if scene.frame_current == removal_frame:
                    # Remove collision modifier for the object
                    for modifier in obj.modifiers:
                        if modifier.type == 'COLLISION':
                            obj.modifiers.remove(modifier)
                            if hide:
                                obj.hide_viewport = True
                    # Remove this handler after it has executed to avoid it running again
                    bpy.app.handlers.frame_change_post.remove(frame_change_handler)
            else:
                # If the object doesn't exist anymore, remove the handler
                bpy.app.handlers.frame_change_post.remove(frame_change_handler)

        # Remove any existing handlers to avoid duplicates
        handlers = bpy.app.handlers.frame_change_post.copy()
        for handler in handlers:
            if handler.__name__ == "frame_change_handler":
                bpy.app.handlers.frame_change_post.remove(handler)

        # Add the frame change handler
        bpy.app.handlers.frame_change_post.append(frame_change_handler)



def generate_data(textures_dir, renders_dir, surfaces_dir, cam_name, target_name,
        surface_name, n_samples=-1):
        """Generate data by changing textures, scene and views"""
        
        # ----- General constants -----
        FRAME_NUMBER = 6            # Animation frame to generate
        FRAME_REMOVE_CRUMPLED = 5   # If > 0 the crumpled object is removed at that frame
        HIDE_CRUMPLED = True        # Hide crumpled object when its removal is set
        RES_X = 1920                # Render resolution X
        RES_Y = 1080                # Render resolution Y
        RES_COEF = 0.15             # Resolution reduction (normalized num of rays to render)
        
        # ----- Crumpled plane constants -----
        CRUMPLED_PLANE_CUTS = 3           # Subdivision cuts
        CRUMPLE_FACTOR = 0.40             # How strong (high) the peaks are
        REDUCTION_RANGE = (0.95, 1.25)    # Percentual reduction compared to the texture
        
        # ----- Target object constants -----
        TARGET_OBJECT_CUTS = 100          # Subdivision cuts
        HEIGHT_ABOVE_CRUMPLED = 0.05      # How high above the pad the target is
        TARGET_OBJECT_MAX_SIZE = 3        # Image/texture max size in meters 
        
        texture_material_props = {
            'ROUGHNESS_RANGE': (0.4, 1.0) # 1.0 is fully matte, 0.0 is fully glossy
        }
        
        surface_material_props = {
            'ROUGHNESS_RANGE': (0.2, 1.0),
            'METALLIC_RANGE': (0.0, 0.3)
        }
        
        physics_props = {
            'CLOTH_QUALITY': 10,                # Quality steps
            'VERTEX_MASS': 0.5,                 # Vertex mass
            'AIR_DAMPING': 1.0,                 # Air viscosity
            'TENSION_STIFFNESS': 2000,          # Tension
            'BENDING': 0.1,                     # Bending
            'COMPRESSION_STIFFNESS': 100,       # Compression
            'SHEAR_STIFFNESS': 200,             # Shear
            'COLLISION_QUALITY': 5,             # Collision quality
            'MIN_COLLISION_DISTANCE': 0.01,     # Minimum distance for collisions
            'GRAVITY': 10,                      # Gravity
            'SPEED_MULTIPLIER': 1,              # Speed multiplier
            'SMOOTH_FACTOR': 0.5,               # Smooth factor
            'CORRECTIVE_SMOOTH_FACTOR': 0.1,    # Corrective smooth factor
            'CORRECTIVE_SMOOTH_ITERATIONS': 10  # Corrective smooth repeat
        }
        
        # ----- Light constants -----
        main_light_props = {
            'HUE': (0.0, 0.4),
            'SATURATION': (0.0, 1.0),
            'VALUE': (0.5, 1.0),
            'POWER': (450, 1300),
            'x': (-4.0, 4.0),
            'y': (-4.0, 4.0),
            'z': (2.0, 7.0)
        }
        
        # ----- Camera constants -----
        distance = 7 # [m] from the target object
        d = distance / math.sqrt(2)
        # The distance from the texture center is approximately 6m
        camera_views = [
            # Side views
            {'location': [0.0, -d, d], 'rotation': [45, 0,  0]}, # Bottom
            {'location': [d,  0.0, d], 'rotation': [0, 45,  0]}, # Right
            {'location': [0.0,  d, d], 'rotation': [-45, 0, 0]}, # Top
            {'location': [-d, 0.0, d], 'rotation': [0, -45, 0]}, # Left
            # Top view
            {'location': [0.0, 0.0, distance], 'rotation': [0, 0, 0]},
        ]


        # Load steady objects
        surface_object = bpy.data.objects[surface_name]
        cam = bpy.data.objects[cam_name]
        bpy.context.scene.camera = cam
        
        # Get static surface nodes for materials
        nodes_surface, texture_node_surface = data_generation.set_material_nodes(surface_object, 'Surface_texture_material')
        
        # Get all target object texture file names and surface file names
        texture_names = sorted(os.listdir(textures_dir))
        n_samples = n_samples if n_samples >= 0 else len(texture_names)
        
        surface_names = os.listdir(surfaces_dir)
        random.shuffle(surface_names)
        
        # For each target texture
        for texture_name in texture_names[:n_samples]:
            if texture_name.lower().endswith(('.png', '.jpg', '.jpeg')):      
                      
                texture_name_without_extension = texture_name.rsplit('.', 1)[0]
                texture_path = os.path.join(textures_dir, texture_name)
                
                # ----- Target object settings -----
                # Change the image texture
                loaded_texture = bpy.data.images.load(texture_path)
                texture_width, texture_height = loaded_texture.size
                size_ratio = texture_width / texture_height
                
                # Target object dimensions
                dim_w = TARGET_OBJECT_MAX_SIZE if size_ratio >= 1 else TARGET_OBJECT_MAX_SIZE * size_ratio
                dim_h = TARGET_OBJECT_MAX_SIZE if size_ratio <= 1 else TARGET_OBJECT_MAX_SIZE / size_ratio
                
                # Create a target object plane
                target_object = data_generation.create_target_object("Target_object",
                    subdivisions=TARGET_OBJECT_CUTS, dimensions=(dim_w, dim_h, 0.0),
                    location=(0, 0, CRUMPLE_FACTOR + HEIGHT_ABOVE_CRUMPLED))
                data_generation.add_physics(target_name, physics_props)
                nodes_target, texture_node_target = data_generation.set_material_nodes(target_object, 'Target_texture_material')
                
                # Set the target object texture
                scene_init.set_uv_map_texture(target_object, loaded_texture)
                
                # Adjust the material properties
                data_generation.adjust_material(nodes_target, texture_material_props)
                
                # Poke square faces to 4 triangles each
                bpy.ops.object.mode_set(mode='EDIT')
                bm = bmesh.from_edit_mesh(target_object.data)
                bmesh.ops.poke(bm, faces=bm.faces)
                bmesh.update_edit_mesh(target_object.data)
                bpy.ops.object.mode_set(mode='OBJECT')
                
                
                # ----- Crumpled plane settings -----
                # Create the crumpled plane to form the main texture object
                reduction = random.uniform(*REDUCTION_RANGE)
                crumpled_object = data_generation.create_crumpled_plane(obj_name="Crumpled_plane",
                    size=(dim_w * reduction, dim_h * reduction), location=(0, 0, 0),
                    cuts=CRUMPLED_PLANE_CUTS, crumple_factor=CRUMPLE_FACTOR)
                crumpled_object.hide_render = True
                
                
                # ----- Surface settings -----
                # Random surface texture select (the list is shuffled)
                surface_name = surface_names.pop(0)
                surface_names.append(surface_name)
                data_generation.adjust_surface(nodes_surface, surface_material_props)
                
                surface_path = os.path.join(surfaces_dir, surface_name)
                texture_node_surface.image = bpy.data.images.load(surface_path)
                
                
                # ----- Light settings -----
                light_objects = []
                light_objects.append(data_generation.create_light('Light_0', main_light_props))
                light_objects.append(data_generation.create_light('Light_1', main_light_props))
                
                
                # ----- Animation of the target object -----
                cloth_modifier = target_object.modifiers.get('Cloth')
                   
                # Remove the collision crumpled object at the specific frame
                # Using this the target falls a bit under to touch the background plane
                setup_collision_removal_handler(crumpled_object, FRAME_REMOVE_CRUMPLED, HIDE_CRUMPLED, FRAME_NUMBER)

                cloth_modifier.point_cache.frame_start = 1
                cloth_modifier.point_cache.frame_end = FRAME_NUMBER
                # Bake and shift the animation to a random frame
                bpy.ops.ptcache.bake_all(bake=True)
                bpy.context.scene.frame_set(FRAME_NUMBER)
                bpy.context.view_layer.update()
                

                
                # ----- Render -----
                # Change a camera view and render a result
                # TODO camera views will not be 5 constant points
                for view_index, view_item in enumerate(camera_views):
                    # Set the camera to a specific view position
                    data_generation.set_camera_position_and_rotation(cam, view_item)
                    bpy.context.view_layer.update()
                    
                    # Render the view
                    render_path = os.path.join(renders_dir,
                        f"{texture_name_without_extension}_view_{view_index:02d}.jpg")
                    data_generation.render_view(render_path)    
                    
                    # Cast the rays from the camera to the target object's texture 
                    cast_rays(render_path.rsplit('.', 1)[0],
                        res_x=RES_X, res_y=RES_Y, res_coef=RES_COEF, visualize=False)
                
                # Remove created objects
                for light_object in light_objects:
                    bpy.data.objects.remove(light_object)
                bpy.ops.object.select_all(action='DESELECT') 
                
                print(f"Views for {texture_name} rendered.")
    


if __name__ == "__main__":
    initialize_scene(surface_size=9)
    
    if not os.path.exists(renders_dir):
        os.makedirs(renders_dir)
    
    # Set random seed
    random.seed(27)
    
    num_images = 2
    generate_data(textures_dir, renders_dir, surfaces_dir,
             MAIN_CAMERA, TARGET_OBJECT, SURFACE_OBJECT, num_images)
