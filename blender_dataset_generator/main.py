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
CRUMPLED_PLANE = 'Crumpled_plane'

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
        texture_ray_casting.get_uv_coords_map(outputs, res_x, res_y, res_coef, file_path_name + '_uv.tiff')
        # Save Z-buffer image
        texture_ray_casting.get_z_value_map(z_vals, res_x, res_y, res_coef, file_path_name + '_z.tiff')
        # Save the render image
#        texture_ray_casting.export_outputs(file_path_name + '.txt', outputs)


def generate_data(textures_dir, renders_dir, surfaces_dir, cam_name, target_name,
        surface_name, n_samples=-1):
        """Generate data by changing textures, scene and views"""
        
        # ----- General constants -----
        FRAME_NUMBER = 6            # Animation frame to generate
        FRAME_REMOVE_CRUMPLED = 5   # If > 0 the crumpled object is removed at that frame
        HIDE_CRUMPLED = True        # Hide crumpled object when its removal is set
        RES_X = 1920                # Render resolution X
        RES_Y = 1080                # Render resolution Y
        RES_COEF = 0.30             # Resolution reduction (normalized num of rays to render)
        ORIGIN_WORLD_CENTER = True  # Set the target origin to the (0,0,0) (else the bbox center)
        
        # ----- Camera constants -----
        VIEWS_PER_TEXTURE = 2               # Camera random views to render for each image
        FOCAL_LENGTH = 50                   # Focal length
        PADDING_PERC = 0.15                 # The camera will not look further than (1-padding) from the center
        DIST_RADIUS_RANGE = (2.5, 7)        # Radius range in meters
        SECTOR_ANGLE_RANGE = (0, math.pi/3) # Sector angle rangle to place camera at
        CAMERA_DEC_PLACES = 9               # Decimal places to round output camera data

        # ----- Crumpled plane constants -----
        CRUMPLED_PLANE_CUTS_RANGE = (2, 15) # Subdivision cuts
        CRUMPLE_FACTOR_RANGE = (0.3, 0.40)  # Range to pick the max strengh (height) of the peaks
        REDUCTION_RANGE = (0.95, 1.25)      # Percentual reduction compared to the texture
        
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
            'TENSION_STIFFNESS': 2000,          # Tension (stretching resistance)
            'BENDING': 2.0,                     # Bending
            'COMPRESSION_STIFFNESS': 200,       # Compression
            'SHEAR_STIFFNESS': 200,             # Shear
            'COLLISION_QUALITY': 5,             # Collision quality
            'MIN_COLLISION_DISTANCE': 0.01,     # Minimum distance for collisions
            'GRAVITY': 10,                      # Gravity
            'SPEED_MULTIPLIER': 1,              # Speed multiplier
            'SMOOTH_FACTOR': 0.5,               # Smooth factor
            'CORRECTIVE_SMOOTH_FACTOR': 0.5,    # Corrective smooth factor
            'CORRECTIVE_SMOOTH_ITERATIONS': 5   # Corrective smooth repeat
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
        AREA_LOCATION = (0,0,5)       # Area light location
        AREA_SIZE = 5                 # Area light radius size
        AREA_ENERGY_RANGE = (50, 100) # Area light watt energy range
        

        # Load steady objects
        surface_object = bpy.data.objects[surface_name]
        cam = bpy.data.objects[cam_name]
        bpy.context.scene.camera = cam
        cam.data.lens = FOCAL_LENGTH
        
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
                
                # ----- Initial settings -----
                CRUMPLE_FACTOR = random.uniform(*CRUMPLE_FACTOR_RANGE)
                
                # ----- Target object settings -----
                # Change the image texture
                loaded_texture = bpy.data.images.load(texture_path)
                texture_width, texture_height = loaded_texture.size
                size_ratio = texture_width / texture_height
                
                # Target object dimensions
                dim_w = TARGET_OBJECT_MAX_SIZE if size_ratio >= 1 else TARGET_OBJECT_MAX_SIZE * size_ratio
                dim_h = TARGET_OBJECT_MAX_SIZE if size_ratio <= 1 else TARGET_OBJECT_MAX_SIZE / size_ratio
                
                # Create a target object plane
                target_object = data_generation.create_target_object(TARGET_OBJECT,
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
                crumpled_object = data_generation.create_crumpled_plane(obj_name=CRUMPLED_PLANE,
                    size=(dim_w * reduction, dim_h * reduction), location=(0, 0, 0),
                    cuts_range=CRUMPLED_PLANE_CUTS_RANGE, crumple_factor=CRUMPLE_FACTOR)
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
                light_objects.append(data_generation.create_area_light('Light_area_0',
                    AREA_LOCATION, AREA_SIZE, random.randint(*AREA_ENERGY_RANGE)))
                light_objects.append(data_generation.create_light('Light_0', main_light_props))
                light_objects.append(data_generation.create_light('Light_1', main_light_props))
                
                
                # ----- Animation of the target object -----
                cloth_modifier = target_object.modifiers.get('Cloth')
                   
                # Remove the collision crumpled object at the specific frame
                # Using this the target falls a bit under to touch the background plane
                scene_init.setup_collision_removal_handler(crumpled_object, FRAME_REMOVE_CRUMPLED, HIDE_CRUMPLED, FRAME_NUMBER)

                cloth_modifier.point_cache.frame_start = 1
                cloth_modifier.point_cache.frame_end = FRAME_NUMBER
                # Bake and shift the animation to a random frame
                bpy.ops.ptcache.bake_all(bake=True)
                bpy.context.scene.frame_set(FRAME_NUMBER)
                bpy.context.view_layer.update()
                
                
                # ----- Target object info -----
                world_bbox_coords = [target_object.matrix_world @ Vector(corner) for corner in target_object.bound_box]

                x_coords = [coord.x for coord in world_bbox_coords]
                y_coords = [coord.y for coord in world_bbox_coords]
                z_coords = [coord.z for coord in world_bbox_coords]
                
                bbox_min = Vector((min(x_coords), min(y_coords), min(z_coords)))
                bbox_max = Vector((max(x_coords), max(y_coords), max(z_coords)))

                
                # ----- Shift target object origin -----
                # Using the cursor set the target object origin to the world center
                # or the object bounding box center
                center = (0.0, 0.0, 0.0) if ORIGIN_WORLD_CENTER else (bbox_min + bbox_max) / 2
                bpy.context.scene.cursor.location = center
                # Recenter the origin after the animation
                target_object.select_set(True)
                bpy.context.view_layer.objects.active = target_object
                bpy.ops.object.origin_set(type='ORIGIN_CURSOR')
                target_object.select_set(False)
                
                
#                # Export the target object plane mesh to /path/<file><number>.obj
#                filepath = '/home/darbix/Desktop/planes/plane.obj'
#                base, ext = os.path.splitext(filepath)
#                i = 1
#                while os.path.exists(f"{base}{i}{ext}"):
#                    i += 1
#                filepath = f"{base}{i}{ext}"
#                target_object.select_set(True)
#                bpy.ops.wm.obj_export(filepath=filepath, export_selected_objects=True,
#                    apply_modifiers=True, filter_glob='*.obj')
#                target_object.select_set(False)
#                continue
                
                
                # ----- Render -----
                # Change a camera view and render a result
                
                camera_info_list = []
                for view_index in range(0, VIEWS_PER_TEXTURE):
                    # Get a random target object point to look at
                    point_x = random.uniform(bbox_min[0], bbox_max[0]) * (1 - PADDING_PERC)
                    point_y = random.uniform(bbox_min[1], bbox_max[1]) * (1 - PADDING_PERC)

                    # Set the camera to a specific view position
                    random_location = data_generation.get_random_camera_location(
                        random.uniform(*DIST_RADIUS_RANGE), *SECTOR_ANGLE_RANGE)
                    camera = bpy.data.objects[MAIN_CAMERA]
                    camera.location = random_location
                    # Change the camera look to the random point
                    data_generation.look_at(camera, Vector((point_x, point_y, 0)))
                    bpy.context.view_layer.update()
                    
                    str_view_index = f"{view_index:02d}"
                    
                    # Keep data for the save at the end of the render
                    data_row = [str_view_index, *data_generation.get_camera_info(cam, target_object, CAMERA_DEC_PLACES)]
                    camera_info_list.append(data_row)

                    # Render the view
                    render_path = os.path.join(renders_dir,
                        f"{texture_name_without_extension}_view_{str_view_index}.jpg")
                    data_generation.render_view(render_path)    
                    
                    # Cast the rays from the camera to the target object's texture 
                    cast_rays(render_path.rsplit('.', 1)[0],
                        res_x=RES_X, res_y=RES_Y, res_coef=RES_COEF, visualize=False)
                
#                    return # TODO shows only one view 
                
                
                # Remove created objects
                for light_object in light_objects:
                    bpy.data.objects.remove(light_object)
                bpy.ops.object.select_all(action='DESELECT') 
                
                # Save data for cameras
                info_data_path = os.path.join(renders_dir,
                        f"{texture_name_without_extension}_data.txt")
                data_generation.save_camera_info(info_data_path, camera_info_list)
                
                print(f"Views for {texture_name} rendered.")



if __name__ == "__main__":
    initialize_scene(surface_size=10)
    
    if not os.path.exists(renders_dir):
        os.makedirs(renders_dir)
    
    # Set random seed
    random.seed(5)
    
    num_images = 2
    generate_data(textures_dir, renders_dir, surfaces_dir,
             MAIN_CAMERA, TARGET_OBJECT, SURFACE_OBJECT, num_images)
