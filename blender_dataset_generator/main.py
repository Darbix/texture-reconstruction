import numpy as np
from mathutils import Vector
import importlib
import random
import bmesh
import math
import bpy
import sys
import os

# Absolute path to the project directory (can be set manually)
abs_curr_dir = os.path.dirname(bpy.data.filepath) if bpy.data.filepath else os.getcwd()

# Import and reload modules to update
sys.path.insert(0, abs_curr_dir + '/scene_init')
import scene_init
importlib.reload(scene_init)

sys.path.insert(0, abs_curr_dir + '/texture_ray_casting')
import texture_ray_casting
importlib.reload(texture_ray_casting)

sys.path.insert(0, abs_curr_dir + '/data_generation')
import data_generation
importlib.reload(data_generation)
    

# ----- Constants -----
TARGET_OBJECT = 'Target_object'   # Target texture object plane name
MAIN_CAMERA = 'Main_camera'       # Name of the main camera for rendering 
SURFACE_OBJECT = 'Surface_object' # Background sourface object name
CRUMPLED_PLANE = 'Crumpled_plane' # Name of the crumpled plane for target deformations

TEXTURES_DIR = abs_curr_dir + '/texture_images' # Path to the source texture images
SURFACES_DIR = abs_curr_dir + '/surface_images' # Path to the surface textures
RENDERS_DIR = abs_curr_dir + '/camera_images'   # Path to the render directory
OBJ_DIR = abs_curr_dir + '/scene_objects'       # Path to the exported scene objects

NUM_SAMPLES = 2     # Max number of unique image sets to generate
EXPORT_OBJ = False  # Bool to generate .obj objects
SURFACE_SIZE = 10   # Constant background surface size in meters


def generate_data(textures_dir, renders_dir, surfaces_dir, cam_name, target_name,
        surface_name, export_obj=False, n_samples=-1):
        """Generate data by changing textures, scene and views"""
        
        # ----- General constants -----
        FRAME_NUMBER = 6            # Animation frame to generate
        FRAME_REMOVE_CRUMPLED = 5   # If > 0 the crumpled object is removed at that frame
        HIDE_CRUMPLED = True        # Hide crumpled object when its removal is set
        RES_X = 1920                # Render resolution X
        RES_Y = 1080                # Render resolution Y
        RES_COEF = 1.00             # Resolution reduction (normalized num of rays to render)
        ORIGIN_WORLD_CENTER = True  # Set the target origin to the (0,0,0) (else the bbox center)
        FLIP_UV = False             # Flip the UV coordinations vertically (0,0 will be the top left)
        
        # ----- Camera constants -----
        VIEWS_PER_TEXTURE = 5               # Camera random views to render for each image
        FOCAL_LENGTH = 50                   # Focal length
        PADDING_PERC = 0.2                  # The camera will not look further than (1-padding) from the center
        DIST_RADIUS_RANGE = (2.5, 7)        # Radius range in meters
        SECTOR_ANGLE_RANGE = (0, math.pi/3) # Sector angle rangle to place camera at
        CAMERA_DEC_PLACES = 9               # Decimal places to round output camera data

        # ----- Crumpled plane constants -----
        CRUMPLED_PLANE_CUTS_RANGE = (1, 10) # Subdivision cuts
        CRUMPLE_FACTOR_RANGE = (0.20, 0.35) # Range to pick the max strengh (height [m]) of the peaks
        REDUCTION_RANGE = (0.95, 1.25)      # Percentual reduction compared to the texture
        PERC_DEFORMED_RANGE = (0.10, 0.60)  # Procentual amount of vertices to be randomly increased in Z
        DISSOLVE_RANGE = (0.0, 0.30)        # Procentual amount of vertices to dissolve (retransforms faces)
        
        # ----- Target object constants -----
        TARGET_OBJECT_CUTS = 100            # Subdivision cuts
        HEIGHT_ABOVE_CRUMPLED = 0.02        # How high above the pad the target is
        TARGET_OBJECT_MAX_SIZE = 3          # Image/texture max size in meters 
        
        texture_material_props = {
            'ROUGHNESS_RANGE': (0.4, 1.0) # 1.0 is fully matte, 0.0 is fully glossy
        }
        
        surface_material_props = {
            'ROUGHNESS_RANGE': (0.2, 1.0),
            'METALLIC_RANGE': (0.0, 0.3)
        }
        
        physics_props = {
            'CLOTH_QUALITY': 5,                 # Quality steps
            'VERTEX_MASS': 0.5,                 # Vertex mass
            'AIR_DAMPING': 1.0,                 # Air viscosity
            'TENSION_STIFFNESS': 5000,          # Tension (stretching resistance)
            'BENDING': 1000,                    # Bending
            'COMPRESSION_STIFFNESS': 1000,      # Compression
            'SHEAR_STIFFNESS': 1000,            # Shear
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
            'z': (2.0, 7.0),
            'SHADOW_FILTER_RANGE': (3, 6)
        }
        AREA_LOCATION = (0,0,5)            # Area light location
        AREA_SIZE = 5                      # Area light radius size
        AREA_ENERGY_RANGE = (50, 100)      # Area light watt energy range
        AREA_SHADOW_FILTER_RANGE = (1, 5) # Area shadow filter range
        
        # ----- Loading -----
        # Load steady objects
        surface_object = bpy.data.objects[surface_name]
        cam = bpy.data.objects[cam_name]
        bpy.context.scene.camera = cam
        cam.data.lens = FOCAL_LENGTH
        
        # Get static surface nodes for materials
        nodes_surface, texture_node_surface = data_generation.set_material_nodes(surface_object, 'Surface_texture_material')
        
        # Get all target object texture file names and surface file names
        texture_names = sorted(os.listdir(textures_dir))
        cyclic_texture_names = [texture_names[i % len(texture_names)] for i in range(n_samples)]
        
        surface_names = os.listdir(surfaces_dir)
        random.shuffle(surface_names)
        
        
        render_set = 0 # Identificator number for the singular data collection
        # ----- Iterating over source textures -----
        for texture_name in cyclic_texture_names:
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
                    cuts_range=CRUMPLED_PLANE_CUTS_RANGE, crumple_factor=CRUMPLE_FACTOR,
                    perc_deformed_range=PERC_DEFORMED_RANGE, dissolve_range=DISSOLVE_RANGE)
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
                    AREA_LOCATION, AREA_SIZE, AREA_ENERGY_RANGE, AREA_SHADOW_FILTER_RANGE))
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
                            
                
                # Export .obj files
                if(export_obj):
                    export_scene_objects(OBJ_DIR, render_set, [target_object])
                
                # ----- Render -----
                camera_info_list = []
                # Change a camera view and render a result
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
                        f"set_{render_set:05d}_{texture_name_without_extension}_view_{str_view_index}.jpg")
                    data_generation.render_view(render_path)
                    
                    # Cast the rays from the camera to the target object's texture 
                    ray_cast_and_export_maps(render_path.rsplit('.', 1)[0],
                        res_x=RES_X, res_y=RES_Y, res_coef=RES_COEF, flip_uv=FLIP_UV, visualize=False)

                # ----- Remove created lights -----
                for light_object in light_objects:
                    bpy.data.objects.remove(light_object)
                bpy.ops.object.select_all(action='DESELECT') 
                
                # ----- Save data for cameras -----
                info_data_path = os.path.join(renders_dir,
                        f"set_{render_set:05d}_{texture_name_without_extension}_data.txt")
                data_generation.save_camera_info(info_data_path, camera_info_list)
                
                print(f"Views for {texture_name} rendered.")
                render_set += 1


def initialize_scene(surface_size):
    """Initialize the scene"""
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


def ray_cast_and_export_maps(file_path_name, res_x=1920, res_y=1080, visualize=False,
    res_coef=1.0, flip_uv=False):
    """Call ray casting and export result per-pixel maps for UV and Z"""
    # Target object with the texture
    target_object = bpy.data.objects[TARGET_OBJECT]

    # Camera which is the ray source
    cam = bpy.data.objects[MAIN_CAMERA]

    texture_ray_casting.set_render_resolution(res_x, res_y)
    
    # Cast rays from the camera to the texture for res_coef * <n_pixels> rays
    outputs, z_vals = texture_ray_casting.cast_rays_to_texture(cam, target_object,
        res_coef=res_coef, flip_uv=flip_uv, visualize=visualize)
    
    # If file path to the general name is set, save data
    if(file_path_name):
        # Save UV map image
        texture_ray_casting.get_uv_coords_map(outputs, res_x, res_y, res_coef, file_path_name + '_uv.tiff')
        # Save Z-buffer image
        texture_ray_casting.get_z_value_map(z_vals, res_x, res_y, res_coef, file_path_name + '_z.tiff')


def export_scene_objects(export_obj_path, render_set, objects):
    """Exports .obj files with objects to the directory"""
    filepath = os.path.join(export_obj_path, f'plane_{render_set:05d}.obj')
    
    # Deselect all objects in the scene
    bpy.ops.object.select_all(action='DESELECT')
    # Select objects to export
    for exp_object in objects:
        exp_object.select_set(True)
        
    bpy.ops.wm.obj_export(filepath=filepath, export_selected_objects=True,
        apply_modifiers=True, filter_glob='*.obj')
    
    bpy.ops.object.select_all(action='DESELECT')



random.seed(5)

if __name__ == "__main__":
    initialize_scene(surface_size=SURFACE_SIZE)
    
    if not os.path.exists(RENDERS_DIR):
        os.makedirs(RENDERS_DIR)
    if not os.path.exists(OBJ_DIR):
        os.makedirs(OBJ_DIR)
    
    generate_data(TEXTURES_DIR, RENDERS_DIR, SURFACES_DIR, MAIN_CAMERA, TARGET_OBJECT,
        SURFACE_OBJECT, export_obj=EXPORT_OBJ, n_samples=NUM_SAMPLES)
