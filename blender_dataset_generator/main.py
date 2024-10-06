import os
import sys
import random
import bpy

# Absolute path to the project directory (can be set manually)
abs_curr_dir = os.path.dirname(bpy.data.filepath) if bpy.data.filepath else os.getcwd()

sys.path.insert(0, abs_curr_dir + '/scene_init')
import scene_init

sys.path.insert(0, abs_curr_dir + '/texture_ray_casting')
import texture_ray_casting

sys.path.insert(0, abs_curr_dir + '/data_generation')
import data_generation


TARGET_OBJECT = 'Target_object'
MAIN_CAMERA = 'Main_camera'
SURFACE_OBJECT = 'Surface_object'

textures_dir = abs_curr_dir + '/texture_images' 
renders_dir = abs_curr_dir + '/camera_images'
surfaces_dir = abs_curr_dir + '/surface_images'


def initialize_scene():
    # Delete all objects
    scene_init.delete_all_objects()
    
    # Target object
    target_object = scene_init.create_target_object(TARGET_OBJECT, subdivisions=10, dimensions=(2, 2, 0), location=(0, 0, 0.05))
    scene_init.set_target_object_modifiers(target_object, mass=0.005, quality=5)
    
    # Background object
    background_object = scene_init.create_background_object(SURFACE_OBJECT, dimensions=(5, 5, 0.2), location=(0, 0, -0.1))
    scene_init.set_background_object_modifiers(background_object)
    
    # Camera
    scene_init.create_camera(name=MAIN_CAMERA, location=(0.0, -5.0, 5.0), rotation_deg=(45, 0, 0))
    
    # Other settings
    curr_frame = 1
    scene_init.apply_simulation(target_object, start_frame=1, end_frame=curr_frame)


def cast_rays(file_path, visualize=False):
    # Target object with the texture
    target_object = bpy.data.objects[TARGET_OBJECT]

    # Camera which is the ray source
    cam = bpy.data.objects[MAIN_CAMERA]

    texture_ray_casting.set_render_resolution(1920, 1080)
    
    # Cast rays from the camera to the texture for res_coef * <n_pixels> rays
    outputs = texture_ray_casting.cast_rays_to_texture(cam, target_object, res_coef=0.01, visualize=visualize)
    
    texture_ray_casting.export_outputs(file_path, outputs)


def generate_data(textures_dir, renders_dir, surfaces_dir, cam_name, target_name,
    surface_name, n_samples=-1):
    """Generate data by changing textures, scene and views"""
    
    # -------------------- Settings -------------------- 
    frame_range = (1, 500)

    texture_material_props = {
        # 1.0 is fully matte, 0.0 is fully glossy
        'roughness_range': (0.0, 1.0)
    }

    main_light_props = {
        'hue': (0.0, 0.4),
        'saturation': (0.0, 1.0),
        'value': (0.5, 1.0),
        'power': (500, 2000),
        'x': (-4.0, 4.0),
        'y': (-4.0, 4.0),
        'z': (1.0, 4.0)
    }

    dist = 3.5
    camera_views = [
        # Side views
        {'location': [0.0, -dist, 2.0], 'rotation': [60, 0,   0]},
        {'location': [dist,  0.0, 2.0], 'rotation': [60, 0,  90]},
        {'location': [0.0,  dist, 2.0], 'rotation': [60, 0, 180]},
        {'location': [-dist, 0.0, 2.0], 'rotation': [60, 0, 270]},
        # Top view
        {'location': [0.0,   0.0, 6.0], 'rotation': [0,  0,   0]},
    ]
    # -------------------------------------------------- 
        
    # Load objects
    target_object = bpy.data.objects[target_name]
    cam = bpy.data.objects[cam_name]
    bpy.context.scene.camera = cam
    
    # Get nodes of the target object
    nodes, texture_node = data_generation.set_material_nodes(target_object, 'Target_texture_material')
    
    # Get all input target object texture names
    image_names = sorted(os.listdir(textures_dir))
    n_samples = n_samples if n_samples >= 0 else len(image_names)
    
    # For each target texture
    for image_name in image_names[:n_samples]:
        if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            name_without_extension = image_name.rsplit('.', 1)[0]
            image_path = os.path.join(textures_dir, image_name)
            
            # Change the image texture
            # TODO modify the dimensions of the plane to fit the image ratio
            texture_node.image = bpy.data.images.load(image_path)
            
            # Adjust the material properties
            data_generation.adjust_material(nodes, texture_material_props)
            # TODO modify physics of the target object (change gravity, ...)
            # TODO modify surface background texture, material, ...
            bpy.context.view_layer.update()
            
            # Shift the animation to a random frame
            bpy.context.scene.frame_set(random.randint(*frame_range))
            
            # Create the main light
            light_object = data_generation.create_light('Light_0', main_light_props)
            
            # Change a camera view and render a result
            # TODO camera views will not be 5 constant points
            for view_index, view_item in enumerate(camera_views):
                # Set the camera to a specific view position
                data_generation.set_camera_position_and_rotation(cam, view_item)
            
                # Render the view
                render_path = os.path.join(renders_dir,
                    f"{name_without_extension}_view{view_index:02d}.jpg")
                data_generation.render_view(render_path)    
                
                cast_rays(render_path.rsplit('.', 1)[0] + '.txt')
                
            # Remove created objects
            bpy.data.objects.remove(light_object)
            print(f"Views for {image_name} rendered.")
    


if __name__ == "__main__":
    initialize_scene()
    
    if not os.path.exists(renders_dir):
        # Create the directory
        os.makedirs(renders_dir)
    
    generate_data(textures_dir, renders_dir, surfaces_dir,
             MAIN_CAMERA, TARGET_OBJECT, SURFACE_OBJECT, 2)
