import bpy
import os
import math
from mathutils import Euler
import random
import colorsys


def set_camera_position_and_rotation(cam, view_item):
    """
    Set the camera to a specific location and rotation
    Args:
        view_item: dictionary with items 'location' and 'rotation' per each view
    """
    cam.location = view_item['location']
    cam.rotation_euler = Euler([math.radians(r) for r in view_item['rotation']], 'XYZ')


def create_light(name, props, light_type='POINT'):
    """Creates and sets a light object"""
        
    light_data = bpy.data.lights.new(name, light_type)
    light_object = bpy.data.objects.new(name, light_data)
    
    color_hsv = (
        random.uniform(*props['hue']),
        random.uniform(*props['saturation']),
        random.uniform(*props['value'])
    )

    light_location = (
        random.uniform(*props['x']),
        random.uniform(*props['y']),
        random.uniform(*props['z'])
    )

    light_object.data.energy = random.uniform(*props['power'])
    light_object.data.color = colorsys.hsv_to_rgb(*color_hsv)
    light_object.location = light_location
    
    # Link to the current collection
    bpy.context.collection.objects.link(light_object)
    
    return light_object


def adjust_material(nodes, props):
    """Sets the material properties to nodes"""
    # BSDF node
    bsdf_node = nodes.get("Principled BSDF")
    
    roughness = random.uniform(*props['roughness_range'])
    bsdf_node.inputs['Roughness'].default_value = roughness


def set_material_nodes(target_object, target_material_name):
    """Creates a material to link image textures to it"""
    material = bpy.data.materials.new(name=target_material_name)
    material.use_nodes = True

    # Get the material nodes
    nodes = material.node_tree.nodes
    # Get links for material nodes
    links = material.node_tree.links

    # Add an image texture node
    texture_node = nodes.new(type="ShaderNodeTexImage")

    # Get the default Principled BSDF node (automatically added when use_nodes is True)
    bsdf_node = nodes.get("Principled BSDF")

    # Link the texture color to the material's base color
    links.new(texture_node.outputs['Color'], bsdf_node.inputs['Base Color'])

    # Assign the material to the object (replace or add new)
    if target_object.data.materials:
        target_object.data.materials[0] = material
    else:
        target_object.data.materials.append(material)
    
    return nodes, texture_node


def render_view(render_path):
    """Renders a camera view to a file"""  
    bpy.context.scene.render.filepath = render_path
    bpy.ops.render.render(write_still=True)



if __name__ == "__main__":
    MAIN_CAMERA = 'Main_camera'
    TARGET_OBJECT = 'Target_object'
    SURFACE_OBJECT = 'Surface_object'
    
    
    # Only testing purpose
    def generate_data(textures_dir, renders_dir, surfaces_dir, cam_name, target_name,
        surface_name, n_samples=-1):
        """Generate data by changing textures, scene and views"""
             
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

       
        # Load objects
        target_object = bpy.data.objects[target_name]
        cam = bpy.data.objects[cam_name]
        
        # Get nodes of the target object
        nodes, texture_node = set_material_nodes(target_object, 'Target_texture_material')
        
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
                adjust_material(nodes, texture_material_props)
                # TODO modify physics of the target object (change gravity, ...)
                # TODO modify surface background texture, material, ...
                bpy.context.view_layer.update()
                
                # Shift the animation to a random frame
                bpy.context.scene.frame_set(random.randint(*frame_range))
                
                # Create the main light
                light_object = create_light('Light_0', main_light_props)
                
                # Change a camera view and render a result
                # TODO camera views will not be 5 constant points
                for view_index, view_item in enumerate(camera_views):
                    # Set the camera to a specific view position
                    set_camera_position_and_rotation(cam, view_item)
                
                    # Render the view
                    render_path = os.path.join(renders_dir,
                        f"{name_without_extension}_view{view_index:02d}.jpg")
                    render_view(render_path)    
                    
                    
                # Remove created objects
                bpy.data.objects.remove(light_object)
                print(f"Views for {image_name} rendered.")
    
    abs_curr_dir = os.path.dirname(bpy.data.filepath) if bpy.data.filepath else os.getcwd()

    generate_data(os.path.join(abs_curr_dir, '../texture_images'), 
                  os.path.join(abs_curr_dir, '../camera_images'),
                  os.path.join(abs_curr_dir, '../surface_images'),
                  MAIN_CAMERA, TARGET_OBJECT, SURFACE_OBJECT, 1)
