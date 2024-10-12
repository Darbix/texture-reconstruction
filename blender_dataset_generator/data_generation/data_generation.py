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
        random.uniform(*props['HUE']),
        random.uniform(*props['SATURATION']),
        random.uniform(*props['VALUE'])
    )

    light_location = (
        random.uniform(*props['x']),
        random.uniform(*props['y']),
        random.uniform(*props['z'])
    )
    
    light_object.data.use_shadow = True                  # Enable shadows
    light_object.data.shadow_maximum_resolution = 0.001  # Set resolution limit
    light_object.data.shadow_filter_radius = 3.0         # Blur shaddows

    light_object.data.energy = random.uniform(*props['POWER'])
    light_object.data.color = colorsys.hsv_to_rgb(*color_hsv)
    light_object.location = light_location
    
    # Link to the current collection
    bpy.context.collection.objects.link(light_object)
    
    return light_object


def adjust_material(nodes, props):
    """Sets the material properties to nodes"""
    # BSDF node
    bsdf_node = nodes.get("Principled BSDF")
    
    roughness = random.uniform(*props['ROUGHNESS_RANGE'])
    bsdf_node.inputs['Roughness'].default_value = roughness


def adjust_surface(nodes, props):
    """Sets the material properties to nodes"""
    # BSDF node
    bsdf_node = nodes.get("Principled BSDF")
    
    roughness = random.uniform(*props['ROUGHNESS_RANGE'])
    bsdf_node.inputs['Roughness'].default_value = roughness

    metallic = random.uniform(*props['METALLIC_RANGE'])
    bsdf_node.inputs['Metallic'].default_value = metallic


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


def create_crumpled_plane(obj_name="Crumpled_plane", size=(2, 2), location=(0, 0, 0), cuts=10, crumple_factor=0.2):
    """Creates a crumpled pad to act as an auxiliary surface"""
    
    # Remove object if it exists
    if bpy.data.objects.get(obj_name):
        bpy.data.objects.remove(bpy.data.objects[obj_name], do_unlink=True)

    # Add a new plane
    bpy.ops.mesh.primitive_plane_add(size=1, align='WORLD', location=location)
    plane = bpy.context.object
    plane.name = obj_name

    # Scale the plane to the desired x and y dimensions
    plane.dimensions = (*size, 0)

    # Subdivide the plane
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.subdivide(number_cuts=cuts)
    bpy.ops.object.mode_set(mode='OBJECT')
    
    # Beta distribution parameters
    alpha = 2
    beta = 6

    # Crumple the vertices
    for vert in plane.data.vertices:
        # Beta distributed values are added as a height to Z
        vert.co.z += random.betavariate(alpha, beta) * crumple_factor

    # Apply smooth shading
    bpy.ops.object.shade_smooth()

    # Enable collision physics
    bpy.context.view_layer.objects.active = plane
    collision_modifier = plane.modifiers.new(name="Collision", type='COLLISION')

    # Disable single-sided collision
    collision_modifier.settings.use_culling = False
    
    return plane


def add_physics(target_object_name, props):
    """Adds cloth physics to the specified object and sets its attributes"""

    # Check if the target object exists
    if target_object_name in bpy.data.objects:
        # Get the target object
        target_object = bpy.data.objects[target_object_name]

    # Make sure the object is of type MESH
    if target_object and target_object.type == 'MESH':
        bpy.ops.object.select_all(action='DESELECT')
        bpy.context.view_layer.objects.active = target_object
        target_object.select_set(True)
        
        # Get cloth modifier
        cloth_modifier = target_object.modifiers.get("Cloth")
        if cloth_modifier is None:
            # Add cloth physics since it doesn't exist
            cloth_modifier = target_object.modifiers.new(name="Cloth", type='CLOTH')
        
        # Set cloth modifier properties using the props dictionary
        cloth_modifier.settings.quality = props['CLOTH_QUALITY']
        cloth_modifier.settings.mass = props['VERTEX_MASS']
        cloth_modifier.settings.air_damping = props['AIR_DAMPING']
        cloth_modifier.settings.time_scale = props['SPEED_MULTIPLIER']

        cloth_modifier.settings.tension_stiffness = props['TENSION_STIFFNESS']
        cloth_modifier.settings.bending_stiffness = props['BENDING']
        
        # Physical properties
        cloth_modifier.settings.compression_stiffness = props['COMPRESSION_STIFFNESS']
        cloth_modifier.settings.shear_stiffness = props['SHEAR_STIFFNESS']
        
        # Collisions
        cloth_modifier.collision_settings.collision_quality = props['COLLISION_QUALITY']
        cloth_modifier.collision_settings.distance_min = props['MIN_COLLISION_DISTANCE']
        
        # Field weights
        cloth_modifier.settings.effector_weights.gravity = props['GRAVITY']
        
        # Get smooth modifier
        smooth_modifier = target_object.modifiers.get("Smooth")
        if smooth_modifier is None:
            # Add smooth modifier since it doesn't exist
            smooth_modifier = target_object.modifiers.new(name="Smooth", type='SMOOTH')
        smooth_modifier.factor = props['SMOOTH_FACTOR']
        
        # Get corrective smooth modifier
        # TODO constants move out
        corrective_smooth_modifier = target_object.modifiers.get("CorrectiveSmooth")
        if corrective_smooth_modifier is None:
            # Add Corrective Smooth modifier
            corrective_smooth_modifier = target_object.modifiers.new(name="CorrectiveSmooth", type='CORRECTIVE_SMOOTH')
        corrective_smooth_modifier.factor = props['CORRECTIVE_SMOOTH_FACTOR']
        corrective_smooth_modifier.iterations = props['CORRECTIVE_SMOOTH_ITERATIONS']


def create_target_object(name, subdivisions, dimensions=(2, 2, 0), location=(0, 0, 0)):
    if bpy.data.objects.get(name):
        bpy.data.objects.remove(bpy.data.objects[name], do_unlink=True)   
    
    bpy.ops.mesh.primitive_plane_add(size=1, enter_editmode=False, location=location)
    target_object = bpy.context.object # The object becomes active
    target_object.name = name
    target_object.dimensions = dimensions
    
    # Create fragments
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.subdivide(number_cuts=subdivisions)
    bpy.ops.object.mode_set(mode='OBJECT')
    
    target_object = bpy.context.active_object
    bpy.ops.object.shade_smooth()
    return target_object


def subdivide(plane_obj, cuts):
    # Set the object as active and ensure it's selected
    bpy.context.view_layer.objects.active = plane_obj
    plane_obj.select_set(True)

    # Subdivide the plane
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.subdivide(number_cuts=cuts)
    bpy.ops.object.mode_set(mode='OBJECT')



# Only for testing
if __name__ == "__main__":
    MAIN_CAMERA = 'Main_camera'
    TARGET_OBJECT = 'Target_object'
    SURFACE_OBJECT = 'Surface_object'
    
    
    # Only testing purpose
    random.seed(9)
    
    def generate_data(textures_dir, renders_dir, surfaces_dir, cam_name, target_name,
        surface_name, n_samples=-1):
        """Generate data by changing textures, scene and views"""
        
        # ----- General constants -----
        FRAME_NUMBER = 5
        
        # ----- Crumpled plane constants -----
        CRUMPLED_PLANE_CUTS = 10
        CRUMPLE_FACTOR = 0.3
        REDUCTION_RANGE = (0.8, 1.0) # Percentual reduction compared to the texture
        
        # ----- Target object constants -----
        HEIGHT_ABOVE_CRUMPLED = 0.05
        TARGET_OBJECT_CUTS = 10
        TARGET_OBJECT_MAX_SIZE = 3 # [m] the size of the longest side 
        
        texture_material_props = {
            # 1.0 is fully matte, 0.0 is fully glossy
            'ROUGHNESS_RANGE': (0.3, 1.0)
        }
        
        physics_props = {
            'CLOTH_QUALITY': 10,                # Quality steps
            'VERTEX_MASS': 0.5,                 # Vertex mass
            'AIR_DAMPING': 1.0,                 # Air viscosity
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
            'POWER': (500, 1500),
            'x': (-4.0, 4.0),
            'y': (-4.0, 4.0),
            'z': (1.0, 7.0)
        }
        
        # ----- Camera constants -----
        distance = 6 # [m] from the target object
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
        
        # Get static surface nodes for materials
        nodes_surface, texture_node_surface = set_material_nodes(surface_object, 'Surface_texture_material')
        
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
                target_object = create_target_object("Target_object", subdivisions=TARGET_OBJECT_CUTS, dimensions=(dim_w, dim_h, 0.0),
                    location=(0, 0, CRUMPLE_FACTOR + HEIGHT_ABOVE_CRUMPLED))
                add_physics(target_name, physics_props)
                nodes_target, texture_node_target = set_material_nodes(target_object, 'Target_texture_material')
                # Apply the texture to the target object
                texture_node_target.image = loaded_texture
                
                # Adjust the material properties
                adjust_material(nodes_target, texture_material_props)
                
                
                # ----- Crumpled plane settings -----
                # Create the crumpled plane to form the main texture object
                reduction = random.uniform(*REDUCTION_RANGE)
                crumpled_object = create_crumpled_plane(obj_name="Crumpled_plane",
                    size=(dim_w * reduction, dim_h * reduction), location=(0, 0, 0),
                    cuts=CRUMPLED_PLANE_CUTS, crumple_factor=CRUMPLE_FACTOR)
                crumpled_object.hide_render = True
                
                
                # ----- Surface settings -----
                # TODO modify surface material, ...
                # Random surface texture select (the list is shuffled)
                surface_name = surface_names.pop()
                surface_names.append(surface_name)
                
                surface_path = os.path.join(surfaces_dir, surface_name)
                texture_node_surface.image = bpy.data.images.load(surface_path)

                
                # ----- Light settings -----
                light_object = create_light('Light_0', main_light_props)
                
                
                # ----- Animation of the target object -----
                cloth_modifier = target_object.modifiers.get('Cloth')
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
                    set_camera_position_and_rotation(cam, view_item)
                
                    # Render the view
                    render_path = os.path.join(renders_dir,
                        f"{texture_name_without_extension}_view{view_index:02d}.jpg")
                    render_view(render_path)    
               
                    
                # Remove created objects
                bpy.data.objects.remove(light_object)
                bpy.ops.object.select_all(action='DESELECT') 
                
                print(f"Views for {texture_name} rendered.")
    
    
    
    abs_curr_dir = os.path.dirname(bpy.data.filepath) if bpy.data.filepath else os.getcwd()
    
    bpy.context.scene.render.resolution_x = 1920
    bpy.context.scene.render.resolution_y = 1080
    
    generate_data(os.path.join(abs_curr_dir, '../texture_images'), 
                  os.path.join(abs_curr_dir, '../camera_images'),
                  os.path.join(abs_curr_dir, '../surface_images'),
                  MAIN_CAMERA, TARGET_OBJECT, SURFACE_OBJECT, 1)



