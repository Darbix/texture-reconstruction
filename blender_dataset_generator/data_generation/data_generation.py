import bpy
import os
import math
import bmesh
from mathutils import Euler, Quaternion
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


def create_area_light(name, location, size, energy):
    """Creates an area light object"""
    
    light_data = bpy.data.lights.new(name, 'AREA')
    light_object = bpy.data.objects.new(name, light_data)

    light_object.location = location
    light_data.energy = energy
    light_data.size = size

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


def create_crumpled_plane(obj_name="Crumpled_plane", size=(2, 2), location=(0, 0, 0),
        cuts_range=(2,12), crumple_factor=0.2, perc_deformed_range=(0.25, 0.75), dissolve_range=(0.0,1.0)):
    """
    Creates a crumpled pad to act as an auxiliary surface

    Args:
        obj_name: Name of the object
        size: Size tuple in meters
        location: Location in the world
        cuts_range: Range for random select of a number of division cuts
        crumple_factor: The max height to deform vertices to
        perc_deformed_range: Procentual range for a number of vertices to deform
        dissolve_range: Procentual range for a number of inner vertices to dissolve
    """
    # Remove object if it exists
    if bpy.data.objects.get(obj_name):
        bpy.data.objects.remove(bpy.data.objects[obj_name], do_unlink=True)

    # Add a new plane
    bpy.ops.mesh.primitive_plane_add(size=1, align='WORLD', location=location)
    plane = bpy.context.object
    plane.name = obj_name

    # Scale the plane to the desired x and y dimensions
    plane.dimensions = (*size, 0)

    cuts = random.randint(*cuts_range)

    bpy.ops.object.mode_set(mode='EDIT')
    # Subdivide faces
    bpy.ops.mesh.subdivide(number_cuts=cuts)

    # Dissolve random number of inner vertices to create diverse non-square faces
    mesh = plane.data
    bm = bmesh.from_edit_mesh(mesh)
    inner_vertices = [v for v in bm.verts if v.is_valid and all(abs(coord) < 0.5 for coord in v.co.xy)]
    num_to_dissolve = random.randint(int(len(inner_vertices) * dissolve_range[0]),
                                     int(len(inner_vertices) * dissolve_range[1]))
    if num_to_dissolve > 0:
        vertices_to_dissolve = random.sample(inner_vertices, num_to_dissolve)
        bmesh.ops.dissolve_verts(bm, verts=vertices_to_dissolve)

    bmesh.update_edit_mesh(mesh)

    # Divide faces to triangles 
    bmesh.ops.poke(bm, faces=bm.faces)
    bmesh.update_edit_mesh(plane.data)
    bpy.ops.object.mode_set(mode='OBJECT')


    # Beta distribution parameters
    alpha = 1.5
    beta = 4

    # Number of vertices to deform in Z coordinate
    perc_deformed = random.uniform(*perc_deformed_range)
    # Crumple the vertices
    for vert in plane.data.vertices:
        # Beta distributed values are added as a height to Z
        if(random.uniform(0.0, 1.0) < perc_deformed):
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


def get_random_camera_location(radius, min_angle=0, max_angle=math.pi/3):
    """
    Get a random camera location in the specific sphere sector.
    The upper hemisphere is in the range min_angle=0 and max_angle=math.pi/2
    """
    # Hemisphere sector from the top (0 is the top, math.pi / 4 is +-45°)
    theta = random.uniform(min_angle, max_angle)
    phi = random.uniform(0, 2 * math.pi)
    
    # Spherical to Cartesian conversion
    x = radius * math.sin(theta) * math.cos(phi)
    y = radius * math.sin(theta) * math.sin(phi)
    z = radius * math.cos(theta)
    
    return [x, y, z]


def look_at(camera_obj, target):
    """Rotates the camera so that it looks at the specific target point"""
    # Compute the direction vector from the camera to the target
    direction = camera_obj.location - target
    
    # Quaternion rotation to track the target point
    rot_quat = direction.to_track_quat('Z', 'Y')
    
    # Apply the rotation to look at the point
    camera_obj.rotation_mode = 'QUATERNION'
    camera_obj.rotation_quaternion = rot_quat

    # Roll the camera around its view axis depending on z rotation angle
    camera_obj.rotation_mode = 'XYZ'
    roll_quat = Quaternion((0.0, 0.0, 1.0), -camera_obj.rotation_euler.z)
    camera_obj.rotation_mode = 'QUATERNION'
    camera_obj.rotation_quaternion = camera_obj.rotation_quaternion @ roll_quat

    
def save_camera_info(info_data_path, camera_info_list):
    """Writes ',' separated data to the file at info_data_path"""
    with open(info_data_path, 'w') as f:
        for data_row in camera_info_list:
            f.write(','.join(map(str, data_row)) + '\n')
    

def get_camera_info(camera, target, dec_places):
    """
    Extract camera relative position and quaternion rotation info 
    
    Returns: Relative coordinates to the target origin and the quaternion rotation
        [pos_x, pos_y, pos_z, rot_w, rot_x, rot_y, rot_z]
    """
    # Get the camera's world location and rotation
    cam_loc = camera.matrix_world.to_translation()
    cam_rot = camera.matrix_world.to_quaternion()

    # Get the target's world location and its inverse rotation
    target_loc = target.matrix_world.to_translation()
    target_rot_inv = target.matrix_world.to_quaternion().inverted()

    # Calculate relative location and rotation
    relative_loc = cam_loc - target_loc
    relative_rot = target_rot_inv @ cam_rot

    result = [*relative_loc, *relative_rot]
    rounded_result = [round(value, dec_places) for value in result]
    
    return rounded_result
