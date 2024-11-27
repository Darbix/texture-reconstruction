import bpy
import math
import random

def set_uv_map_texture(plane_obj, image):
    # Create a new material and enable nodes
    material = bpy.data.materials.new(name="ImageMaterial")
    material.use_nodes = True

    # Clear default nodes and create necessary nodes
    nodes = material.node_tree.nodes
    nodes.clear()
    texture_node = nodes.new(type='ShaderNodeTexImage')
    texture_node.image = image
    bsdf_node = nodes.new(type='ShaderNodeBsdfPrincipled')
    output_node = nodes.new(type='ShaderNodeOutputMaterial')

    # Link nodes
    links = material.node_tree.links
    links.new(texture_node.outputs['Color'], bsdf_node.inputs['Base Color'])
    links.new(bsdf_node.outputs['BSDF'], output_node.inputs['Surface'])

    # Assign the material to the plane object
    plane_obj.data.materials.clear()
    plane_obj.data.materials.append(material)

    # Create UV map if it doesn't exist
    if not plane_obj.data.uv_layers:
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.uv.unwrap(method='ANGLE_BASED', margin=0.00)
        bpy.ops.object.mode_set(mode='OBJECT')

    # Set texture to UV map (if needed, currently unused)
    uv_map = plane_obj.data.uv_layers.active


# Add physics modifiers (physics of the target object)
def set_target_object_modifiers(obj, mass=0.005, quality=5):
    cloth_mod = obj.modifiers.new(name="Cloth", type='CLOTH')
    cloth_mod.settings.mass = mass
    cloth_mod.settings.quality = quality
    cloth_mod.settings.gravity = (0, 0, 0)
    
    smooth_mod = obj.modifiers.new(name="Smooth", type='SMOOTH')
    smooth_mod.factor = 0.15 # 0.0-1.0
    

# Create a background object with collision physics
def create_background_object(name, dimensions=(5, 5, 0), location=(0, 0, 0)):
    if bpy.data.objects.get(name):
        bpy.data.objects.remove(bpy.data.objects[name], do_unlink=True)

    bpy.ops.mesh.primitive_plane_add(size=1, enter_editmode=False, location=location)
    background_object = bpy.context.object
    background_object.name = name
    background_object.dimensions = dimensions
    return background_object


# Add physics modifiers (physics of the background object)
def set_background_object_modifiers(obj):
    collision_mod = obj.modifiers.new(name="Collision", type='COLLISION')


def create_camera(name, location=(0, 0, 0), rotation_deg=(0, 0, 0)):
    camera_data = bpy.data.cameras.new(name=name)
    camera_object = bpy.data.objects.new(name, camera_data)
 
    camera_object.location = location
    camera_object.rotation_euler = tuple(math.radians(deg) for deg in rotation_deg)
    
    # Link the camera object to the current collection
    bpy.context.collection.objects.link(camera_object)

    return camera_object


def apply_simulation(obj, start_frame, end_frame):
    try:
        # Attempt to set the frame
        bpy.context.scene.frame_set(start_frame)
        # Check if the cloth modifier exists
        cloth_modifier = obj.modifiers["Cloth"]
        bpy.ops.ptcache.bake_all(bake=True)
        
        bpy.context.scene.frame_set(end_frame)
    except Exception as e:
        raise e
        
        
def delete_all_objects():
    """Delete all objects in the current scene."""
    for obj in bpy.context.scene.objects:
        bpy.data.objects.remove(obj, do_unlink=True)


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

