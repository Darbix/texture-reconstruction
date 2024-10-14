import bpy
import math
import random

# # Create and subdivide a plane
# def create_target_object(name, subdivisions, dimensions=(2, 2, 0), location=(0, 0, 0)):
#     bpy.ops.mesh.primitive_plane_add(size=1, enter_editmode=False, location=location)
#     target_object = bpy.context.object # The object becomes active
#     target_object.name = name
#     target_object.dimensions = dimensions
#     
#     # Create fragments
#     bpy.ops.object.mode_set(mode='EDIT')
#     bpy.ops.mesh.subdivide(number_cuts=subdivisions)
#     bpy.ops.object.mode_set(mode='OBJECT')
#     
#     bpy.ops.object.shade_smooth()
#     return target_object

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



if __name__ == "__main__":
    delete_all_objects()
    
    # Target object
    target_object = create_target_object("Target_object", subdivisions=10, dimensions=(2, 2, 0), location=(0, 0, 0.05))
    set_target_object_modifiers(target_object, mass=0.005, quality=5)
    
    # Background object
    background_object = create_background_object("Background_object", dimensions=(5, 5, 0), location=(0, 0, 0))
    set_background_object_modifiers(background_object)
    
    # Camera
    create_camera(name="Main_camera", location=(0.0, -5.0, 5.0), rotation_deg=(45, 0, 0))
    
    # Other settings
    curr_frame = 1
    apply_simulation(target_object, start_frame=1, end_frame=curr_frame)

