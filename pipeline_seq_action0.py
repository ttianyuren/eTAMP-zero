import os
from re import S
import sys
from tabnanny import verbose

from torch import set_float32_matmul_precision

script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(script_dir, os.pardir))
if repo_root not in sys.path:
    sys.path.append(repo_root)

import pybullet as p
import pybullet_data
import numpy as np
import math
import random
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import trimesh
from utils.pybullet_tools.utils import (
    Pose,
    pairwise_collision,
    set_pose,
    get_attachement,
    get_pose,
    sample_placement_on_aabb_seed,
    get_aabb,
    HideOutput,
    LockRenderer
)

from motion import (
    plan_joint_motion,
    get_joint_positions,
    set_joint_positions,
    ik_solver,
)
from motion import (
    MOTION_JOINTS_FRANKA,
    GRIPPER_JOINTS_FRANKA,
    EE_LINK_FRANKA,
    CONTRL_PATH,
    CONTRL_ATTACH,
    CONTRL_DETTACH,
)

from scene import Scene_Panda

# import open3d as o3d
from PIL import Image, ImageDraw
import imageio
from itertools import product, combinations, count


def draw_frame(tform, life_time=1000, high_light=False):
    length = 0.7
    width = 3

    po = tform[:3, 3]
    px = tform[:3, 0] * length + po
    py = tform[:3, 1] * length + po
    pz = tform[:3, 2] * length + po

    cx = (1, 0, 0)
    cy = (0, 1, 0)
    cz = (0, 0, 1)

    if high_light:
        cx = (1, 0.7, 0.7)
        cy = (0.7, 1, 0.7)
        cz = (0.7, 0.7, 1)

    line_x = p.addUserDebugLine(po, px, cx, width, lifeTime=life_time)
    line_y = p.addUserDebugLine(po, py, cy, width, lifeTime=life_time)
    line_z = p.addUserDebugLine(po, pz, cz, width, lifeTime=life_time)

    return [line_x, line_y, line_z]


def random_pose(x=[-0.8, 0.8], y=[-0.8, 0.8], z=[0.0, 1.8]):
    position = [
        random.uniform(x[0], x[1]),
        random.uniform(y[0], y[1]),
        random.uniform(z[0], z[1]),
    ]
    orientation = p.getQuaternionFromEuler(
        [
            random.uniform(0, 2 * math.pi),
            random.uniform(0, 2 * math.pi),
            random.uniform(0, 2 * math.pi),
        ]
    )
    return position, orientation


def create_wall():
    # Wall parameters
    wall_width = 0.1
    wall_height = 2.0
    wall_length = 3.0

    # Create a cuboid as the wall
    wall_collision_id = p.createCollisionShape(
        shapeType=p.GEOM_BOX,
        halfExtents=[wall_length / 2, wall_width / 2, wall_height / 2],
    )
    wall_visual_id = p.createVisualShape(
        shapeType=p.GEOM_BOX,
        halfExtents=[wall_length / 2, wall_width / 2, wall_height / 2],
        rgbaColor=[0.6, 0.6, 0.6, 1],
    )

    # Position the wall 2 meters to the right of the robot along the x-axis
    wall_position = (0, 1.5, wall_height / 2)
    wall_id = p.createMultiBody(
        baseCollisionShapeIndex=wall_collision_id,
        baseVisualShapeIndex=wall_visual_id,
        basePosition=wall_position,
    )
    return wall_id


def draw_frame(tform, life_time=0.5, high_light=False):
    length = 0.3
    width = 3

    po = tform[:3, 3]
    px = tform[:3, 0] * length + po
    py = tform[:3, 1] * length + po
    pz = tform[:3, 2] * length + po

    cx = (1, 0, 0)
    cy = (0, 1, 0)
    cz = (0, 0, 1)

    if high_light:
        cx = (1, 0.7, 0.7)
        cy = (0.7, 1, 0.7)
        cz = (0.7, 0.7, 1)

    line_x = p.addUserDebugLine(po, px, cx, width, lifeTime=life_time)
    line_y = p.addUserDebugLine(po, py, cy, width, lifeTime=life_time)
    line_z = p.addUserDebugLine(po, pz, cz, width, lifeTime=life_time)

    return [line_x, line_y, line_z]


def matrix_from_quat(quat):
    return np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)


def tform_from_pose(pose):
    """Get homogeneous transformation matrix from pose"""
    (point, quat) = pose
    tform = np.eye(4)
    tform[:3, 3] = point
    tform[:3, :3] = matrix_from_quat(quat)
    return tform


def gen_tform(mat_rot=np.eye(3), point=[0, 0, 0]):
    tform = np.eye(4)
    tform[:3, 3] = point
    tform[:3, :3] = mat_rot
    return tform


def gen_grasping(half_extents=[0.04, 0.08, 0.16], ft=0.018, ft2=0.06):
    # ft: grasping margin on the object, from the finger tip center to the obj edge
    hx = half_extents[0]
    hy = half_extents[1]
    hz = half_extents[2]

    list_g = []
    # grasping from -y wrt object frame
    yo = hy - ft
    zo = hz - ft
    zo2 = hz - ft2
    list_g.append(gen_tform([[0, 1, 0], [0, 0, 1], [1, 0, 0]], [0, -yo, zo2]))
    list_g.append(gen_tform([[0, 1, 0], [0, 0, 1], [1, 0, 0]], [0, -yo, 0]))
    list_g.append(gen_tform([[0, 1, 0], [0, 0, 1], [1, 0, 0]], [0, -yo, -zo2]))
    # +y
    list_g.append(gen_tform([[0, 1, 0], [0, 0, -1], [-1, 0, 0]], [0, yo, zo2]))
    list_g.append(gen_tform([[0, 1, 0], [0, 0, -1], [-1, 0, 0]], [0, yo, 0]))
    list_g.append(gen_tform([[0, 1, 0], [0, 0, -1], [-1, 0, 0]], [0, yo, -zo2]))
    # +z
    list_g.append(gen_tform([[0, 1, 0], [1, 0, 0], [0, 0, -1]], [0, 0, zo]))
    # -z
    list_g.append(gen_tform([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], [0, 0, -zo]))

    mat_z_180 = np.array([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    list_ng = []

    for g in list_g:
        list_ng.append(np.dot(g, mat_z_180))

    return list_g + list_ng


def gen_obj_grasping(obj_id):
    position, orientation = p.getBasePositionAndOrientation(obj_id)
    pose_obj = tform_from_pose((position, orientation))
    list_g = gen_grasping()
    list_obj_g = []

    for g in list_g:
        list_obj_g.append(np.dot(pose_obj, g))

    return list_obj_g


def gen_down_dir():
    # determine which surface faces downwards

    y_down = Pose(euler=(0.5 * np.pi, 0, 0))
    ny_down = Pose(euler=(-0.5 * np.pi, 0, 0))
    z_down = Pose(euler=(0, np.pi, 0))
    nz_down = Pose()

    list_down_dir = [y_down, ny_down, z_down, nz_down]

    return list_down_dir


def fix_gripper_opening(robot_id, opening_width):
    # Finger length = 50
    # Grasping force [N] 30-70
    # Travel Range [mm] 80

    # Set the target position of the gripper fingers and apply high force to lock them
    left_finger_joint, right_finger_joint = GRIPPER_JOINTS_FRANKA

    # Divide the opening width by 2 for each finger
    left_finger_position = opening_width / 2
    right_finger_position = opening_width / 2

    # Use a high force to lock the joints in place
    max_force = 1000

    # Set joint control for the fingers to fix their position
    p.setJointMotorControl2(
        robot_id,
        left_finger_joint,
        p.POSITION_CONTROL,
        targetPosition=left_finger_position,
        force=max_force,
    )
    p.setJointMotorControl2(
        robot_id,
        right_finger_joint,
        p.POSITION_CONTROL,
        targetPosition=right_finger_position,
        force=max_force,
    )


def reset_objects(list_obj_pose):

    for o, p in list_obj_pose:
        set_pose(o, p)


def create_box(half_extents, pose=Pose(), color=[1, 0, 0, 1], fixed=False):
    collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
    visual_shape = p.createVisualShape(
        p.GEOM_BOX, halfExtents=half_extents, rgbaColor=color
    )
    position, orientation = pose
    obj_id = p.createMultiBody(
        baseMass=1,
        baseCollisionShapeIndex=collision_shape,
        baseVisualShapeIndex=visual_shape,
        basePosition=position,
        baseOrientation=orientation,
    )
    if fixed:
        p.changeDynamics(obj_id, -1, mass=0)
    return obj_id


def add_objects(half_extents=[0.038, 0.08, 0.16]):

    pos = (0.196, 0.611, 0.4501)
    ori = (0, 0, 0, 1)
    o1 = create_box(half_extents, (pos, ori), [1, 0, 0, 1])

    pos = (-0.075, 0.635, 0.7976)
    ori = (
        0.020368044547872787,
        -0.6977657834774182,
        0.7157624702789898,
        -0.019802532907656954,
    )
    o2 = create_box(half_extents, (pos, ori), [0, 1, 0, 1])

    pos = (-0.4, 0, 0.1605)
    # # position = (0.15, 0.3, 0.16)
    ori = (0, 0, 0, 1)
    o3 = create_box(half_extents, (pos, ori), [0, 0, 1, 1])
    # p.changeDynamics(o3, -1, mass=0)

    return o1, o2, o3


def add_regions():
    # Simplified the half extents [0.038, 0.08, 0.16]
    # half_extents=np.array(half_extents)*0.1
    half_extents2 = [0.11, 0.14, 0.0001]
    half_extents1 = [0.18, 0.14, 0.0001]
    half_extents0 = [0.3, 0.3, 0.0001]

    pos = (-0.5, 0.0, 0.0001)
    ori = (0, 0, 0, 1)
    r01 = create_box(half_extents0, (pos, ori), [1, 0, 0, 0.2], True)

    pos = (0.5, 0.0, 0.0001)
    ori = (0, 0, 0, 1)
    r02 = create_box(half_extents0, (pos, ori), [1, 0, 0, 0.2], True)

    pos = (-0.23, 0.63, 0.290)
    ori = (0, 0, 0, 1)
    r11 = create_box(half_extents1, (pos, ori), [0, 1, 0, 0.2], True)

    pos = (0.23, 0.63, 0.290)
    ori = (0, 0, 0, 1)
    r12 = create_box(half_extents1, (pos, ori), [0, 1, 0, 0.2], True)

    pos = (-0.28, 0.63, 0.714)
    ori = (0, 0, 0, 1)
    r21 = create_box(half_extents2, (pos, ori), [0, 0, 1, 0.2], True)

    pos = (0.0, 0.63, 0.714)
    ori = (0, 0, 0, 1)
    r22 = create_box(half_extents2, (pos, ori), [0, 0, 1, 0.2], True)

    pos = (0.28, 0.63, 0.714)
    ori = (0, 0, 0, 1)
    r23 = create_box(half_extents2, (pos, ori), [0, 0, 1, 0.2], True)

    return r01, r02, r11, r12, r21, r22, r23


def set_camera_view():

    # Move the camera closer and higher
    cameraDistance = 1.4  # Decrease to move the camera closer
    cameraYaw = 45  # Angle in degrees (horizontal rotation)
    cameraPitch = -30  # Angle in degrees (negative for a top-down view)
    cameraTargetPosition = [0.4, 0.4, 0.4]  # The point the camera is focused on

    # Set the camera view
    p.resetDebugVisualizerCamera(
        cameraDistance=cameraDistance,
        cameraYaw=cameraYaw,
        cameraPitch=cameraPitch,
        cameraTargetPosition=cameraTargetPosition,
    )


def draw_ee_frame(robot_id):
    # Get the position and orientation of the Franka end effector (Link 11)
    end_effector_state = p.getLinkState(robot_id, 11)
    end_effector_pos = end_effector_state[4]  # Position of the end-effector
    end_effector_orn = end_effector_state[5]  # Orientation of the end-effector

    draw_frame(tform_from_pose((end_effector_pos, end_effector_orn)))


def draw_obj_frame(obj_id, life_time=0.5):
    position, orientation = p.getBasePositionAndOrientation(obj_id)
    draw_frame(tform_from_pose((position, orientation)))


def highlight_collision_pairs(collision_pairs, highlight_color=(1, 0.7, 0.5, 0.8)):
    """
    Highlights the collision pairs by changing their visual colors.

    Args:
    - collision_pairs (list of tuples): List of tuples where each tuple contains (robot_id, obstacle_id).
    - highlight_color (tuple): The RGBA color to highlight the objects. Default is red.
    """
    for robot_id, obstacle_id in collision_pairs:
        # Change color of the obstacle involved in collision
        p.changeVisualShape(obstacle_id, -1, rgbaColor=highlight_color)

        # Optionally, highlight specific robot links (if the collision is with a particular robot link)
        # For simplicity, we'll just highlight the entire robot for now.
        # p.changeVisualShape(robot_id, -1, rgbaColor=highlight_color)


def control_path(
    robot_id, joint_ids, path, dt=0.1, attachments=[], width=960, height=680
):
    print("Path starts.")
    list_img = []
    for i, values in enumerate(path):
        # control_joints(robot_id, joint_ids, values)
        set_joint_positions(robot_id, values, joint_ids, attachments)
        img = p.getCameraImage(width, height, renderer=p.ER_BULLET_HARDWARE_OPENGL)

        rgb_array = img[2]  # This contains RGBA values (4 channels)
        rgb_array = np.reshape(rgb_array, (height, width, 4))  # 4 channels: R, G, B, A
        rgb_image = rgb_array[:, :, :3]  # Keep only R, G, B channels
        image = Image.fromarray(rgb_image.astype(np.uint8))

        list_img.append(image)
        # p.stepSimulation()
        time.sleep(dt)
    print("Path ends.")
    return list_img


def play_control(
    robot_id,
    joint_ids,
    list_control,
    dt=0.05,
    width=960,
    height=680,
    folder_path="animation_play",
):

    # Create the folder if it doesn't exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    print("Control play starts.")
    image_files = []
    attachments = []
    list_img = []
    for i, (type, value) in enumerate(list_control):

        if type == CONTRL_PATH:
            imgs = control_path(
                robot_id, joint_ids, value, dt, attachments, width, height
            )
            list_img.extend(imgs)
        elif type == CONTRL_ATTACH:
            attachments = value
            for a in attachments:
                a.enable_constraint()
        elif type == CONTRL_DETTACH:
            for a in attachments:
                a.disable_constraint()
            attachments = []

        # img = p.getCameraImage(width, height, renderer=p.ER_BULLET_HARDWARE_OPENGL)
        # rgb_array = img[2]  # Index 2 contains the RGB data
        # pil_img = Image.fromarray(rgb_array)

        # # Save the image to the folder
        # img_file_path = os.path.join(folder_path, f"image_{i + 1}.png")
        # pil_img.save(img_file_path)
        # image_files.append(img_file_path)

        # p.stepSimulation()
        # time.sleep(dt)

    # images_for_gif = [Image.fromarray(img) for img in list_img]

    # # Step 3: Generate a GIF from the saved images
    # images_for_gif = [imageio.imread(image_file) for image_file in image_files]
    gif_path = os.path.join(folder_path, "animated.gif")

    # Save the GIF (duration is in milliseconds, loop=0 means infinite loop)
    imageio.mimsave(gif_path, list_img, duration=100 * dt, loop=0)

    print("Control play ends.")


def action_place(robot_id, obj_id, region_id, decision, list_obstacle, attachment):
    # decision (subgoal): pose of obj

    list_q = []
    list_control = []

    q_init = get_joint_positions(robot_id)
    pose_obj_init = get_pose(obj_id)
    list_q.append(q_init)

    list_down_dir = gen_down_dir()

    down_dir_idx, x, y, w = decision
    pose_obj = sample_placement_on_aabb_seed(
        obj_id, get_aabb(region_id), (x, y, w), top_pose=list_down_dir[down_dir_idx]
    )

    set_pose(obj_id, pose_obj)
    list_obstacle = list(set(list_obstacle) - set([obj_id]))

    check_body_pairs = list(
        product([obj_id], list_obstacle)
    )  # Pairs of bodies to check for collisions
    for body1, body2 in check_body_pairs:
        if pairwise_collision(body1, body2):
            highlight_collision_pairs([(body1, body2)])
            print("Failed to place obj ", (body1, body2))
            return None

    rel_pose_retreat = gen_tform(point=[0, 0, -0.035])
    rel_pose_hover = gen_tform(point=[0, 0, 0.015])

    ee_pose_land = np.dot(
        tform_from_pose(pose_obj), tform_from_pose(attachment.grasp_pose)
    )
    ee_pose_retreat = np.dot(ee_pose_land, rel_pose_retreat)
    ee_pose_hover = np.dot(rel_pose_hover, ee_pose_land)

    # hover
    q_hover, ik_info = ik_solver(
        robot_id, ee_pose_hover, list_obstacle, q0=q_init, attachments=[attachment]
    )
    print("IK: q_hover", q_hover)
    print(ik_info)
    if not q_hover:
        highlight_collision_pairs(ik_info[-1])
        return None
    list_q.append(q_hover)

    # init -> hover
    set_joint_positions(robot_id, q_init, attachments=[attachment])
    path = plan_joint_motion(
        robot_id,
        MOTION_JOINTS_FRANKA,
        q_hover[:7],
        obstacles=list_obstacle,
        attachments=[attachment],
        self_collisions=1,
        verbose=1,
    )
    print("Path: init -> hover ", len(path) if path else "Fails")
    if not path:
        return None
    list_control.append((CONTRL_PATH, path))

    # land
    q_land, ik_info = ik_solver(
        robot_id, ee_pose_land, list_obstacle, q0=q_hover, attachments=[attachment]
    )
    print("IK: q_land", q_land)
    print(ik_info)
    if not q_land:
        highlight_collision_pairs(ik_info[-1])
        return None
    list_q.append(q_land)

    # hover -> land
    set_joint_positions(robot_id, q_hover, attachments=[attachment])
    path = plan_joint_motion(
        robot_id,
        MOTION_JOINTS_FRANKA,
        q_land[:7],
        obstacles=list_obstacle,
        attachments=[attachment],
        self_collisions=1,
        verbose=1,
    )
    print("Path: hover -> land ", len(path) if path else "Fails")
    if not path:
        return None
    list_control.append((CONTRL_PATH, path))

    # dettach
    list_control.append((CONTRL_DETTACH, [attachment]))

    # retreat
    q_retreat, ik_info = ik_solver(
        robot_id, ee_pose_retreat, list_obstacle, q0=q_land, attachments=[]
    )
    print("IK: q_retreat", q_retreat)
    print(ik_info)
    if not q_land:
        highlight_collision_pairs(ik_info[-1])
        return None
    list_q.append(q_retreat)

    # land -> retreat
    set_joint_positions(robot_id, q_land)
    path = plan_joint_motion(
        robot_id,
        MOTION_JOINTS_FRANKA,
        q_retreat[:7],
        obstacles=list_obstacle,
        attachments=[attachment],
        self_collisions=1,
        verbose=1,
    )
    print("Path: land -> retreat", len(path) if path else "Fails")
    if not path:
        return None
    list_control.append((CONTRL_PATH, path))

    # set_pose(obj_id,pose_obj_init)
    # set_joint_positions(robot_id, q_init)

    print("Succeed, action_place ", len(list_control))
    return list_control


def action_pick(robot_id, obj_id, decision, list_obstacle):
    # decision (subgoal): graspingd direction 0-15

    list_q = []
    list_control = []

    q_init = get_joint_positions(robot_id)
    list_q.append(q_init)
    pose_obj_init = get_pose(obj_id)

    approach_pose = gen_tform(point=[0, 0, -0.035])
    lift_pose = gen_tform(point=[0, 0, 0.015])

    # remove_init_collision(obj_id)

    list_g = gen_obj_grasping(obj_id)
    ee_pose_grasp = list_g[decision]
    ee_pose_approach = np.dot(ee_pose_grasp, approach_pose)
    ee_pose_lift = np.dot(lift_pose, ee_pose_grasp)

    # approach
    q_approach, ik_info = ik_solver(
        robot_id, ee_pose_approach, list_obstacle, q0=q_init
    )
    print("IK: q_approach", q_approach)
    print(ik_info)
    if not q_approach:
        highlight_collision_pairs(ik_info[-1])
        return None
    list_q.append(q_approach)

    # init -> approach
    set_joint_positions(robot_id, q_init)
    path = plan_joint_motion(
        robot_id,
        MOTION_JOINTS_FRANKA,
        q_approach[:7],
        obstacles=list_obstacle,
        self_collisions=1,
        verbose=1,
    )
    print("Path: init -> approach ", len(path) if path else "Fails")
    if not path:
        return None
    list_control.append((CONTRL_PATH, path))

    # grasp
    q_grasp, ik_info = ik_solver(robot_id, ee_pose_grasp, list_obstacle, q0=q_approach)
    print("IK: q_grasp ", q_grasp)
    print(ik_info)
    if not q_grasp:
        highlight_collision_pairs(ik_info[-1])
        return None
    list_q.append(q_grasp)

    # approach -> grasp
    set_joint_positions(robot_id, q_approach)
    path = plan_joint_motion(
        robot_id,
        MOTION_JOINTS_FRANKA,
        q_grasp[:7],
        obstacles=list_obstacle,
        self_collisions=1,
        verbose=1,
    )
    print("Path: approach -> grasp ", len(path) if path else "Fails")
    if not path:
        return None
    list_control.append((CONTRL_PATH, path))

    # attach
    set_joint_positions(robot_id, q_grasp)
    attachment = get_attachement(obj_id, robot_id, EE_LINK_FRANKA)
    list_control.append((CONTRL_ATTACH, [attachment]))

    # lift
    q_lift, ik_info = ik_solver(
        robot_id, ee_pose_lift, list_obstacle, q0=q_grasp, attachments=[attachment]
    )
    print("IK: q_lift", q_lift)
    print(ik_info)
    if not q_lift:
        highlight_collision_pairs(ik_info[-1])
        return None
    list_q.append(q_lift)

    # grasp -> lift
    set_joint_positions(robot_id, q_grasp)
    path = plan_joint_motion(
        robot_id,
        MOTION_JOINTS_FRANKA,
        q_lift[:7],
        obstacles=list_obstacle,
        attachments=[attachment],
        self_collisions=1,
        verbose=1,
    )
    print("Path: grasp -> lift ", len(path) if path else "Fails")
    if not path:
        return None
    list_control.append((CONTRL_PATH, path))

    print("Succeed, action_pick")
    return list_control


def get_attachment_from_control_list(list_control):

    assert isinstance(list_control, list), "The object is not a list."

    for k, v in list(reversed(list_control)):
        if k == CONTRL_ATTACH:
            return v[0]
    return None



class Scene_regrasp1(object):
    def __init__(self):

        robot_urdf = "franka_panda/panda.urdf"
        self.robots={'panda1':p.loadURDF(robot_urdf, basePosition=(0, 0, 0.0011), useFixedBase=True)}
        

        self.fixed={'floor':p.loadURDF("plane.urdf"),
                    'shelf':p.loadSDF("kiva_shelf/model.sdf")[0]
                    }
        p.resetBasePositionAndOrientation(self.fixed['shelf'], [0, 0.9, 1.1], [0, 0, 0, 1])
        self.regions={}
        for k,v in zip(("r01", "r02", "r11", "r12", "r21", "r22", "r23"),add_regions()):
            self.regions[k]=v
        self.fixed.update(self.regions)


        self.movable={}
        for k,v in zip(("o1", "o2", "o3"),add_objects()):
            self.movable[k]=v
        

        # Load the shelf
        shelf_id = p.loadSDF("kiva_shelf/model.sdf")[0]
        p.resetBasePositionAndOrientation(shelf_id, [0, 0.9, 1.1], [0, 0, 0, 1])


        self.reset()

    
    @property
    def robot(self):
        return self.robots["panda1"]
    
    @property
    def all(self):
        dall={}
        dall.update(self.robot)
        dall.update(self.fixed)
        dall.update(self.movable)
        return dall
    
    def reset(self):

        # Robot
        gripper_opening = 0.08
        initial_joint_positions1 = [
            -0.41,
            0.71,
            -0.00,
            -1.92,
            1.95,
            1.33,
            -2.8,
            gripper_opening / 2,
            gripper_opening / 2,
        ]
        set_joint_positions(
            self.all["panda1"],
            initial_joint_positions1,
            MOTION_JOINTS_FRANKA + GRIPPER_JOINTS_FRANKA,
        )
        fix_gripper_opening(self.all["panda1"], gripper_opening)

        # Movable
        set_pose(self.all['o1'],((0.196, 0.611, 0.4501),(0, 0, 0, 1)))
        set_pose(self.all['o2'],((-0.075, 0.635, 0.7976),(
            0.020368044547872787,
            -0.6977657834774182,
            0.7157624702789898,
            -0.019802532907656954,
        )))
        set_pose(self.all['o3'],((-0.4, 0, 0.1605),(0, 0, 0, 1)))



    def get_elemetns(self):
        self.reset()
        return self.robot, self.movable_bodies, self.regions
    


    def show(self):
        time_limit = 200
        start_time = time.time()
        time_step = 1 / 240  # Small time delay
        while True:
            p.stepSimulation()

            # # Get the position and orientation of o2
            # position, orientation = p.getBasePositionAndOrientation(o3)
            # print(f"position = {position} orientation = {orientation}")

            draw_ee_frame(self.robot)
            draw_obj_frame(self.all['o1'])
            draw_obj_frame(self.all['o2'])

            # Introduce a small time delay for smooth simulation
            time.sleep(time_step)

            # Break the loop after the time limit is reached
            elapsed_time = time.time() - start_time
            if elapsed_time > time_limit:
                break

        # Disconnect from simulation
        p.disconnect()
        return



def create_scene():
    # Connect to PyBullet with GUI
    sim_id = p.connect(p.GUI)
    set_camera_view()

    # Set the path for loading assets
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    # Load a plane for the ground
    floor_id = p.loadURDF("plane.urdf")

    # Load the shelf
    shelf_id = p.loadSDF("kiva_shelf/model.sdf")[0]
    p.resetBasePositionAndOrientation(shelf_id, [0, 0.9, 1.1], [0, 0, 0, 1])

    # Load a robot (Franka Panda)
    robot_urdf = "franka_panda/panda.urdf"
    robot1_id = p.loadURDF(robot_urdf, basePosition=(0, 0, 0.0011), useFixedBase=True)

    # Set initial joint positions for the robot
    gripper_opening = 0.08
    initial_joint_positions1 = [
        -0.41,
        0.71,
        -0.00,
        -1.92,
        1.95,
        1.33,
        -2.8,
        gripper_opening / 2,
        gripper_opening / 2,
    ]

    set_joint_positions(
        robot1_id,
        initial_joint_positions1,
        MOTION_JOINTS_FRANKA + GRIPPER_JOINTS_FRANKA,
    )
    fix_gripper_opening(robot1_id, gripper_opening)

    # Add gravity
    p.setGravity(0, 0, -9.81)

    list_region = list(add_regions())

    # Add objects to the scene
    o1, o2, o3 = add_objects()

    list_obstacle = [
        o1,
        o2,
        o3,
        shelf_id,
        floor_id,
    ]
    list_obstacle.extend(list_region)

    list_obj_pose = []
    for o in list_obstacle:
        list_obj_pose.append((o, get_pose(o)))

    # Set a time limit for the simulation
    time_limit = 200
    start_time = time.time()
    time_step = 1 / 240  # Small time delay

    # path = action_pick(robot1_id, o2, 6, list_obstacle)
    # path = action_pick(robot1_id, o1, 1, list_obstacle)
    # path = action_pick(robot1_id, o3, 3, list_obstacle)

    obj = o3
    grasp_dir = 1
    region = list_region[1]

    # obj = o2
    # grasp_dir = 6

    motion_planning = 1
    if motion_planning:
        list_control = action_pick(robot1_id, obj, grasp_dir, list_obstacle)
        attachement = get_attachment_from_control_list(list_control)

        # list_control2 = action_place(
        #     robot1_id, obj, ((0.4, 0, 0.161), (0, 0, 0, 1)), list_obstacle, attachement
        # )

        list_control2 = action_place(
            robot1_id, obj, region, (1, 0.5, 0.5, 0), list_obstacle, attachement
        )

        reset_objects(list_obj_pose)
        if list_control is not None and list_control2 is not None:
            # add_fixed_constraint(obj,robot1_id,EE_LINK_FRANKA)
            play_control(robot1_id, MOTION_JOINTS_FRANKA, list_control + list_control2)
            # add_fixed_constraint(obj,robot1_id,EE_LINK_FRANKA)
        else:
            print("Actions fail.")

    # Simulation loop
    while True:
        p.stepSimulation()

        # # Get the position and orientation of o2
        # position, orientation = p.getBasePositionAndOrientation(o3)
        # print(f"position = {position} orientation = {orientation}")

        draw_ee_frame(robot1_id)
        draw_obj_frame(o1)
        draw_obj_frame(o2)

        # Introduce a small time delay for smooth simulation
        time.sleep(time_step)

        # Break the loop after the time limit is reached
        elapsed_time = time.time() - start_time
        if elapsed_time > time_limit:
            break

    # Disconnect from simulation
    p.disconnect()
    return


# Example usage
if __name__ == "__main__":

    create_scene()
