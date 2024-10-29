import os
from re import S
import sys
from tabnanny import verbose
import itertools

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
from scene import Scene_Panda
from utils.pybullet_tools.utils import (
    Pose,
    pairwise_collision,
    set_pose,
    get_attachement,
    get_pose,
    sample_placement_on_aabb_seed,
    get_aabb,
    HideOutput,
    LockRenderer,
)
from sym_tools import parse_action_string, call_without_printing

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


def action_imp_from_str(action_str):
    """
    action_str: (place o1 r3)
    """
    pddl_to_imp = {
        "pick": Action_Pick,
        "place": Action_Place,
        "push": None,
        "pull": None,
        "open": None,
        "close": None,
        "pivot": None,
        "pour": None,
    }

    return pddl_to_imp[parse_action_string(action_str)[0]]



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
    for objs in collision_pairs:
        # Change color of the obstacle involved in collision

        p.changeVisualShape(objs[-1], -1, rgbaColor=highlight_color)

        # Optionally, highlight specific robot links (if the collision is with a particular robot link)
        # For simplicity, we'll just highlight the entire robot for now.
        # p.changeVisualShape(robot_id, -1, rgbaColor=highlight_color)


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

    if pose_obj is None:
        return None

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
    if not q_retreat:
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
    return (robot_id, obj_id, q_retreat, pose_obj), list_control


def action_pick(robot_id, obj_id, decision, list_obstacle):
    # decision (subgoal): graspingd direction 0-15

    grasp_dir = decision[0]

    list_q = []
    list_control = []

    q_init = get_joint_positions(robot_id)
    list_q.append(q_init)
    pose_obj_init = get_pose(obj_id)

    approach_pose = gen_tform(point=[0, 0, -0.035])
    lift_pose = gen_tform(point=[0, 0, 0.015])

    # remove_init_collision(obj_id)

    list_g = gen_obj_grasping(obj_id)
    ee_pose_grasp = list_g[grasp_dir]
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
    return (robot_id, obj_id, q_lift, attachment), list_control


def get_attachment_from_control_list(list_control):

    assert isinstance(list_control, list), "The object is not a list."

    for k, v in list(reversed(list_control)):
        if k == CONTRL_ATTACH:
            return v[0]
    return None


def get_discrete_combinations(element_ranges, is_discrete):
    """
    :return: List of tuples containing all possible combinations of the discrete elements.
    """
    # Filter out only the discrete ranges based on is_discrete
    discrete_ranges = [
        element_ranges[i] for i in range(len(is_discrete)) if is_discrete[i]
    ]

    # Generate all possible combinations of the discrete dimensions
    all_possible_elements = list(
        itertools.product(*[range(r[0], r[1] + 1) for r in discrete_ranges])
    )

    return all_possible_elements


def voronoi_sample(element_ranges, is_discrete, is_circular, existing_points):
    """
    Performs Voronoi sampling for both discrete and continuous dimensions, considering circular continuous spaces.
    :return: A new subgoal sampled using Voronoi sampling principles.
    """

    dimension = len(element_ranges)
    new_point = []

    # For each dimension, either sample discrete or continuous values
    for i in range(dimension):
        if is_discrete[i]:
            # For discrete dimensions, sample a new element that's not part of existing subgoals
            existing_discrete_vals = {subgoal[i] for subgoal in existing_points}
            possible_vals = list(range(element_ranges[i][0], element_ranges[i][1] + 1))
            remaining_vals = [
                val for val in possible_vals if val not in existing_discrete_vals
            ]

            if remaining_vals:
                new_discrete_val = random.choice(
                    remaining_vals
                )  # Randomly pick a new unvisited discrete value
            else:
                new_discrete_val = random.choice(
                    possible_vals
                )  # If all are visited, pick any randomly

            new_point.append(new_discrete_val)

        else:
            # For continuous dimensions, use Voronoi-like sampling to maximize coverage
            if len(existing_points) == 0:
                # If no subgoals exist, pick a random point in the range
                new_continuous_val = random.uniform(
                    element_ranges[i][0], element_ranges[i][1]
                )
            else:
                # Find the largest "gap" from previously sampled subgoals
                existing_continuous_vals = np.array(
                    [subgoal[i] for subgoal in existing_points]
                )
                new_continuous_val = voronoi_sample_continuous(
                    existing_continuous_vals, element_ranges[i], is_circular[i]
                )

            new_point.append(new_continuous_val)

    return tuple(new_point)


def voronoi_sample_continuous(existing_vals, value_range, is_circular):
    """
    Performs Voronoi-like sampling in a continuous dimension by finding the largest gap between existing values.
    :return: The new value to sample in the continuous space.
    """
    if len(existing_vals) == 0:
        return random.uniform(value_range[0], value_range[1])

    # Sort the existing values
    sorted_vals = np.sort(existing_vals)

    if is_circular:
        # Circular case: Treat the range as wrapping around, so consider the gap between max and min
        gaps = np.diff(
            np.concatenate(
                ([sorted_vals[-1] - (value_range[1] - value_range[0])], sorted_vals)
            )
        )
        largest_gap_idx = np.argmax(gaps)

        if largest_gap_idx == 0:
            # The gap between max and min (circular wrap)
            mid_point = (sorted_vals[-1] + 0.5 * gaps[0]) % (
                value_range[1] - value_range[0]
            )
        else:
            # The largest gap between consecutive values
            mid_point = (
                sorted_vals[largest_gap_idx - 1] + sorted_vals[largest_gap_idx]
            ) / 2
    else:
        # Non-circular case: Find largest gap between consecutive values
        extended_sorted_vals = np.concatenate(
            ([value_range[0]], sorted_vals, [value_range[1]])
        )
        gaps = np.diff(extended_sorted_vals)

        largest_gap_idx = np.argmax(gaps)
        mid_point = (
            extended_sorted_vals[largest_gap_idx]
            + extended_sorted_vals[largest_gap_idx + 1]
        ) / 2

    return mid_point


class Action:
    def __init__(
        self, scene, pddl_parameters, element_ranges, is_discrete, is_circular=None
    ):
        """
        Initializes the decision space for the action.
        :param element_ranges: List of tuples representing the range of each element.
                              For example, [(0, 1), (0, 5)] for two dimensions.
        :param is_discrete: List of booleans indicating whether each dimension is discrete.
        :param is_circular: Optional list of booleans indicating whether each dimension is circular (only for continuous).
        """
        self.scene = scene
        self.pddl_parameters = pddl_parameters
        assert len(element_ranges) == len(is_discrete)
        self.dimension = len(element_ranges)
        self.element_ranges = element_ranges
        self.is_discrete = is_discrete
        self.is_circular = is_circular if is_circular else [False] * self.dimension

    def all_explored(self, existing_subgoals):
        """
        Check if all possible subgoals have been explored.
        :param existing_subgoals: A list of previously explored subgoals.
        :return: True if all subgoals have been explored, False otherwise.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def sample_subgoal(self, existing_subgoals):
        """
        Sample a new subgoal that hasn't been explored yet.
        :param existing_subgoals: A list of previously explored subgoals.
        :return: A new subgoal to explore.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def motion_from_subgoal(self, subgoal):
        """
        Generate a grounded motion plan based on the given subgoal.
        :param subgoal: The subgoal to use for generating motion.
        :return: A motion plan.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def set_world(self, saved_world):
        """
        Set the current world state using a saved world configuration.
        :param saved_world: The saved world configuration to restore.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def __repr__(self):
        return type(self).__name__ + str(self.pddl_parameters)


class Action_Pick(Action):
    def __init__(self, scene, pddl_parameters):
        """
        scene is an instance of class
        """
        element_ranges = [(0, 15)]  # 16 possible grasp directions
        is_discrete = [True]
        is_circular = [False]
        super().__init__(
            scene, pddl_parameters, element_ranges, is_discrete, is_circular
        )

    def all_explored(self, existing_subgoals):
        all_cases = get_discrete_combinations(self.element_ranges, self.is_discrete)
        return len(set(existing_subgoals)) >= len(all_cases)

    def sample_subgoal(self, existing_subgoals):
        # Logic to sample a new 'pick' subgoal that hasn't been explored yet.
        return voronoi_sample(
            self.element_ranges, self.is_discrete, self.is_circular, existing_subgoals
        )

    def motion_from_subgoal(self, subgoal):
        """
        :return: subgoal_world, list_control
        """
        obj_id = self.scene.all[self.pddl_parameters[0]]
        robot_id = self.scene.robot
        result = action_pick(robot_id, obj_id, subgoal, self.scene.list_obstacle)
        if result is None:
            return None, None
        else:
            (robot_id, obj_id, q_lift, attachment), _ = result
            self.scene.attach(robot_id, obj_id, attachment)
            return result

    def set_world(self, saved_world):
        (robot_id, obj_id, q_lift, attachment) = saved_world
        set_joint_positions(robot_id, q_lift, attachments=[attachment])
        self.scene.attach(robot_id, obj_id, attachment)


class Action_Place(Action):
    def __init__(self, scene, pddl_parameters):
        element_ranges = [
            (0, 3),
            (0, 1),
            (0, 1),
            (0, 1),
        ]  # 16 possible grasp directions
        is_discrete = [True, False, False, False]
        is_circular = [False, False, False, True]
        super().__init__(
            scene, pddl_parameters, element_ranges, is_discrete, is_circular
        )

    def all_explored(self, existing_subgoals):
        all_cases = get_discrete_combinations(self.element_ranges, self.is_discrete)
        return len(set(existing_subgoals)) >= len(all_cases)

    def sample_subgoal(self, existing_subgoals):
        # Logic to sample a new 'pick' subgoal that hasn't been explored yet.
        return voronoi_sample(
            self.element_ranges, self.is_discrete, self.is_circular, existing_subgoals
        )

    def motion_from_subgoal(self, subgoal):
        """
        :return: subgoal_world, list_control
        (robot_id, obj_id, region_id, decision, list_obstacle, attachment)
        """
        obj_id = self.scene.all[self.pddl_parameters[0]]
        region_id = self.scene.all[self.pddl_parameters[1]]
        robot_id = self.scene.robot
        attachment = self.scene.pair_to_attachment[(robot_id, obj_id)]
        result = action_place(
            robot_id,
            obj_id,
            region_id,
            subgoal,
            self.scene.list_obstacle,
            attachment,
        )
        if result is None:
            return None, None
        else:
            self.scene.dettach(robot_id, obj_id)
            return result

    def set_world(self, saved_world):
        (robot_id, obj_id, q_retreat, pose_obj) = saved_world
        set_joint_positions(robot_id, q_retreat)
        set_pose(obj_id, pose_obj)
        self.scene.dettach(robot_id, obj_id)


# Example usage
if __name__ == "__main__":

    scene = Scene_Panda(gui=True)

    # obj = scn.all['o3']
    # grasp_dir = 1
    # region = scn.all['r02']

    a_pick = Action_Pick(scene, ("o3",))
    result = a_pick.motion_from_subgoal((1,))
    print("a_pick", result[0])

    if result is None:
        exit()

    a_place = Action_Place(scene, ("o3", "r02"))
    result = a_place.motion_from_subgoal((1, 0.5, 0.5, 1))
    print("a_place", result[0])

    print(a_pick)
