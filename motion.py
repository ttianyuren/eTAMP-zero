import os
import sys
import math
import numpy as np
import random

script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(script_dir, os.pardir))
if repo_root not in sys.path:
    sys.path.append(repo_root)
import pybullet as p
from utils.pybullet_tools.utils import (
    MAX_DISTANCE,
    get_sample_fn,
    get_distance_fn,
    get_extend_fn,
    get_collision_fn,
    get_joint_positions,
    check_initial_end,
    inverse_kinematics,
    pose_from_tform,
)
from motion_planners.rrt_connect import birrt
from motion_planners.rrt_star import rrt_star, birrt_star
from motion_planners.multi_rrt import multi_rrt
from motion_planners.smoothing import smooth_path

MOTION_JOINTS_FRANKA = [0, 1, 2, 3, 4, 5, 6]
GRIPPER_JOINTS_FRANKA = [9, 10]
EE_LINK_FRANKA = 11

CONTRL_PATH, CONTRL_ATTACH, CONTRL_DETTACH = 0, 1, 2


def get_joint_positions(robot_id, joint_ids=MOTION_JOINTS_FRANKA):
    """
    Get the current positions of the specified joints for a robot.

    Args:
    - robot_id (int): The ID of the robot in the PyBullet simulation.
    - joint_ids (list of int): List of joint IDs whose positions are to be retrieved.

    Returns:
    - joint_positions (list of float): List of current joint positions.
    """
    joint_positions = []

    for joint_id in joint_ids:
        joint_state = p.getJointState(robot_id, joint_id)
        joint_position = joint_state[0]  # Extract the joint position
        joint_positions.append(joint_position)

    return joint_positions


def set_joint_positions(
    robot_id, joint_positions, joint_ids=MOTION_JOINTS_FRANKA, attachments=[]
):
    for pos, id in zip(joint_positions, joint_ids):
        p.resetJointState(robot_id, id, pos)
    for a in attachments:
        a.assign()


def set_random_joint_positions(robot_id, joint_ids=MOTION_JOINTS_FRANKA):
    for joint_index in joint_ids:
        # Get joint info to retrieve limits
        joint_info = p.getJointInfo(robot_id, joint_index)
        joint_type = joint_info[2]

        # Skip non-revolute joints (e.g., grippers)
        assert joint_type != p.JOINT_FIXED
        # Get the joint limits
        joint_lower_limit = joint_info[8]
        joint_upper_limit = joint_info[9]

        # If limits are defined, choose a random position within limits
        if joint_lower_limit < joint_upper_limit:
            random_position = random.uniform(joint_lower_limit, joint_upper_limit)
        else:
            # If the limits are identical or reversed, use 0 as a fallback
            random_position = 0.0

        # Set the joint to the random position
        p.resetJointState(robot_id, joint_index, random_position)


def calculate_radius(gamma=10, n=10000, d=7):
    """
    Calculates the radius for RRT* algorithm based on the number of nodes.

    :param gamma: Scaling constant (typically between 5 and 15)
    :param n: Number of nodes in the tree. For very cluttered or constrained environments 10,000–50,000
    :param d: Dimensionality of the configuration space (7 for Franka Panda arm)
    :return: Radius value
    """
    if n <= 0:
        raise ValueError("Number of nodes (n) must be greater than 0.")

    # Calculate the radius using the formula
    radius = gamma * (math.log(n) / n) ** (1 / d)

    return radius


def ik_solver(
    robot_id, ee_pose, list_obstacle, num_attempts=100, q0=None, attachments=[]
):
    ee_link = EE_LINK_FRANKA
    max_distance = 0
    out_of_reach_count = 0
    collision_count = 0
    collision_pairs = []  # List to store the collision pairs

    # Generate the collision function once using get_collision_fn
    self_collisions = True  # Enable self-collision checking if required
    disabled_collisions = []  # If any collisions should be ignored
    collision_fn = get_collision_fn(
        robot_id,
        MOTION_JOINTS_FRANKA + GRIPPER_JOINTS_FRANKA,
        list_obstacle,
        attachments,
        self_collisions,
        disabled_collisions,
    )

    for i in range(num_attempts):
        if q0 and i < np.ceil(0.2 * num_attempts):
            # in the ideal case ik solution is close to q0
            set_joint_positions(robot_id, q0)
        else:
            set_random_joint_positions(robot_id)

        q = inverse_kinematics(
            robot_id, ee_link, pose_from_tform(ee_pose)
        )  # IK solution

        if not q:
            out_of_reach_count += 1
            continue

        # Check for collisions using the generated collision function
        flag_collision, collision_pair = collision_fn(q, True)
        if flag_collision:
            collision_count += 1
            collision_pairs.append(
                collision_pair
            )  # Append robot_id (you can customize this to append more info)
            continue

        # If no collision, return the valid configuration
        return q, ("success", out_of_reach_count, collision_count, collision_pairs)

    # Handling various failure cases
    if out_of_reach_count == num_attempts:
        return None, (
            "out_of_reach",
            out_of_reach_count,
            collision_count,
            collision_pairs,
        )

    if collision_count > 0:
        return None, ("collision", out_of_reach_count, collision_count, collision_pairs)

    return None, (
        "unknown_failure",
        out_of_reach_count,
        collision_count,
        collision_pairs,
    )


def plan_joint_motion(
    robot,
    joints,
    end_conf,
    obstacles=[],
    attachments=[],
    self_collisions=True,
    disabled_collisions=set(),
    weights=[2, 2, 1.0, 1.0, 0.5, 0.5, 0.2],
    resolutions=0.05,
    max_distance=MAX_DISTANCE,
    custom_limits={},
    max_iterations=100,
    restarts=200,
    smoothing=True,
    **kwargs
):

    assert len(joints) == len(end_conf)
    sample_fn = get_sample_fn(robot, joints, custom_limits=custom_limits)
    distance_fn = get_distance_fn(robot, joints, weights=weights)
    extend_fn = get_extend_fn(robot, joints, resolutions=resolutions)
    collision_fn = get_collision_fn(
        robot,
        joints,
        obstacles,
        attachments,
        self_collisions,
        disabled_collisions,
        custom_limits=custom_limits,
        max_distance=max_distance,
    )

    start_conf = get_joint_positions(robot, joints)

    if not check_initial_end(start_conf, end_conf, collision_fn):
        return None

    # return multi_rrt(start_conf,
    #     end_conf,
    #     distance_fn,
    #     sample_fn,
    #     extend_fn,
    #     collision_fn,
    #     max_time=60)

    path = birrt(
        start_conf,
        end_conf,
        distance_fn,
        sample_fn,
        extend_fn,
        collision_fn,
        restarts=restarts,
        max_iterations=max_iterations,
    )

    if smoothing:
        path = smooth_path(path, extend_fn, collision_fn, distance_fn)

    return path

    radius = calculate_radius(gamma=10, n=20000)
    return rrt_star(
        start_conf,
        end_conf,
        distance_fn,
        sample_fn,
        extend_fn,
        collision_fn,
        radius,
        max_time=60,
    )

    # return birrt_star(start_conf,
    #     end_conf,
    #     distance_fn,
    #     sample_fn,
    #     extend_fn,
    #     collision_fn,
    #     radius,max_time=60)

    # return plan_lazy_prm(
    #     start_conf,
    #     end_conf,
    #     sample_fn,
    #     extend_fn,
    #     collision_fn,
    #     # restarts=restarts,
    #     # max_iterations=max_iterations,
    # )
