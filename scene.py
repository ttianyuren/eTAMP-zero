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
    LockRenderer,
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


class Scene:
    def __init__(self, robots, fixed, movable):
        """ """
        # Models
        self.robots = robots
        self.fixed = fixed
        self.movable = movable

        # Current states
        self.pair_to_attachment = {}

        self.reset()

    def attach(self, robot_id, obj_id, attachment):
        self.pair_to_attachment[(robot_id, obj_id)] = attachment

    def dettach(self, robot_id, obj_id):
        key = (robot_id, obj_id)
        if key in self.pair_to_attachment:
            self.pair_to_attachment.pop(key)

    @property
    def robot(self):
        if len(self.robots) == 0:
            return None
        else:
            return list(self.robots.values())[0]

    @property
    def all(self):
        dall = {}
        dall.update(self.robots)
        dall.update(self.fixed)
        dall.update(self.movable)
        return dall

    @property
    def list_obstacle(self):
        # anything other than the robot
        return list(self.fixed.values()) + list(self.movable.values())

    def reset(self):
        raise NotImplementedError("This method should be overridden by subclasses.")

    def __repr__(self):
        return type(self).__name__ + str(list(self.all.keys()))


class Scene_Panda(Scene):
    def __init__(self, gui=False, show=False):

        # for stand alone showcasing the scene
        if gui:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        """Load models below."""

        robot_urdf = "franka_panda/panda.urdf"
        robots = {
            "panda1": p.loadURDF(
                robot_urdf, basePosition=(0, 0, 0.0011), useFixedBase=True
            )
        }

        fixed = {
            "floor": p.loadURDF("plane.urdf"),
            "shelf": p.loadSDF("kiva_shelf/model.sdf")[0],
        }
        p.resetBasePositionAndOrientation(fixed["shelf"], [0, 0.9, 1.1], [0, 0, 0, 1])
        regions = {}
        for k, v in zip(
            ("r01", "r02", "r11", "r12", "r21", "r22", "r23"), add_regions()
        ):
            regions[k] = v
        fixed.update(regions)

        movable = {}
        for k, v in zip(("o1", "o2", "o3"), add_objects()):
            movable[k] = v

        p.resetBasePositionAndOrientation(fixed["shelf"], [0, 0.9, 1.1], [0, 0, 0, 1])

        """Load models above."""
        super().__init__(robots, fixed, movable)
        if show and gui:
            self._show()

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
        set_pose(self.all["o1"], ((0.196, 0.611, 0.4501), (0, 0, 0, 1)))
        set_pose(
            self.all["o2"],
            (
                (-0.075, 0.635, 0.7976),
                (
                    0.020368044547872787,
                    -0.6977657834774182,
                    0.7157624702789898,
                    -0.019802532907656954,
                ),
            ),
        )
        set_pose(self.all["o3"], ((-0.4, 0, 0.1605), (0, 0, 0, 1)))

    def get_elemetns(self):
        self.reset()
        return self.robot, self.movable_bodies, self.regions

    def _show(self):
        set_camera_view()

        time_limit = 200
        start_time = time.time()
        time_step = 1 / 240  # Small time delay
        while True:
            p.stepSimulation()

            # # Get the position and orientation of o2
            # position, orientation = p.getBasePositionAndOrientation(o3)
            # print(f"position = {position} orientation = {orientation}")

            draw_ee_frame(self.robot)
            draw_obj_frame(self.all["o1"])
            draw_obj_frame(self.all["o2"])

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

    scn = Scene_Panda(gui=True, show=True)
    # scn = Scene_Panda()
    print(scn)
