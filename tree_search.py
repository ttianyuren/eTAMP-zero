import os, sys
from anytree import Node, RenderTree
from collections import namedtuple, defaultdict, Counter
import numpy as np
from sympy import im

script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(script_dir, os.pardir))
if repo_root not in sys.path:
    sys.path.append(repo_root)

from action_imp import action_imp_from_str
from scene import Scene_Panda
from sym_tools import parse_action_string, call_without_printing

Subgoal = namedtuple("Subgoal", ["target", "visits", "achievable"])


def get_distances_to_leaves(node, current_depth=0):
    """
    Recursively calculates the distances from a given node to all its leaf nodes.

    :param node: The starting node
    :param current_depth: The current depth of the traversal (starts at 0)
    :return: A list of distances to each leaf node
    """
    # If the node is a leaf, return the current depth (distance)
    if node.is_leaf:
        return [current_depth]

    # Otherwise, recursively calculate the distances for all children
    distances = []
    for child in node.children:
        distances.extend(get_distances_to_leaves(child, current_depth + 1))

    return distances


def cal_ucb_value(reward, visits, total_visits, exploration_constant=1.0):
    # reward: the best reward of that choice so far, for optimistic exploration
    assert visits >= 1
    exploration_term = exploration_constant * np.sqrt(2 * np.log(total_visits) / visits)

    return reward + exploration_term


def cal_subgoal_to_ucb(subgoal_to_visits, subgoal_to_reward, exploration_constant=1.0):

    total_visits = sum(subgoal_to_visits.values())
    subgoal_to_ucb = {}

    for subgoal, visits in subgoal_to_visits.items():
        assert visits > 0
        subgoal_to_ucb[subgoal] = cal_ucb_value(
            subgoal_to_reward[subgoal], visits, total_visits, exploration_constant
        )

    return subgoal_to_ucb


def backpropage_reward(node, reward):
    if node.is_root:
        return
    else:
        node.parent.child_to_subgoals[node].append(
            (node.parent.current_subgoal, reward)
        )
        backpropage_reward(node.parent, reward)


def get_subgoal_reward(list_tuple):
    list_subgoal = [v[0] for v in list_tuple]
    list_reward = [v[1] for v in list_tuple]
    return list_subgoal, list_reward


class Node_action(Node):
    """
    Custom Node class that extends anytree's Node.
    Initializes the node with the parsed PDDL action and parameters.
    """

    def __init__(self, scene, action_str, action_imp=None, parent=None):
        """
        action_imp is a class
        """

        # Initialize the base Node class
        super().__init__(action_str, parent)

        self.pw_const = 13.5
        self.explr_const_node = 5
        self.explr_const_subgoal = 2

        # Parse the action string into PDDL action and parameters
        self.pddl_action, self.pddl_parameters = parse_action_string(action_str)
        self.visits = 0
        if action_imp is not None:
            self.action_imp = action_imp(scene, self.pddl_parameters)

        self.current_subgoal = None
        self.is_dead = False

        # key: node, value: list of (subgoal,reward), reward <- None
        self.child_to_subgoals = defaultdict(list)
        self.subgoal_to_control = {}  # -> list_constrol
        self.all_subgoals = []
        self.subgoal_to_world = {}  # -> saved_world of subgoal

        self.scene = scene

    @property
    def min_dist_to_leaf(self):
        return min(get_distances_to_leaves(self))

    @property
    def value(self):
        if self.is_dead:
            return -np.inf
        len_plan = self.min_dist_to_leaf + self.depth
        return -len_plan

    @property
    def ucb_value(self):
        return cal_ucb_value(
            self.value, self.visits + 1, self.parent.visits, self.explr_const_node
        )

    def get_subgoal_info(self, next_action_node):

        list_subgoal, list_reward = get_subgoal_reward(
            self.child_to_subgoals[next_action_node]
        )
        if len(list_subgoal) == 0:
            return True, False, {}

        subgoal_to_visits = Counter(list_subgoal)
        all_explored = self.action_imp.all_explored(subgoal_to_visits.keys())

        # if all_explored and max(list_reward) == -np.inf:
        #     return None, None, None

        subgoal_to_reward = {}
        for key, value in zip(list_subgoal, list_reward):
            if key not in subgoal_to_reward:
                subgoal_to_reward[key] = value  # If key is not in the dict, add it
            else:
                subgoal_to_reward[key] = max(
                    subgoal_to_reward[key], value
                )  # Update with max value

        subgoal_to_ucb = cal_subgoal_to_ucb(
            subgoal_to_visits, subgoal_to_reward, self.explr_const_subgoal
        )

        total_visits = sum(subgoal_to_visits.values())
        total_subgoal = len(subgoal_to_visits)
        flag_pw = total_visits >= self.pw_const * (total_subgoal**2)

        return flag_pw, all_explored, subgoal_to_ucb

    def decide_subgoal(self, next_action_node=None):
        # by PW-UCB
        # return: flag_successful

        flag_pw, all_explored, subgoal_to_ucb = self.get_subgoal_info(next_action_node)
        assert not (not flag_pw and len(subgoal_to_ucb) == 0)
        force_expand = (
            len(subgoal_to_ucb) > 0 and max(subgoal_to_ucb.values()) == -np.inf
        )

        if (flag_pw or force_expand) and not all_explored:
            self.current_subgoal = self.action_imp.sample_subgoal(subgoal_to_ucb.keys())
            subgoal_world, list_control = call_without_printing(
                self.action_imp.motion_from_subgoal, self.current_subgoal
            )

            if subgoal_world is None:
                parent_reward = -self.min_dist_to_leaf - 1
                self.child_to_subgoals[next_action_node].append(
                    (self.current_subgoal, -np.inf)
                )
                backpropage_reward(self, parent_reward)
                return None
            else:
                self.subgoal_to_world[self.current_subgoal] = subgoal_world
                self.subgoal_to_control[self.current_subgoal] = list_control
        else:
            self.current_subgoal = max(subgoal_to_ucb, key=subgoal_to_ucb.get)
            if (
                self.current_subgoal is None
                or subgoal_to_ucb[self.current_subgoal] == -np.inf
            ) and all_explored:
                self.is_dead = True
                return None
            self.action_imp.set_world(self.subgoal_to_world[self.current_subgoal])

        return self.current_subgoal

    def decide_next_action(self):
        # by UCB
        if not self.children:
            return None  # No children to choose from
        best_child = max(self.children, key=lambda child: child.ucb_value)
        if best_child is None:
            self.is_dead = True

        return best_child

    def search(self):
        self.visits += 1

        if self.is_leaf:
            subgoal = self.decide_subgoal()
            if subgoal is None:
                return False, self.path[1:]
            else:
                return True, self.path[1:]
        elif self.is_root:
            next_action_node = self.decide_next_action()
            self.scene.reset()
            if next_action_node is None:
                print("The tree is dead.")
                return None, self.path[1:]
        else:
            next_action_node = self.decide_next_action()
            if next_action_node is None:
                return False, self.path[1:]
            subgoal = self.decide_subgoal(next_action_node)
            if subgoal is None:
                return False, self.path[1:]
        return next_action_node.search()

    def print(self):
        """Prints the tree in an easy-to-read format, highlighting leaf nodes with plan lengths."""
        for pre, _, node in RenderTree(self):
            if node.is_leaf:
                print(f"{pre}{node.name} (Plan length: {node.depth})")
            else:
                print(f"{pre}{node.name}")

    def get_control_cmd(self):
        """
        return a list of low-level control for robot execution
        """
        if (
            self.current_subgoal is not None
            and self.current_subgoal in self.subgoal_to_control
        ):
            return self.subgoal_to_control[self.current_subgoal]
        return None

    def __repr__(self):
        return self.name + "@" + str(self.depth)


def read_plans_from_folder(folder_path):
    """Reads all files from the given folder and extracts the plans."""
    plans = {}
    for file_name in os.listdir(folder_path):
        if file_name.startswith("sk"):
            with open(os.path.join(folder_path, file_name), "r") as file:
                plan_actions = [
                    line.strip()
                    for line in file.readlines()
                    if not line.startswith(";")
                ]
                if plan_actions:  # Only consider non-empty plans
                    plans[file_name] = plan_actions
    return plans


def filter_plans_by_last_action(plans, reference_plan):
    """
    Filters out plans where the last action of the reference plan:
    - Appears more than once.
    - Appears in any position other than the last.
    """
    reference_last_action = reference_plan[-1]
    filtered_plans = {}

    for name, actions in plans.items():
        # Check how many times the reference last action appears in the plan
        action_count = actions.count(reference_last_action)

        if action_count == 1 and actions[-1] == reference_last_action:
            # Only keep the plan if the reference last action appears exactly once and is the last action
            filtered_plans[name] = actions

    return filtered_plans


def find_child_with_action(current_node, action):
    """Find a direct child of the current node that matches the action."""
    for child in current_node.children:
        if child.name == action:
            return child
    return None


def build_tree_from_plans(plans, action_imp_from_str, scene):
    """Merges the remaining plans into a tree structure."""
    root = Node_action(scene, "Root", None, parent=None)

    for plan in plans.values():
        current_node = root
        for i, action_str in enumerate(plan):
            # Search only among direct children of the current node
            existing_child = find_child_with_action(current_node, action_str)
            if existing_child:
                current_node = existing_child
            else:
                # If not found, create a new child node
                # new_node = Node(action_str, parent=current_node)
                new_node = Node_action(
                    scene,
                    action_str,
                    action_imp_from_str(action_str),
                    parent=current_node,
                )
                pddl_action, pddl_parameters = parse_action_string(action_str)
                new_node.pddl_action = pddl_action
                new_node.pddl_parameters = pddl_parameters

                current_node = new_node

    return root


def build_extended_tree(folder_path, action_imp_from_str, scene):
    # Step 1: Read all plan files from the folder
    plans = read_plans_from_folder(folder_path)

    if not plans:
        print("No plans found in the folder.")
        return

    # Step 2: Delete plans that have the last action of the first plan in a non-last place
    first_plan_key = sorted(plans.keys())[0]
    reference_plan = plans[first_plan_key]
    filtered_plans = filter_plans_by_last_action(plans, reference_plan)

    # Step 3: Merge the remaining plans
    merged_tree_root = build_tree_from_plans(filtered_plans, action_imp_from_str, scene)

    return merged_tree_root


# Provide the folder path containing the plan files
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    folder_path = script_dir + "/sk_examples"  # Replace with the actual folder path

    scene = Scene_Panda(gui=1)

    node_root = build_extended_tree(folder_path, action_imp_from_str, scene)
    node_root.print()

    for i in range(100):
        result = node_root.search()
        if isinstance(result, tuple):
            print("eTAMP succeeds.", i)
            print(result)
            break
        elif result is False:
            print("iter", i)
        elif result is None:
            print("unsolvable.", i)
            break
        elif result is None:
            print("unknown output.", i)
            break
