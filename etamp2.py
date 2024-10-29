import os, sys
from unittest import result
from anytree import Node, RenderTree
import re
from collections import namedtuple, defaultdict, Counter
import numpy as np
from sympy import im

script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(script_dir, os.pardir))
if repo_root not in sys.path:
    sys.path.append(repo_root)

from action_imp import action_imp_from_str
from scene import Scene_Panda
from tree_search import Node_action, build_extended_tree
from connect_topk import run_symk
from num_tools import play_control
from sym_tools import save_data


def etamp(
    path_domain,
    path_problem,
    dir_plans,
    num_plan,
    num_search,
    pddl_to_action_imp,
    scene,
    return_first=True,
):

    path_sk_plan = dir_plans + "/sk"
    result_files_sorted = run_symk(
        path_domain,
        path_problem,
        path_sk_plan,
        num_plan=num_plan,
    )

    if len(result_files_sorted) == 0:
        return None

    node_root = build_extended_tree(dir_plans, pddl_to_action_imp, scene)
    node_root.print()

    solutions = []

    for i in range(num_search):
        print("iter", i)
        flag, path_of_node = node_root.search()
        if flag is True:
            print("Successful sk:", path_of_node)
            list_control = []
            for n in path_of_node:
                list_control.extend(n.get_control_cmd())
            solutions.append(list_control)
            if return_first:
                break
        elif flag is False:
            print(path_of_node)
            continue
        elif flag is None:
            print("Unsolvable.", i)
            break
        else:
            print("Unknown error.", i)
            break

    scene.reset()
    return solutions


if __name__ == "__main__":

    dir_path = os.path.dirname(os.path.realpath(__file__))
    path_domain = dir_path + "/pddl/domain.pddl"
    path_problem = dir_path + "/pddl/problem.pddl"
    dir_sk = dir_path + "/sk_plans"

    scene_class=Scene_Panda
    
    scene = scene_class(gui=1)
    solutions = etamp(
        path_domain,
        path_problem,
        dir_sk,
        20,
        10,
        action_imp_from_str,
        scene,
        return_first=0,
    )
    
    save_data((scene_class,solutions),"scene_class_solutions.pk")

    # if len(solutions) > 0:
    #     for list_control in solutions:
    #         play_control(scene, list_control)
    #         scene.reset()
