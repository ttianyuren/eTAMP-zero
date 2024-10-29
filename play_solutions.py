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
from sym_tools import load_data


if __name__ == "__main__":

    d = load_data("scene_class_solutions.pk")

    if d is None:
        print("Cannot find file.")
        exit()

    scene_class, solutions = d
    scene = scene_class(gui=1)

    if len(solutions) > 0:
        for i,list_control in enumerate(solutions):
            print("solution",i)
            play_control(scene, list_control)
            scene.reset()
