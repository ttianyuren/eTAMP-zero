from __future__ import print_function

from random import random,choice
from time import time

from .utils import INF, argmin, elapsed_time, BLUE, RED, apply_alpha

EPSILON = 1e-6
PRINT_FREQUENCY = 100

class OptimalNode(object):

    def __init__(self, config, parent=None, d=0, path=[], iteration=None):
        self.config = config
        self.parent = parent
        self.children = set()
        self.d = d
        self.path = path
        if parent is not None:
            self.cost = parent.cost + d
            self.parent.children.add(self)
        else:
            self.cost = d
        self.solution = False
        self.creation = iteration
        self.last_rewire = iteration

    def set_solution(self, solution):
        if self.solution is solution:
            return
        self.solution = solution
        if self.parent is not None:
            self.parent.set_solution(solution)

    def retrace(self):
        if self.parent is None:
            return self.path + [self.config]
        return self.parent.retrace() + self.path + [self.config]

    def rewire(self, parent, d, path, iteration=None):
        if self.solution:
            self.parent.set_solution(False)
        self.parent.children.remove(self)
        self.parent = parent
        self.parent.children.add(self)
        if self.solution:
            self.parent.set_solution(True)
        self.d = d
        self.path = path
        self.update()
        self.last_rewire = iteration

    def update(self):
        self.cost = self.parent.cost + self.d
        for n in self.children:
            n.update()

    def clear(self):
        self.node_handle = None
        self.edge_handle = None

    def draw(self, env):
        # https://github.mit.edu/caelan/lis-openrave
        from manipulation.primitives.display import draw_node, draw_edge
        color = apply_alpha(BLUE if self.solution else RED, alpha=0.5)
        self.node_handle = draw_node(env, self.config, color=color)
        if self.parent is not None:
            self.edge_handle = draw_edge(
                env, self.config, self.parent.config, color=color)

    def __str__(self):
        return self.__class__.__name__ + '(' + str(self.config) + ')'
    __repr__ = __str__


def safe_path(sequence, collision):
    path = []
    for q in sequence:
        if collision(q):
            break
        path.append(q)
    return path

##################################################

def rrt_star(start, goal, distance_fn, sample_fn, extend_fn, collision_fn, radius,
             max_time=INF, max_iterations=INF, goal_probability=.2, informed=True):
    """
    :param start: Start configuration - conf
    :param goal: End configuration - conf
    :param distance_fn: Distance function - distance_fn(q1, q2)->float
    :param sample_fn: Sample function - sample_fn()->conf
    :param extend_fn: Extension function - extend_fn(q1, q2)->[q', ..., q"]
    :param collision_fn: Collision function - collision_fn(q)->bool
    :param max_time: Maximum runtime - float
    :return: Path [q', ..., q"] or None if unable to find a solution
    """
    if collision_fn(start) or collision_fn(goal):
        return None
    nodes = [OptimalNode(start)]
    goal_n = None
    start_time = time()
    iteration = 0
    while (elapsed_time(start_time) < max_time) and (iteration < max_iterations):
        do_goal = goal_n is None and (iteration == 0 or random() < goal_probability)
        s = goal if do_goal else sample_fn()
        # Informed RRT*
        if informed and (goal_n is not None) and (distance_fn(start, s) + distance_fn(s, goal) >= goal_n.cost):
            continue
        if iteration % PRINT_FREQUENCY == 0:
            success = goal_n is not None
            cost = goal_n.cost if success else INF
            print('Iteration: {} | Time: {:.3f} | Success: {} | {} | Cost: {:.3f}| Nodes: {}'.format(
                iteration, elapsed_time(start_time), success, do_goal, cost,len(nodes)))
        iteration += 1

        nearest = argmin(lambda n: distance_fn(n.config, s), nodes)
        path = safe_path(extend_fn(nearest.config, s), collision_fn)
        if len(path) == 0:
            continue
        new = OptimalNode(path[-1], parent=nearest, d=distance_fn(
            nearest.config, path[-1]), path=path[:-1], iteration=iteration)
        # if safe and do_goal:
        if do_goal and (distance_fn(new.config, goal) < EPSILON):
            goal_n = new
            goal_n.set_solution(True)
            return goal_n.retrace()
        # TODO - k-nearest neighbor version
        neighbors = filter(lambda n: distance_fn(n.config, new.config) < radius, nodes)
        nodes.append(new)

        # TODO: smooth solution once found to improve the cost bound
        for n in neighbors:
            d = distance_fn(n.config, new.config)
            if (n.cost + d) < new.cost:
                path = safe_path(extend_fn(n.config, new.config), collision_fn)
                if (len(path) != 0) and (distance_fn(new.config, path[-1]) < EPSILON):
                    new.rewire(n, d, path[:-1], iteration=iteration)
        for n in neighbors:  # TODO - avoid repeating work
            d = distance_fn(new.config, n.config)
            if (new.cost + d) < n.cost:
                path = safe_path(extend_fn(new.config, n.config), collision_fn)
                if (len(path) != 0) and (distance_fn(n.config, path[-1]) < EPSILON):
                    n.rewire(new, d, path[:-1], iteration=iteration)
    if goal_n is None:
        return None
    return goal_n.retrace()

def informed_rrt_star(start, goal, distance_fn, sample_fn, extend_fn, collision_fn, radius, **kwargs):
    return rrt_star(start, goal, distance_fn, sample_fn, extend_fn, collision_fn, radius, informed=True, **kwargs)




def birrt_star(start, goal, distance_fn, sample_fn, extend_fn, collision_fn, radius,
               max_time=float('inf'), max_iterations=float('inf'), goal_probability=0.2, informed=True):
    """
    Bi-directional RRT* implementation for motion planning.

    :param start: Start configuration
    :param goal: Goal configuration
    :param distance_fn: Distance function (distance_fn(q1, q2) -> float)
    :param sample_fn: Sampling function (sample_fn() -> config)
    :param extend_fn: Extension function (extend_fn(q1, q2) -> [q', ..., q"])
    :param collision_fn: Collision checking function (collision_fn(q) -> bool)
    :param radius: Connection radius for rewiring
    :param max_time: Maximum runtime in seconds
    :param max_iterations: Maximum number of iterations
    :param goal_probability: Probability of biasing towards the other tree or goal
    :param informed: Whether to use informed RRT*
    :return: Path from start to goal or None if no path is found
    """
    if collision_fn(start) or collision_fn(goal):
        return None

    # Initialize two trees: one from the start and one from the goal
    start_tree = [OptimalNode(start)]
    goal_tree = [OptimalNode(goal)]
    
    start_time = time()
    iteration = 0
    best_connection = None  # Will store the best connection between the two trees

    best_solutions = []
    
    while (elapsed_time(start_time) < max_time) and (iteration < max_iterations):
        # Alternate between growing from the start_tree and goal_tree
        if iteration % 2 == 0:
            source_tree = start_tree
            target_tree = goal_tree
        else:
            source_tree = goal_tree
            target_tree = start_tree
        
        # Sampling, biased towards the goal or other tree
        do_goal = random() < goal_probability
        s = sample_fn() if not do_goal else choice(target_tree).config
        
        # Grow the source tree towards the sampled point or goal
        nearest = argmin(lambda n: distance_fn(n.config, s), source_tree)
        path = safe_path(extend_fn(nearest.config, s), collision_fn)
        
        if len(path) == 0:
            iteration += 1
            continue
        
        new_node = OptimalNode(path[-1], parent=nearest, d=distance_fn(nearest.config, path[-1]), path=path[:-1], iteration=iteration)
        source_tree.append(new_node)
        
        # Check if new node can connect to the other tree
        nearest_in_target = argmin(lambda n: distance_fn(n.config, new_node.config), target_tree)
        connection_path = safe_path(extend_fn(new_node.config, nearest_in_target.config), collision_fn)
        
        if len(connection_path) > 0:
            total_cost = new_node.cost + distance_fn(new_node.config, nearest_in_target.config) + nearest_in_target.cost
            if best_connection is None or total_cost < best_connection[2]:
                best_connection = (new_node, nearest_in_target, total_cost)
            
            # Track solution
            best_solutions.append(best_connection)
            best_solutions = sorted(best_solutions, key=lambda x: x[2])  # Sort by total cost
            print(f"Solution {len(best_solutions)}: Cost {total_cost}, Nodes in start tree: {len(start_tree)}, Nodes in goal tree: {len(goal_tree)}")
            # Return the best solution after 3 solutions are found
            if len(best_solutions) >= 3:
                print(f"Returning best solution after {len(best_solutions)} solutions found.")
                best_connection = best_solutions[0]
                start_to_connect = best_connection[0].retrace()
                goal_to_connect = best_connection[1].retrace()[::-1]  # Reverse the path from goal to connection point
                return start_to_connect + goal_to_connect
        
        # Rewiring within the source tree using neighbors
        neighbors = filter(lambda n: distance_fn(n.config, new_node.config) < radius, source_tree)
        for neighbor in neighbors:
            d = distance_fn(neighbor.config, new_node.config)
            if neighbor.cost + d < new_node.cost:
                neighbor_path = safe_path(extend_fn(neighbor.config, new_node.config), collision_fn)
                if len(neighbor_path) != 0:
                    new_node.rewire(neighbor, d, neighbor_path[:-1], iteration=iteration)

        # Rewiring the target tree from the new node
        for neighbor in neighbors:
            d = distance_fn(new_node.config, neighbor.config)
            if new_node.cost + d < neighbor.cost:
                neighbor_path = safe_path(extend_fn(new_node.config, neighbor.config), collision_fn)
                if len(neighbor_path) != 0:
                    neighbor.rewire(new_node, d, neighbor_path[:-1], iteration=iteration)

        if iteration % PRINT_FREQUENCY == 0:
            print(f"Iteration {iteration}: Best cost: {best_connection[2] if best_connection else float('inf')}")
        
        iteration += 1

    # If a valid connection between the two trees has been found, retrace the full path
    if best_connection is not None:
        start_to_connect = best_connection[0].retrace()
        goal_to_connect = best_connection[1].retrace()[::-1]  # Reverse the path from goal to connection point
        return start_to_connect + goal_to_connect
    
    return None
