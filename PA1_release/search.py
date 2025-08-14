"""
CS311 Programming Assignment 1: Search

Full Name: AJ Noyes

Brief description of my heuristic: The heuristic I used is Manhattan distance with linear conflicts. When a tile is right next to it's goal tile, it might get a good manhattan distance, but that doesn't
take into account how easily the tile will be able to move into its correct position. Linear conflict accounts for this, if two tiles are in the right row but must swap with each other, lots of moving around
must be done. Each time a linear conflict is found, the heuristic is increased by 1, penalizing the score. 

This is more efficient than manhattan distance because it calculates a more accurate description of how close a state is to the goal state. The more accurate the description, the better we can calculate 
how close a given state is to the goal state, and with A*, that will make sure we search the most optimally close states first in the priority queue, finding the goal quicker and more efficiently. 
"""
import heapq
import argparse, itertools, random, sys
from typing import Callable, List, Optional, Sequence, Tuple
from collections import deque


# You are welcome to add constants, but do not modify the pre-existing constants

# Problem size 
BOARD_SIZE = 3

# The goal is a "blank" (0) in bottom right corner
GOAL = tuple(range(1, BOARD_SIZE**2)) + (0,)


def inversions(board: Sequence[int]) -> int:
    """Return the number of times a larger 'piece' precedes a 'smaller' piece in board"""
    return sum(
        (a > b and a != 0 and b != 0) for (a, b) in itertools.combinations(board, 2)
    )


class Node:
    def __init__(self, state: Sequence[int], parent: "Node" = None, cost=0):
        """Create Node to track particular state and associated parent and cost

        State is tracked as a "row-wise" sequence, i.e., the board (with _ as the blank)
        1 2 3
        4 5 6
        7 8 _
        is represented as (1, 2, 3, 4, 5, 6, 7, 8, 0) with the blank represented with a 0

        Args:
            state (Sequence[int]): State for this node, typically a list, e.g. [0, 1, 2, 3, 4, 5, 6, 7, 8]
            parent (Node, optional): Parent node, None indicates the root node. Defaults to None.
            cost (int, optional): Cost in moves to reach this node. Defaults to 0.
        """
        self.state = tuple(state)  # To facilitate "hashable" make state immutable
        self.parent = parent
        self.cost = cost

    def is_goal(self) -> bool:
        """Return True if Node has goal state"""
        return self.state == GOAL

    def expand(self) -> List["Node"]:
        """Expand current node into possible child nodes with corresponding parent and cost"""

        # TODO: Implement this function to generate child nodes based on the current state
        children = []
        blank = self.state.index(0)

        blank_row = None
        blank_col = None

        if blank == 0 or blank == 3 or blank == 6:
            blank_col = 0
        elif blank == blank == 1 or blank == 4 or blank == 7:
            blank_col = 1
        else:
            blank_col = 2
        
        if blank == 0 or blank == 1 or blank == 2:
            blank_row = 0
        elif blank == 3 or blank == 4 or blank == 5:
            blank_row = 1
        else:
            blank_row = 2
        
        # If it can go up
        if blank_row > 0:
            children.append(Node(state = self._swap(row1 = blank_row, col1 = blank_col, row2 = blank_row - 1, col2 = blank_col), parent = self, cost = self.cost + 1))

        # If it can go down
        if blank_row < BOARD_SIZE - 1:
            children.append(Node(state = self._swap(row1 = blank_row, col1 = blank_col, row2 = blank_row + 1, col2 = blank_col), parent = self, cost = self.cost + 1))

        # If it can go right
        if blank_col < BOARD_SIZE - 1:
            children.append(Node(state = self._swap(row1 = blank_row, col1 = blank_col, row2 = blank_row, col2 = blank_col + 1), parent = self, cost = self.cost + 1))

        # If it can go left
        if blank_col > 0:
            children.append(Node(state = self._swap(row1 = blank_row, col1 = blank_col, row2 = blank_row, col2 = blank_col - 1), parent = self, cost = self.cost + 1))

        return children

    def _swap(self, row1: int, col1: int, row2: int, col2: int) -> Sequence[int]:
        """Swap values in current state between row1,col1 and row2,col2, returning new "state" to construct a Node"""
        state = list(self.state)
        state[row1 * BOARD_SIZE + col1], state[row2 * BOARD_SIZE + col2] = (
            state[row2 * BOARD_SIZE + col2],
            state[row1 * BOARD_SIZE + col1],
        )
        return state

    def __str__(self):
        return str(self.state)

    # The following methods enable Node to be used in types that use hashing (sets, dictionaries) or perform comparisons. Note
    # that the comparisons are performed exclusively on the state and ignore parent and cost values.

    def __hash__(self):
        return self.state.__hash__()

    def __eq__(self, other):
        return isinstance(other, Node) and self.state == other.state

    def __lt__(self, other):
        return self.state < other.state


def bfs(initial_board: Sequence[int], max_depth=12) -> Tuple[Optional[Node], int]:
    """Perform breadth-first search to find 8-squares solution

    Args:
        initial_board (Sequence[int]): Starting board
        max_depth (int, optional): Maximum moves to search. Defaults to 12.

    Returns:
        Tuple[Optional[Node], int]: Tuple of solution Node (or None if no solution found) and number of unique nodes explored
    """
    # TODO: Implement BFS. Your function should return a tuple containing the solution node and number of unique node explored
    queue = deque([Node(initial_board)])
    visited = {tuple(initial_board)}  # Set to track unique states
    count = 1  # Start with the initial node
    
    while queue:
        current_node = queue.popleft()
        
        if current_node.is_goal():
            return current_node, count
        
        if current_node.cost >= max_depth:
            continue
        
        for child in current_node.expand():
            if child.state not in visited:
                visited.add(child.state)
                queue.append(child)
                count += 1
    
    return None, count




def manhattan_distance(node: Node) -> int:
    """Compute manhattan distance f(node), i.e., g(node) + h(node)"""
    # TODO: Implement the Manhattan distance heuristic (sum of Manhattan distances to goal location)
    manhattan_num = 0
    for i in range(len(node.state)):
        if node.state[i] != 0:
            # Find current pos
            curr_row = i // BOARD_SIZE
            curr_col = i % BOARD_SIZE

            # Find final position
            goal_index = node.state[i] - 1
            goal_row = goal_index // BOARD_SIZE
            goal_col = goal_index % BOARD_SIZE

            # Add distances
            manhattan_num += abs(curr_row - goal_row) + abs(curr_col - goal_col)

    return node.cost + manhattan_num

def custom_heuristic(node: Node) -> int:
    # TODO: Implement and document your _admissable_ heuristic function: Manhattan distance with linear conflicts

    heuristic_val = manhattan_distance(node) - node.cost  # Get just the manhattan distance
    
    # Checking for linear conflicts in each row
    for row in range(BOARD_SIZE):
        row_spots = []
        for col in range(BOARD_SIZE):
            spot = node.state[row * BOARD_SIZE + col]
            if spot != 0:
                goal_row = (spot - 1) // BOARD_SIZE
                # If the tile is in the right row, add it to be checked. We don't have to worry about order given the order we search
                if goal_row == row:
                    row_spots.append(spot)
        
        # Check each pair in this row for conflicts
        for i, spot1 in enumerate(row_spots):
            for spot2 in row_spots[i+1:]:
                # If a conflict is found, penalize the score
                if spot1 > spot2: 
                    heuristic_val += 1
    
    # Check for linear conflicts in each column
    for col in range(BOARD_SIZE):
        col_spots = []
        for row in range(BOARD_SIZE):
            spot = node.state[row * BOARD_SIZE + col]
            if spot != 0:
                goal_col = (spot - 1) % BOARD_SIZE
                # If the tile is in the right column, add it to be checked. We don't have to worry about order given the order we search
                if goal_col == col: 
                    col_spots.append(spot)
        
        # Check each pair in this column for conflicts
        for i, spot1 in enumerate(col_spots):
            for spot2 in col_spots[i+1:]:
                # If a conflict is found, penalize the score
                if spot1 > spot2:  
                    heuristic_val += 1  
    
    return node.cost + heuristic_val


def astar(initial_board: Sequence[int], max_depth=12, heuristic: Callable[[Node], int] = manhattan_distance) -> Tuple[Optional[Node], int]:
    """Perform astar search to find 8-squares solution

    Args:
        initial_board (Sequence[int]): Starting board
        max_depth (int, optional): Maximum moves to search. Defaults to 12.
        heuristic (_Callable[[Node], int], optional): Heuristic function. Defaults to manhattan_distance.

    Returns:
        Tuple[Optional[Node], int]: Tuple of solution Node (or None if no solution found) and number of unique nodes explored
    """
    # TODO: Implement A* search. Make sure that your code uses the heuristic function provided as
    # an argument so that the test code can switch in your custom heuristic (i.e., do not "hard code"
    # manhattan distance as the heuristic)
    initial_state = Node(initial_board)

    if initial_state.is_goal():
        return initial_state, 1
    
    # Tracking states already visited and number of explored nodes
    visited = {initial_state.state: initial_state.cost}
    nodes_explored = 1

    # Initializing min heap ordered by the heuristic value with id to break ties
    frontier = [(heuristic(initial_state), id(initial_state), initial_state)]
    heapq.heapify(frontier)

    while frontier:
        # Popping from the heap, only care about the actual state value
        _, _, current = heapq.heappop(frontier)

        # If we find the goal node, return it
        if current.is_goal():
            return current, nodes_explored
        
        # If the max depth is reached, stop
        if current.cost >= max_depth: 
            continue
        
        # Explore children
        for child in current.expand():
            # Check if we've seen this state and it doesn't improve cost, if so stop
            if child.state in visited and visited[child.state] <= child.cost:
                continue 

            # Add the child to reached, add to the frontier, and count the visited node
            visited[child.state] = child.cost
            heapq.heappush(frontier, (heuristic(child), id(child), child))
            nodes_explored += 1

    return None, nodes_explored

if __name__ == "__main__":

    # You should not need to modify any of this code
    parser = argparse.ArgumentParser(
        description="Run search algorithms in random inputs"
    )
    parser.add_argument(
        "-a",
        "--algo",
        default="bfs",
        help="Algorithm (one of bfs, astar, astar_custom)",
    )
    parser.add_argument(
        "-i",
        "--iter",
        type=int,
        default=1000,
        help="Number of iterations",
    )
    parser.add_argument(
        "-s",
        "--state",
        type=str,
        default=None,
        help="Execute a single iteration using this board configuration specified as a string, e.g., 123456780",
    )

    args = parser.parse_args()

    num_solutions = 0
    num_cost = 0
    num_nodes = 0

    if args.algo == "bfs":
        algo = bfs
    elif args.algo == "astar":
        algo = astar
    elif args.algo == "astar_custom":
        algo = lambda board: astar(board, heuristic=custom_heuristic)
    else:
        raise ValueError("Unknown algorithm type")

    if args.state is None:
        iterations = args.iter
        while iterations > 0:
            init_state = list(range(BOARD_SIZE**2))
            random.shuffle(init_state)

            # A problem is only solvable if the parity of the initial state matches that
            # of the goal.
            if inversions(init_state) % 2 != inversions(GOAL) % 2:
                continue

            solution, nodes = algo(init_state)
            if solution:
                num_solutions += 1
                num_cost += solution.cost
                num_nodes += nodes

            iterations -= 1
    else:
        # Attempt single input state
        solution, nodes = algo([int(s) for s in args.state])
        if solution:
            num_solutions = 1
            num_cost = solution.cost
            num_nodes = nodes

    if num_solutions:
        print(
            "Iterations:",
            args.iter,
            "Solutions:",
            num_solutions,
            "Average moves:",
            num_cost / num_solutions,
            "Average nodes:",
            num_nodes / num_solutions,
        )
    else:
        print("Iterations:", args.iter, "Solutions: 0")
