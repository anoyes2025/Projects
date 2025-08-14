"""
CS311 Programming Assignment 3: CSP

Full Name: AJ Noyes

Brief description of my solver:

My backtracking search uses the most constrained variable heuristic and forward checking instead of ac3. The heuristic reduces the possible number of assignments and thus 
less recursive calls occur. The forward checking detects inconsistencies earlier and decreases the search space. Also, I use the AC3 as a preprocessor, so if constraints are not
satisfied, quit early. These combine to make my backtracking search more efficient. My implementation could be extended and improved by adding the least constraining value heuristic. 
"""

import argparse, time
from functools import wraps
from typing import Dict, List, Optional, Set, Tuple
import copy

# You are welcome to add constants, but do not modify the pre-existing constants

# Length of side of a Soduku board
SIDE = 9

# Length of side of "box" within a Soduku board
BOX = 3

# Domain for cells in Soduku board
DOMAIN = range(1, 10)

# Helper constant for checking a Soduku solution
SOLUTION = set(DOMAIN)


def check_solution(board: List[int], original_board: List[int]) -> bool:
    """Return True if board is a valid Sudoku solution to original_board puzzle"""
    # Original board values are maintained
    for s, o in zip(board, original_board):
        if o != 0 and s != o:
            return False
    for i in range(SIDE):
        # Valid row
        if set(board[i * SIDE : (i + 1) * SIDE]) != SOLUTION:
            return False
        # Valid column
        if set(board[i : SIDE * SIDE : SIDE]) != SOLUTION:
            return False
        # Valid Box (here i is serving as the "box" id since there are SIDE boxes)
        box_row, box_col = (i // BOX) * BOX, (i % BOX) * BOX
        box = set()
        for r in range(box_row, box_row + BOX):
            box.update(board[r * SIDE + box_col : r * SIDE + box_col + BOX])
        if box != SOLUTION:
            return False
    return True


def countcalls(func):
    """Decorator to track the number of times a function is called. Provides `calls` attribute."""
    countcalls.calls = 0

    @wraps(func)
    def wrapper(*args, **kwargs):
        initial_calls = countcalls.calls
        countcalls.calls += 1
        result = func(*args, **kwargs)
        wrapper.calls = countcalls.calls - initial_calls
        return result

    return wrapper


def revise(domains: List[List[int]], val_index: int, neighbor_index: int) -> bool:
    """Revise domain of x using constraint with y."""
    revised = False
    domain1 = domains[val_index]
    
    for val in domain1[:]:
        if not any(neighbor != val for neighbor in domains[neighbor_index]):
            domains[val_index].remove(val)
            revised = True
    
    return revised

def ac3(domains: List[List[int]], neighbors: List[List[int]], queue: Set[Tuple[int, int]]) -> bool:
    """AC3 Implementation"""
    queue_copy = copy.deepcopy(queue)

    while queue_copy:
        val1_index, val2_index = queue_copy.pop()
        
        if revise(domains, val1_index, val2_index):
            if not domains[val1_index]:
                return False
            
            for neighbor_index in neighbors[val1_index]:
                if neighbor_index != val2_index and (neighbor_index, val1_index) not in queue_copy:
                    queue_copy.add((neighbor_index, val1_index))
    
    return True

# The @countcalls decorator tracks the number of times we call the recursive function. Make sure the decorator
# is included on your recursive search function if you change the implementation.
@countcalls
def backtracking_search(
    neighbors: List[List[int]],
    queue: Set[Tuple[int, int]],
    domains: List[List[int]],
    assignment: Dict[int, int],
) -> Optional[Dict[int, int]]:
    """Perform backtracking search on CSP using AC3

    Args:
        neighbors (List[List[int]]): Indices of neighbors for each variable
        queue (Set[Tuple[int, int]]): Variable constraints; (x, y) indicates x must be consistent with y
        domains (List[List[int]]): Domains for each variable
        assignment (Dict[int, int]): Current variable->value assignment 

    Returns:
        Optional[Dict[int, int]]: Solution or None indicating no solution found
    """
    # If complete just return
    if len(assignment) == SIDE * SIDE:
        return assignment

    # Select variables without assignment
    unassigned_variables = [v for v in range(SIDE * SIDE) if v not in assignment]
    var = unassigned_variables[0]

    # Try values in the domain
    for value in domains[var]:
        if all(value != assignment.get(neighbor) for neighbor in neighbors[var]):
            # Try assignment
            assignment[var] = value
            domains_copy = copy.deepcopy(domains)
            domains_copy[var] = [value]

            # If consistent
            if ac3(domains_copy, neighbors, queue):
                result = backtracking_search(
                    neighbors, queue, domains_copy, assignment
                )

                if result:
                    return result

            # Backtrack
            del assignment[var]

    return None


def generate_neighbors() -> List[List[int]]:
    neighbors = [[] for _ in range(SIDE * SIDE)]
    
    for var in range(SIDE * SIDE):
        row, col = var // SIDE, var % SIDE
        
        # Row neighbors
        for c in range(SIDE):
            if c != col:
                neighbors[var].append(row * SIDE + c)
        
        # Column neighbors
        for r in range(SIDE):
            if r != row:
                neighbors[var].append(r * SIDE + col)
        
        # Box neighbors
        box_row, box_col = (row // BOX) * BOX, (col // BOX) * BOX
        for r in range(box_row, box_row + BOX):
            for c in range(box_col, box_col + BOX):
                neighbor = r * SIDE + c
                if neighbor != var and neighbor not in neighbors[var]:
                    neighbors[var].append(neighbor)
    
    return neighbors

def generate_queue(neighbors: List[List[int]]) -> Set[Tuple[int, int]]:
    queue = set()
    for var, var_neighbors in enumerate(neighbors):
        for neighbor in var_neighbors:
            queue.add((var, neighbor))
    return queue

def sudoku(board: List[int]) -> Tuple[Optional[List[int]], int]:
    """Solve Sudoku puzzle using backtracking search with the AC3 algorithm

    Do not change the signature of this function

    Args:
        board (List[int]): Flattened list of board in row-wise order. Cells that are not initially filled should be 0.

    Returns:
        Tuple[Optional[List[int]], int]: Solution as flattened list in row-wise order, or None, if no solution found and
            a count of calls to recursive backtracking function
    """

    # TODO: Complete the initialization of the neighbors and queue data structures

    domains = [[val] if val else list(DOMAIN) for val in board]
    neighbors = generate_neighbors()
    queue = generate_queue(neighbors)

    # Initialize the assignment for any squares with domains of size 1 (e.g., pre-specified squares).
    # While not necessary for correctness, initializing the assignment improves performance, especially
    # for plain backtracking search.

    assignment = {
        var: domain[0] for var, domain in enumerate(domains) if len(domain) == 1
    }
    result = backtracking_search(neighbors, queue, domains, assignment)

    # Convert result dictionary to list
    if result is not None:
        result = [result[i] for i in range(SIDE * SIDE)]
    return result, backtracking_search.calls

@countcalls
def my_backtracking_search(
    neighbors: List[List[int]],
    queue: Set[Tuple[int, int]],
    domains: List[List[int]],
    assignment: Dict[int, int],
) -> Optional[Dict[int, int]]:
    """Custom backtracking search implementing efficient heuristics

    Args:
        neighbors (List[List[int]]): Indices of neighbors for each variable
        queue (Set[Tuple[int, int]]): Variable constraints; (x, y) indicates x must be consistent with y
        domains (List[List[int]]): Domains for each variable
        assignment (Dict[int, int]): Current variable->value assignment 

    Returns:
        Optional[Dict[int, int]]: Solution or None indicating no solution found
    """

    # If complete, return solution
    if len(assignment) == SIDE * SIDE:
        return assignment
    
    # Most constrained variable heuristic 
    unassigned_vars = [v for v in range(SIDE * SIDE) if v not in assignment]
    # Select variable with smallest domain first
    min_var = min(
        unassigned_vars, 
        key=lambda v: len(domains[v])
    )

    domain_save = domains[min_var][:]
    # Try each value
    for value in domains[min_var]:
        # Check if value is consistent with current assignment
        if all(value != assignment.get(neighbor) for neighbor in neighbors[min_var]):
            assignment[min_var] = value
            domains[min_var] = [value]

            # Forward checking 
            is_consistent = True
            removed_values = {}
            for neighbor in neighbors[min_var]:
                # If this value is in neighbor's domain, remove it
                if value in domains[neighbor]:
                    domains[neighbor].remove(value)
                    if neighbor not in removed_values:
                        removed_values[neighbor] = []
                    removed_values[neighbor].append(value)

                    # If neighbor's domain becomes empty
                    if not domains[neighbor]:
                        is_consistent = False
                        break
            
            # If consistent
            if is_consistent:
                # Recursive search
                result = my_backtracking_search(
                    neighbors, 
                    queue, 
                    domains, 
                    assignment
                )
                
                # If solution found
                if result:
                    return result
                
            del assignment[min_var]
            for neighbor, removed in removed_values.items():
                domains[neighbor].extend(removed)
    
    # No solution found
    domains[min_var] = domain_save
    return None


def my_sudoku(board: List[int]) -> Tuple[Optional[List[int]], int]:
    """Solve Sudoku puzzle using your own custom solver

    Do not change the signature of this function

    Args:
        board (List[int]): Flattened list of board in row-wise order. Cells that are not initially filled should be 0.

    Returns:
        Tuple[Optional[List[int]], int]: Solution as flattened list in row-wise order, or None, if no solution found and
            a count of calls to recursive backtracking function
    """

    # TODO: Complete the initialization of the neighbors and queue data structures

    domains = [[val] if val else list(DOMAIN) for val in board]
    neighbors = generate_neighbors()
    queue = generate_queue(neighbors)   

    # Initialize the assignment for any squares with domains of size 1 (e.g., pre-specified squares).
    assignment = {
        var: domain[0] for var, domain in enumerate(domains) if len(domain) == 1
    }

    if not ac3(domains, neighbors, queue):
        return None, 0
    
    result = my_backtracking_search(neighbors, queue, domains, assignment)

    # Convert assignment dictionary to list
    if result is not None:
        result = [result[i] for i in range(SIDE * SIDE)]
    return result, my_backtracking_search.calls


if __name__ == "__main__":
    # You should not need to modify any of this code
    parser = argparse.ArgumentParser(description="Run sudoku solver")
    parser.add_argument(
        "-a",
        "--algo",
        default="ac3",
        help="Algorithm (one of ac3, custom)",
    )
    parser.add_argument(
        "-l",
        "--level",
        default="easy",
        help="Difficulty level (one of easy, medium, hard)",
    )
    parser.add_argument(
        "-t",
        "--trials",
        default=1,
        type=int,
        help="Number of trials for timing",
    )
    parser.add_argument("puzzle", nargs="?", type=str, default=None)

    args = parser.parse_args()

    # fmt: off
    if args.puzzle:
        board = [int(c) for c in args.puzzle]
        if len(board) != SIDE*SIDE or set(board) > (set(DOMAIN) | { 0 }):
            raise ValueError("Invalid puzzle specification, it must be board length string with digits 0-9")
    elif args.level == "easy":
        board = [
            0,0,0,1,3,0,0,0,0,
            7,0,0,0,4,2,0,8,3,
            8,0,0,0,0,0,0,4,0,
            0,6,0,0,8,4,0,3,9,
            0,0,0,0,0,0,0,0,0,
            9,8,0,3,6,0,0,5,0,
            0,1,0,0,0,0,0,0,4,
            3,4,0,5,2,0,0,0,8,
            0,0,0,0,7,3,0,0,0,
        ]
    elif args.level == "medium":
        board = [
            0,4,0,0,9,8,0,0,5,
            0,0,0,4,0,0,6,0,8,
            0,5,0,0,0,0,0,0,0,
            7,0,1,0,0,9,0,2,0,
            0,0,0,0,8,0,0,0,0,
            0,9,0,6,0,0,3,0,1,
            0,0,0,0,0,0,0,7,0,
            6,0,2,0,0,7,0,0,0,
            3,0,0,8,4,0,0,6,0,
        ]
    elif args.level == "hard":
        board = [
            1,2,0,4,0,0,3,0,0,
            3,0,0,0,1,0,0,5,0,  
            0,0,6,0,0,0,1,0,0,  
            7,0,0,0,9,0,0,0,0,    
            0,4,0,6,0,3,0,0,0,    
            0,0,3,0,0,2,0,0,0,    
            5,0,0,0,8,0,7,0,0,    
            0,0,7,0,0,0,0,0,5,    
            0,0,0,0,0,0,0,9,8,
        ]
    else:
        raise ValueError("Unknown level")
    # fmt: on

    if args.algo == "ac3":
        solver = sudoku
    elif args.algo == "custom":
        solver = my_sudoku
    else:
        raise ValueError("Unknown algorithm type")

    times = []
    for i in range(args.trials):
        test_board = board[:]  # Ensure original board is not modified
        start = time.perf_counter()
        solution, recursions = solver(test_board)
        end = time.perf_counter()
        times.append(end - start)
        if solution and not check_solution(solution, board):
            print(solution)
            raise ValueError("Invalid solution")

        if solution:
            print(f"Trial {i} solved with {recursions} recursions")
            print(solution)
        else:
            print(f"Trial {i} not solved with {recursions} recursions")

    print(
        f"Minimum time {min(times)}s, Average time {sum(times) / args.trials}s (over {args.trials} trials)"
    )
