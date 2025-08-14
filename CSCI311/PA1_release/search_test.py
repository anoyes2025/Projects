import sys, unittest

import search

class NodeTests(unittest.TestCase):
    def test_constants(self):
        self.assertEqual(search.GOAL, tuple(range(1, search.BOARD_SIZE**2)) + (0,))
    
    def test_expand_count(self):
        # Expected expansions for 'blank' at corresponding index
        for idx, exps in enumerate([2, 3, 2, 3, 4, 3, 2, 3, 2]):
            board = list(range(1,9))
            board.insert(idx, 0)
            
            node = search.Node(board)
            self.assertEqual(len(node.expand()), exps, f"'Blank' at index {idx} should have {exps} expansions")

    def test_expanded_down_right(self):
        root_node = search.Node([0,1,2,3,4,5,6,7,8])
        self.assertIsNone(root_node.parent)
        self.assertEqual(root_node.cost, 0)

        [right, down] = sorted(root_node.expand()) # Right should be "<" when comparing state arrays
        self.assertEqual(right, search.Node([1,0,2,3,4,5,6,7,8], root_node))
        self.assertIs(right.parent, root_node)
        self.assertEqual(right.cost, 1)

        self.assertEqual(down, search.Node([3,1,2,0,4,5,6,7,8], root_node))
        self.assertIs(down.parent, root_node)
        self.assertEqual(down.cost, 1)
    
    def test_expanded_up_left(self):
        root_node = search.Node([1,2,3,4,5,6,7,8,0], None)
        self.assertIsNone(root_node.parent)
        self.assertEqual(root_node.cost, 0)

        [up, left] = sorted(root_node.expand()) # Right should be "<" when comparing state arrays
        self.assertEqual(up, search.Node([1,2,3,4,5,0,7,8,6], root_node))
        self.assertIs(up.parent, root_node)
        self.assertEqual(up.cost, 1)

        self.assertEqual(left, search.Node([1,2,3,4,5,6,7,0,8], root_node))
        self.assertIs(left.parent, root_node)
        self.assertEqual(left.cost, 1)


class BSFTest(unittest.TestCase):
    def test_known_solvable_inputs(self):
        boards = [
            ([1,2,3,4,5,6,0,7,8], 2),
            ([1,2,3,4,6,0,7,5,8], 3),
            ([4,1,2,0,5,3,7,8,6], 5),
            ([4,1,2,7,6,3,0,5,8], 8),
            ([4,1,2,7,6,3,5,8,0], 10),
            ([1,3,8,7,4,2,0,6,5], 12),
        ]
        for board, depth in boards:
            solution, nodes = search.bfs(board)
            self.assertIsNotNone(solution)
            self.assertEqual(solution.cost, depth)
            self.assertGreater(nodes, 0)

    def test_unsolvable_input(self):
        solution, _ = search.bfs([1,3,4,8,0,5,7,2,6])
        self.assertIsNone(solution)

class AStarTest(unittest.TestCase):
    def test_manhattan_distance(self):
        node = search.Node([1, 2, 3, 7, 8, 5, 4, 6, 0], cost=2)
        self.assertEqual(search.manhattan_distance(node), 8)
    
    def test_known_solvable_inputs(self):
        boards = [
            ([1,2,3,4,5,6,0,7,8], 2),
            ([1,2,3,4,6,0,7,5,8], 3),
            ([4,1,2,0,5,3,7,8,6], 5),
            ([4,1,2,7,6,3,0,5,8], 8),
            ([4,1,2,7,6,3,5,8,0], 10),
            ([1,3,8,7,4,2,0,6,5], 12),
        ]
        for board, depth in boards:
            solution, nodes = search.astar(board, heuristic=search.manhattan_distance)
            self.assertIsNotNone(solution)
            self.assertEqual(solution.cost, depth)
            self.assertGreater(nodes, 0)

    def test_unsolvable_input(self):
        solution, _ = search.astar([1,3,4,8,0,5,7,2,6], heuristic=search.manhattan_distance)
        self.assertIsNone(solution)

class AStarCustomTest(unittest.TestCase):
    def test_known_solvable_inputs(self):
        boards = [
            ([1,2,3,4,5,6,0,7,8], 2),
            ([1,2,3,4,6,0,7,5,8], 3),
            ([4,1,2,0,5,3,7,8,6], 5),
            ([4,1,2,7,6,3,0,5,8], 8),
            ([4,1,2,7,6,3,5,8,0], 10),
            ([1,3,8,7,4,2,0,6,5], 12),
        ]
        for board, depth in boards:
            solution, nodes = search.astar(board, heuristic=search.custom_heuristic)
            self.assertIsNotNone(solution)
            self.assertEqual(solution.cost, depth)
            self.assertGreater(nodes, 0)

    def test_unsolvable_input(self):
        solution, _ = search.astar([1,3,4,8,0,5,7,2,6], heuristic=search.custom_heuristic)
        self.assertIsNone(solution)

if __name__ == '__main__':
    unittest.main(argv=sys.argv[:1])
