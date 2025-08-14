import sys, unittest
import numpy as np
import pandas as pd
import decision as dec

class InformationGainTest(unittest.TestCase):
    def test_no_gain(self):
        X = pd.DataFrame({"A": [0, 1] * 50}, dtype="category")
        y = pd.Series([0] * 50 + [1] * 50)
        self.assertAlmostEqual(dec.information_gain(X, y, "A"), 0.0)

    def test_complete_gain(self):
        X = pd.DataFrame({"A": [0, 1] * 50}, dtype="category")
        y = pd.Series([1, 0] * 50)
        self.assertAlmostEqual(dec.information_gain(X, y, "A"), 1.0)

    def test_partial_gain(self):
        X = pd.DataFrame({"A": [1,2,1,2,2,1,0,1,2,2,0,2]}, dtype="category")
        y = pd.Series([1,0,1,1,0,1,0,1,0,0,0,1])
        self.assertAlmostEqual(dec.information_gain(X, y, "A"), 0.541, places=3)

class PredictExamplesTest(unittest.TestCase):
    def test_simple_tree(self):
        tree = dec.DecisionBranch("A", {0: dec.DecisionLeaf(0), 1: dec.DecisionLeaf(1)})
        X = pd.DataFrame({"A": [0, 1] * 50}, dtype="category")
        y_pred = dec.predict(tree, X)
        np.testing.assert_array_equal(y_pred, [0, 1] * 50)

    def test_xor_tree(self):
        tree = dec.DecisionBranch("A", {
            0: dec.DecisionBranch("B", {0: dec.DecisionLeaf(0), 1: dec.DecisionLeaf(1)}),
            1: dec.DecisionBranch("B", {0: dec.DecisionLeaf(1), 1: dec.DecisionLeaf(0)}),
        })
        X = pd.DataFrame({"A": [0, 1] * 50, "B": [0]* 50 + [1] * 50}, dtype="category")
        y_pred = dec.predict(tree, X)
        np.testing.assert_array_equal(y_pred, np.logical_xor(X["A"] == 1, X["B"] == 1))


class LearnTreeTest(unittest.TestCase):
    def test_xor_data(self):
        # Bias the data so that B has more information gain
        X = pd.DataFrame({"A": [0, 1] * 50 + [1, 1], "B": [0]* 50 + [1] * 50 + [1, 0]}, dtype="category")
        y = pd.Series(np.logical_xor(X["A"] == 1, X["B"] == 1).astype(int), name="label")
        
        self.assertGreater(dec.information_gain(X, y, "B"), dec.information_gain(X, y, "A"))
        tree = dec.fit(X, y)
        self.assertEqual(tree.attr, "B")
        self.assertEqual(tree.branches[0].branches[1].label, 1)
        self.assertEqual(tree.branches[1].branches[0].label, 1)
        self.assertEqual(tree.branches[0].branches[0].label, 0)
        self.assertEqual(tree.branches[1].branches[1].label, 0)

if __name__ == "__main__":
    unittest.main(argv=sys.argv[:1])
