"""
CS311 Programming Assignment 5: Decision Trees

Full Name: AJ Noyes

Description of performance on Hepatitis data set:

In terms of performance, accuracy was 86.54% on the hepatitis dataset. Recall was 90.48%, meaning that there were not many false positives (4 occurrences), as in not many people
were incorrectly predicted to survive. Precision was 92.68%, meaning that there were less false negatives (3 occurrences), so the model was less likely to falsely predict someone 
would not survive. This means the model is slightly weighted towards being optimistic. In a real world application, it might be better to have a model that is more cautious, as in
more likely to predict a false negative than a false positive. The tree structure is as I would expect given some brief research into indicators of hepatitis. If someone does not have 
enlarged veins, but they do have a histology, I assume that means that they already died. If they have none, and no histology, they must be healthy. The next feature is a blood result from
serum glutamic oxaloacetic transaminase (SGOT), which indicated liver damage or disease, another strong indicator of fatal hepatitis. Other strong indicators are shown in the structure
as features, like an enlarged liver, gender (males more likely), age, bilirubin (high levels indicate liver problems). The order of features also make sense, as in varices is a primary indicator,
SGOT is a secondary indicator, then bilirubin and so on. The order of features following other features also makes sense, there is no need examine anorexia if the SGOT test already indicated 
liver damage because there are other stronger features to examine in that context, however it can be an indicator of lower chance of survival from hepatitis in the absence of stronger indicators. 

Description of Adult dataset discretization and selected features:

In terms of discretization, I discretized and used all numeric features. I separated age categories by typical age categorizations, and hours per week similarly by a common understanding of what 
part time work versus full time work is. For the capital gains and losses, I estimated a range that would capture most capital gains and losses given an income around 50k. For the 
categorical features, I chose occupation and marital status as strong indicators of a higher income. Occupation will most definitely have a direct effect on income, and someone is more likely to 
get married if they have a higher income to raise a family.

Potential effects of using your model for marketing a cash-back credit card:

Some potential benefits might be that revenue is increased by targeting high income consumers, so that they are more inclined to spend and earn cash back. Their higher spending power makes
their spending more beneficial for the bank. This also means the more important, high paying customers are prioritized, and more likely to stick around given the benefits they receive. However, 
this could be an ethical issue as it could be seen as a form of elitism/discrimination. Lower income families could benefit a lot more from cash back rewards, but the bank might not be able to
squeeze them for as much money, so they are forgotten about. Also there are some data privacy concerns with feeding a model sensitive financial information. Who is responsible for the data and the model?
Can they be held accountable? Potentially the most problematic issue is specific to the data. The threshold for what is considered a high-earner is the 76th percentile of the income distribution, however 
this threshold is imbalanced for the black and female population, at the 88th and 89th percentiles. Because of this, the model may start to associate the female attribute of gender, and the black attribute
for race with the negative label of 0, potentially denying a high-earner a cash-back reward based on their race or gender. 
"""

import argparse, os, random, sys
from typing import Any, Dict, Sequence, Tuple, Union
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold

# Type alias for nodes in decision tree
DecisionNode = Union["DecisionBranch", "DecisionLeaf"]

class DecisionBranch:
    """Branching node in decision tree"""

    def __init__(self, attr: str, branches: Dict[Any, DecisionNode]):
        """Create branching node in decision tree

        Args:
            attr (str): Splitting attribute
            branches (Dict[Any, DecisionNode]): Children nodes for each possible value of `attr`
        """
        self.attr = attr
        self.branches = branches

    def predict(self, x: pd.Series):
        """Return predicted labeled for array-like example x"""
        # Get the value of the attribute for this example
        attr_value = x[self.attr]
        return self.branches[attr_value].predict(x)

    def display(self, indent=0):
        """Pretty print tree starting at optional indent"""
        print("Test Feature", self.attr)
        for val, subtree in self.branches.items():
            print(" " * 4 * indent, self.attr, "=", val, "->", end=" ")
            subtree.display(indent + 1)


class DecisionLeaf:
    """Leaf node in decision tree"""

    def __init__(self, label):
        """Create leaf node in decision tree

        Args:
            label: Label for this node
        """
        self.label = label

    def predict(self, x):
        """Return predicted labeled for array-like example x"""
        return self.label

    def display(self, indent=0):
        """Pretty print tree starting at optional indent"""
        print("Label=", self.label)


def information_gain(X: pd.DataFrame, y: pd.Series, attr: str) -> float:
    """Return the expected reduction in entropy from splitting X, y by attribute"""

    parent_entropy = compute_entropy(y)

    weighted_entropy = 0
    num_examples = len(y)

    # Group by the attribute
    attr_groups = y.groupby(X[attr])

    for _, group_labels in attr_groups:
        group_entropy = compute_entropy(group_labels)
        group_weight = len(group_labels) / num_examples
        weighted_entropy += group_weight * group_entropy

    return parent_entropy - weighted_entropy
    
    
    
def compute_entropy(y: pd.Series) -> float:
    """Compute entropy for binary classification"""
    # Count the number of positive and negative examples
    value_counts = y.value_counts(normalize=True)

    # If all examples are the same, entropy is 0
    if len(value_counts) <= 1:
        return 0.0

    # Compute entropy
    entropy = 0
    for p in value_counts:
        entropy -= p * np.log2(p+1e-10)

    return entropy



def learn_decision_tree(
    X: pd.DataFrame,
    y: pd.Series,
    attrs: Sequence[str],
    y_parent: pd.Series,
) -> DecisionNode:
    """Recursively learn the decision tree

    Args:
        X (pd.DataFrame): Table of examples (as DataFrame)
        y (pd.Series): array-like example labels (target values)
        attrs (Sequence[str]): Possible attributes to split examples
        y_parent (pd.Series): array-like example labels for parents (parent target values)

    Returns:
        DecisionNode: Learned decision tree node
    """
    # TODO: Implement recursive tree construction based on pseudo code in class
    # and the assignment
    # If no examples, return plurality value of parent examples
    if len(y) == 0:
        return DecisionLeaf(y_parent.mode().iloc[0])
    # If all examples have the same classification, return that classification
    elif len(y.unique()) == 1:
        return DecisionLeaf(y.iloc[0])
    # If no more attributes, return plurality value
    elif len(attrs) == 0:
        return DecisionLeaf(y.mode().iloc[0])
    else:
        # Find attribute with maximum information gain
        max_gain = -1
        best_attr = None  # Default to first attribute
        for attr in attrs:
            gain = information_gain(X, y, attr)
            if gain > max_gain:
                max_gain = gain
                best_attr = attr

        branches = {}

        # Ensure branch for each attribute value
        for val in X[best_attr].cat.categories:
            # Subset examples with this attribute value
            subset = X[best_attr] == val
            subset_X = X[subset]
            subset_y = y[subset]
        
            # Remove the current attribute from further consideration
            remaining_attrs = [a for a in attrs if a != best_attr]
            subtree = learn_decision_tree(subset_X, subset_y, remaining_attrs, y)
            branches[val] = subtree

        return DecisionBranch(best_attr, branches)


def fit(X: pd.DataFrame, y: pd.Series) -> DecisionNode:
    """Return train decision tree on examples, X, with labels, y"""
    # You can change the implementation of this function, but do not modify the signature
    return learn_decision_tree(X, y, X.columns, y)


def predict(tree: DecisionNode, X: pd.DataFrame):
    """Return array-like predctions for examples, X and Decision Tree, tree"""

    # You can change the implementation of this function, but do not modify the signature

    # Invoke prediction method on every row in dataframe. `lambda` creates an anonymous function
    # with the specified arguments (in this case a row). The axis argument specifies that the function
    # should be applied to all rows.
    return X.apply(lambda row: tree.predict(row), axis=1)


def load_adult(feature_file: str, label_file: str):

    # Load the feature file
    examples = pd.read_table(
        feature_file,
        dtype={
            "age": int,
            "workclass": "category",
            "education": "category",
            "marital-status": "category",
            "occupation": "category",
            "relationship": "category",
            "race": "category",
            "sex": "category",
            "capital-gain": int,
            "capital-loss": int,
            "hours-per-week": int,
            "native-country": "category",
        },
    )
    labels = pd.read_table(label_file).squeeze().rename("label")


    # TODO: Select columns and choose a discretization for any continuous columns. Our decision tree algorithm
    # only supports discretized features and so any continuous columns (those not already "category") will need
    # to be discretized.

    # For example the following discretizes "hours-per-week" into "part-time" [0,40) hours and
    # "full-time" 40+ hours. Then returns a data table with just "education" and "hours-per-week" features.
   
    examples["age"] = pd.cut(
       examples["age"],
       bins=[0, 10, 20, 35, 70, sys.maxsize],
       right=False,
       labels=["child", "teenager","young-adult", "adult", "senior"],
    )

    examples["capital-gain"] = pd.cut(
       examples["capital-gain"],
       bins=[0, 1000, 2500, sys.maxsize],
       right=False,
       labels=[ "low", "medium", "high"],
    )

    examples["capital-loss"] = pd.cut(
       examples["capital-loss"],
       bins=[0, 1000, 2500, sys.maxsize],
       right=False,
       labels=["low", "medium","high"],
    )

    examples["hours-per-week"] = pd.cut(
       examples["hours-per-week"],
       bins=[0, 30, 40, sys.maxsize],
       right=False,
       labels=["PT", "FT", "OT"],
    )

    return examples[["age", "workclass", "marital-status", "hours-per-week", "occupation", "capital-gain", "capital-loss"]], labels


# You should not need to modify anything below here


def load_examples(
    feature_file: str, label_file: str, **kwargs
) -> Tuple[pd.DataFrame, pd.Series]:
    """Load example features and labels. Additional arguments are passed to
    the pandas.read_table function.

    Args:
        feature_file (str): Delimited file of categorical features
        label_file (str): Single column binary labels. Column name will be renamed to "label".

    Returns:
        Tuple[pd.DataFrame,pd.Series]: Tuple of features and labels
    """
    return (
        pd.read_table(feature_file, dtype="category", **kwargs),
        pd.read_table(label_file, **kwargs).squeeze().rename("label"),
    )


def compute_metrics(y_true, y_pred):
    """Compute metrics to evaluate binary classification accuracy

    Args:
        y_true: Array-like ground truth (correct) target values.
        y_pred: Array-like estimated targets as returned by a classifier.

    Returns:
        dict: Dictionary of metrics in including confusion matrix, accuracy, recall, precision and F1
    """
    return {
        "confusion": metrics.confusion_matrix(y_true, y_pred),
        "accuracy": metrics.accuracy_score(y_true, y_pred),
        "recall": metrics.recall_score(y_true, y_pred),
        "precision": metrics.precision_score(y_true, y_pred),
        "f1": metrics.f1_score(y_true, y_pred),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and test decision tree learner")
    parser.add_argument(
        "-p",
        "--prefix",
        default="small1",
        help="Prefix for dataset files. Expects <prefix>.[train|test]_[data|label].txt files (except for adult). Allowed values: small1, tennis, hepatitis, adult.",
    )
    parser.add_argument(
        "-k",
        "--k_splits",
        default=10,
        type=int,
        help="Number of splits for stratified k-fold testing",
    )

    args = parser.parse_args()

    if args.prefix != "adult":
        # Derive input files names for test sets
        train_data_file = os.path.join(
            os.path.dirname(__file__), "data", f"{args.prefix}.train_data.txt"
        )
        train_labels_file = os.path.join(
            os.path.dirname(__file__), "data", f"{args.prefix}.train_label.txt"
        )
        test_data_file = os.path.join(
            os.path.dirname(__file__), "data", f"{args.prefix}.test_data.txt"
        )
        test_labels_file = os.path.join(
            os.path.dirname(__file__), "data", f"{args.prefix}.test_label.txt"
        )

        # Load training data and learn decision tree
        train_data, train_labels = load_examples(train_data_file, train_labels_file)
        tree = fit(train_data, train_labels)
        tree.display()

        # Load test data and predict labels with previously learned tree
        test_data, test_labels = load_examples(test_data_file, test_labels_file)
        pred_labels = predict(tree, test_data)

        # Compute and print accuracy metrics
        predict_metrics = compute_metrics(test_labels, pred_labels)
        for met, val in predict_metrics.items():
            print(
                met.capitalize(),
                ": ",
                ("\n" if isinstance(val, np.ndarray) else ""),
                val,
                sep="",
            )
    else:
        # We use a slightly different procedure with "adult". Instead of using a fixed split, we split
        # the data k-ways (preserving the ratio of output classes) and test each split with a Decision
        # Tree trained on the other k-1 splits.
        data_file = os.path.join(os.path.dirname(__file__), "data", "adult.data.txt")
        labels_file = os.path.join(os.path.dirname(__file__), "data", "adult.label.txt")
        data, labels = load_adult(data_file, labels_file)

        scores = []

        kfold = StratifiedKFold(n_splits=args.k_splits)
        for train_index, test_index in kfold.split(data, labels):
            X_train, X_test = data.iloc[train_index], data.iloc[test_index]
            y_train, y_test = labels.iloc[train_index], labels.iloc[test_index]

            tree = fit(X_train, y_train)
            y_pred = predict(tree, X_test)
            scores.append(metrics.accuracy_score(y_test, y_pred))

            tree.display()

        print(
            f"Mean (std) Accuracy (for k={kfold.n_splits} splits): {np.mean(scores)} ({np.std(scores)})"
        )
