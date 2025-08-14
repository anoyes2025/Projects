# CS311 Programming Assignment 5: Decision Trees

For this assignment, you will be training and testing Decisions Trees. Refer to the Canvas assignment for assignment specifications. This README describes how to run the skeleton code.

## Running the skeleton program

The skeleton code trains and tests decision trees on a variety of datasets. Executing `decision.py` will train adn test your program on a small dataset by default.  You can change the dataset and the evaluation approach for the adult dataset by changing the optional arguments shown below.

```
$ python3 decision.py -h
usage: decision.py [-h] [-p PREFIX] [-k K_SPLITS]

Train and test decision tree learner

optional arguments:
  -h, --help            show this help message and exit
  -p PREFIX, --prefix PREFIX
                        Prefix for dataset files. Expects <prefix>.[train|test]_[data|label].txt files (except for adult). Allowed values: small1, hepatitis, adult.
  -k K_SPLITS, --k_splits K_SPLITS
                        Number of splits for stratified k-fold testing
```

For example, to train and test with the other datasets, run the program as `python3 decision.py -p adult`.

If you are working with Thonny, recall that you can change the command line arguments by modifying the `%Run` command in the shell, e.g., `%Run decision.py -p adult`.

## Unit testing

To assist you during development, a unit test suite is provided in `decision_test.py`. These tests are a subset of the tests run by Gradescope. You can run the tests by executing the `decision_test.py` file as a program, e.g. `python3 decision_test.py`. 

```
$ python3 decision_test.py
......
----------------------------------------------------------------------
Ran 6 tests in 0.142s

OK
```