# CS311 Programming Assignment 1: Search

For this assignment, you will be solving the 8-Puzzle using Breadth First Search (BFS) and A* Search. Refer to the Canvas assignment for assignment specifications. This README describes how to run the skeleton code.

## Running the skeleton program

The skeleton code benchmarks your search functions on a set of random starting boards. Executing `search.py` will run BFS on 1000 initial boards by default. You can change the algorithm, the number of test boards, or even specify a specific input by changing the optional arguments shown below.

```
$ python3 search.py -h
usage: search.py [-h] [-a ALGO] [-i ITER] [-s STATE]

Run search algorithms in random inputs

optional arguments:
  -h, --help            show this help message and exit
  -a ALGO, --algo ALGO  Algorithm (one of bfs, astar, astar_custom)
  -i ITER, --iter ITER  Number of iterations
  -s STATE, --state STATE
                        Execute a single iteration using this board configuration specified as
                        a string, e.g., 123456780
```

For example, to test the A* algorithm, run the program as `python3 search.py -a astar` or to test a specific input, `python3 search.py -s 123456780`.

If you working with Thonny, recall that you can change the command line arguments by modifying the `%Run` command in the shell, e.g., `%Run search.py -a astar`.

## Unit testing

To assist you during development, a unit test suite is provided in `search_test.py`. These tests are a subset of the tests run by Gradescope. You can run the tests by executing the `search_test.py` file as a program, e.g. `python3 search_test.py`

```
$ python3 search_test.py
.........
----------------------------------------------------------------------
Ran 9 tests in 0.068s

OK
```

