"""
File: evo.py
Description: A concise evolutionary computing framework for solving
            multi-objective optimization problems!
"""

import random as rnd
import copy   # doing deep copies of solutions when generating offspring
from functools import reduce  # for discarding dominated (bad) solutions
import numpy as np
import pandas as pd
from assignta import (
    overallocation,
    conflicts,
    undersupport,
    unavailable,
    unpreferred
)

class Evo:

    def __init__(self):
        """framework constructor"""
        self.pop = {}  # population of solutions: evaluation --> solution
        self.fitness = {}  # objectives:    name --> objective function (f)
        self.agents = {}  # agents:   name --> (operator/function,  num_solutions_input)

    def add_objective(self, name, f):
        """ Register a new objective for evaluating solutions """
        self.fitness[name] = f

    def add_agent(self, name, op, k=1):
        """ Register an agent take works on k input solutions """
        self.agents[name] = (op, k)

    def get_random_solutions(self, k=1):
        """ Picks k random solutions from the population
        and returns them as a list of deep-copies """
        if len(self.pop) == 0:  # No solutions - this shouldn't happen!
            return []
        else:
            solutions = tuple(self.pop.values())
            return [copy.deepcopy(rnd.choice(solutions)) for _ in range(k)]


    def add_solution(self, sol):
        """ Adds the solution to the current population.
        Added solutions are evaluated wrt each registered objective. """

        # Create the evaluation key
        # key:  ( (objname1, objvalue1), (objname2, objvalue2), ...... )
        eval = tuple([(name, f(sol)) for name, f in self.fitness.items()])

        # Add to the dictionary
        self.pop[eval] = sol


    def run_agent(self, name):
        """ Invoking a named agent against the current population """
        op, k = self.agents[name]
        picks = self.get_random_solutions(k)
        new_solution = op(picks)
        self.add_solution(new_solution)


    @staticmethod
    def _dominates(p, q):
        """ p = evaluation of solution: ((obj1, score1), (obj2, score2), ... )"""
        pscores = [score for _,score in p]
        qscores = [score for _,score in q]
        score_diffs = list(map(lambda x,y: y-x, pscores, qscores))
        min_diff = min(score_diffs)
        max_diff = max(score_diffs)
        return min_diff >= 0.0 and max_diff > 0.0

    @staticmethod
    def _reduce_nds(S, p):
        return S - {q for q in S if Evo._dominates(p,q)}

    def remove_dominated(self):
        """ Remove solutions from the pop that are dominated (worse) compared
        to other existing solutions. This is what provides selective pressure
        driving the population towards the pareto optimal tradeoff curve. """
        nds = reduce(Evo._reduce_nds, self.pop.keys(), self.pop.keys())
        self.pop = {k:self.pop[k] for k in nds}

    # Attendance: 8337

    def evolve(self, n=1, dom=100, status=1000):
        """ Run the framework (start evolving solutions)
        n = # of random agent invocations (# of generations) """

        agent_names = list(self.agents.keys())
        for i in range(n):
            pick = rnd.choice(agent_names)  # pick an agent to run
            self.run_agent(pick)
            if i % dom == 0:
                self.remove_dominated()
            if i % status == 0:
                self.remove_dominated()
                print("Iteration: ", i)
                print("Population size: ", len(self.pop))
                print(self)

        self.remove_dominated()


    def __str__(self):
        """ Output the solutions in the population """
        rslt = ""
        for eval,sol in self.pop.items():
            rslt += str(dict(eval))+":\t"+str(sol)+"\n"
        return rslt

# Load test data
def load_test_data(test_num):
    """Load test data from test files"""
    # Load the test data as a binary matrix
    solution = np.loadtxt(f'test{test_num}.csv', delimiter=',')
    return solution

def test_overallocation():
    """Test overallocation objective"""
    # Test case 1 should give score of 34
    solution1 = load_test_data(1)
    assert overallocation(solution1) == 34
    
    # Test case 2 should give score of 37
    solution2 = load_test_data(2)
    assert overallocation(solution2) == 37
    
    # Test case 3 should give score of 19
    solution3 = load_test_data(3)
    assert overallocation(solution3) == 19

def test_conflicts():
    """Test conflicts objective"""
    # Test case 1 should give score of 17
    solution1 = load_test_data(1)
    assert conflicts(solution1) == 17
    
    # Test case 2 should give score of 9
    solution2 = load_test_data(2)
    assert conflicts(solution2) == 9
    
    # Test case 3 should give score of 2
    solution3 = load_test_data(3)
    assert conflicts(solution3) == 2

def test_undersupport():
    """Test undersupport objective"""
    # Test case 1 should give score of 1
    solution1 = load_test_data(1)
    assert undersupport(solution1) == 1
    
    # Test case 2 should give score of 0
    solution2 = load_test_data(2)
    assert undersupport(solution2) == 0
    
    # Test case 3 should give score of 11
    solution3 = load_test_data(3)
    assert undersupport(solution3) == 11

def test_unavailable():
    """Test unavailable objective"""
    # Test case 1 should give score of 59
    solution1 = load_test_data(1)
    assert unavailable(solution1) == 59
    
    # Test case 2 should give score of 57
    solution2 = load_test_data(2)
    assert unavailable(solution2) == 57
    
    # Test case 3 should give score of 34
    solution3 = load_test_data(3)
    assert unavailable(solution3) == 34

def test_unpreferred():
    """Test unpreferred objective"""
    # Test case 1 should give score of 10
    solution1 = load_test_data(1)
    assert unpreferred(solution1) == 10
    
    # Test case 2 should give score of 16
    solution2 = load_test_data(2)
    assert unpreferred(solution2) == 16
    
    # Test case 3 should give score of 17
    solution3 = load_test_data(3)
    assert unpreferred(solution3) == 17



