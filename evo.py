"""
File: evo.py
Description: A concise evolutionary computing framework for solving
            multi-objective optimization problems!
"""

import random as rnd
import copy   # doing deep copies of solutions when generating offspring
from functools import reduce  # for discarding dominated (bad) solutions
import time
import pandas as pd

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

    

    def evolve(self, time_limit=300, dom=100, status=1000):
        """ Run the framework (start evolving solutions)
        time_limit = time limit in seconds (default 5 minutes)
        dom = how often to remove dominated solutions
        status = how often to print status updates """
        
        start_time = time.time()
        agent_names = list(self.agents.keys())
        iteration = 0
        
        while time.time() - start_time < time_limit:
            pick = rnd.choice(agent_names)  # pick an agent to run
            self.run_agent(pick)
            
            if iteration % dom == 0:
                self.remove_dominated()
            
            if iteration % status == 0:
                elapsed = time.time() - start_time
                self.remove_dominated()
                print(f"Time elapsed: {elapsed:.1f}s")
                print(f"Iteration: {iteration}")
                print(f"Population size: {len(self.pop)}")
                print(self)
            
            iteration += 1
        
        self.remove_dominated()
        total_time = time.time() - start_time
        print(f"\nEvolution completed in {total_time:.1f} seconds")
        print(f"Total iterations: {iteration}")
        print(f"Final population size: {len(self.pop)}")

    def summarize(self):
        """Convert the population to a summary table format"""
        if not self.pop:
            return pd.DataFrame()
        
        # Extract objective names from the first solution's evaluation
        obj_names = [name for name, _ in list(self.pop.keys())[0]]
        
        # Create rows for each solution
        rows = []
        for eval_key, solution in self.pop.items():
            row = {name: score for (name, score) in eval_key}
            rows.append(row)
        
        # Convert to DataFrame
        df = pd.DataFrame(rows)
        return df

    def __str__(self):
        """ Output the solutions in the population """
        rslt = ""
        for eval,sol in self.pop.items():
            rslt += str(dict(eval))+":\t"+str(sol)+"\n"
        return rslt



