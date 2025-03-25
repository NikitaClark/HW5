# TA Assignment Optimization

This project implements a multi-objective optimization algorithm for assigning Teaching Assistants (TAs) to lab sections while minimizing various constraints and preferences.

## Project Structure

- `assignta.py`: Contains the objective functions and optimization agents
- `evo.py`: Implements the evolutionary algorithm
- `profiler.py`: Runs and profiles the optimization process
- `test_assignta.py`: Contains test cases for the objective functions
- `sections.csv`: Input data for lab sections
- `tas.csv`: Input data for TAs
- `test1.csv`, `test2.csv`, `test3.csv`: Test data files

## Objective Functions

The optimization aims to minimize five objectives:
1. Overallocation: Number of sections assigned beyond a TA's maximum capacity
2. Conflicts: Number of times a TA is assigned to multiple sections at the same time
3. Undersupport: Number of sections without sufficient TA support
4. Unavailable: Number of times a TA is assigned to sections they marked as unavailable
5. Unpreferred: Number of times a TA is assigned to sections they marked as unpreferred

## Usage

1. Run the optimization:
```bash
python profiler.py
```

2. Run the tests:
```bash
python -m pytest test_assignta.py -v
```

## Output Files

The optimization generates three main output files:
1. `nikit_summary.csv`: Contains the optimization results with scores for each objective
2. `nikit_pytest.txt`: Contains the test results
3. `nikit_profile.txt`: Contains profiling information about the optimization process

## Optimization Process

The optimization runs for 5 minutes using an evolutionary algorithm that:
- Maintains a population of non-dominated solutions
- Uses various agents to generate new solutions:
  - `swap_ta_agent`: Swaps TA assignments between sections
  - `add_remove_agent`: Adds or removes TA assignments
  - `optimize_section_agent`: Optimizes assignments for a specific section
  - `fix_conflicts_agent`: Attempts to resolve scheduling conflicts

## Dependencies

- Python 3.x
- NumPy
- Pandas
- Pytest 