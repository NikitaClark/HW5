import cProfile
import pstats
from pstats import SortKey
import time
from evo import Evo
from assignta import (
    overallocation,
    conflicts,
    undersupport,
    unavailable,
    unpreferred,
    swap_ta_agent,
    add_remove_agent,
    optimize_section_agent,
    fix_conflicts_agent
)

def setup_optimizer():
    """Set up the evolutionary optimizer with objectives and agents"""
    optimizer = Evo()
    
    # Add objectives
    optimizer.add_objective('overallocation', overallocation)
    optimizer.add_objective('conflicts', conflicts)
    optimizer.add_objective('undersupport', undersupport)
    optimizer.add_objective('unavailable', unavailable)
    optimizer.add_objective('unpreferred', unpreferred)
    
    # Add agents
    optimizer.add_agent('swap_ta', swap_ta_agent)
    optimizer.add_agent('add_remove', add_remove_agent)
    optimizer.add_agent('optimize_section', optimize_section_agent)
    optimizer.add_agent('fix_conflicts', fix_conflicts_agent)
    
    return optimizer

def main():
    """Main function to run and profile the optimization"""
    # Create and set up optimizer
    optimizer = setup_optimizer()
    
    # Profile the evolution
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Run evolution for 5 minutes
    start_time = time.time()
    optimizer.evolve(time_limit=300)  # 5 minutes
    total_time = time.time() - start_time
    
    profiler.disable()
    
    # Save profiling results
    stats = pstats.Stats(profiler)
    stats.sort_stats(SortKey.TIME)
    
    # Save profiling information to a separate file
    with open('nikit_profile.txt', 'w') as f:
        f.write("Optimization Profiling Report\n")
        f.write("===========================\n\n")
        f.write(f"Total runtime: {total_time:.2f} seconds\n\n")
        f.write("Top 20 time-consuming functions:\n")
        f.write("--------------------------------\n")
        stats.stream = f
        stats.print_stats(20)
        
        f.write("\nDetailed function call statistics:\n")
        f.write("--------------------------------\n")
        stats.print_callers()
    
    # Save final solutions
    solutions = optimizer.summarize()
    solutions.to_csv('nikit_summary.csv', index=False)

if __name__ == '__main__':
    main()
