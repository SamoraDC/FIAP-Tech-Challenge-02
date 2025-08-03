"""
Matplotlib Visualization Demo (Pygame-independent)
FIAP Tech Challenge Phase 2 - Algorithm visualization with matplotlib only.
"""

import sys
sys.path.append('src')

from utils.data_loader import load_transportation_data
from utils.distance_utils import DistanceCalculator
from algorithms.genetic_algorithm import GeneticAlgorithm, GeneticConfig
from algorithms.conventional_algorithms import run_conventional_algorithms
from algorithms.metaheuristic_algorithms import run_metaheuristic_algorithms, MetaheuristicConfig
from visualization.convergence_plotter import ConvergencePlotter

import time
import os
from typing import Dict, List, Tuple

def run_algorithm_comparison(num_cities: int = 10) -> Tuple[Dict, Dict, List, Dict]:
    """
    Run comprehensive algorithm comparison and collect data for visualization.
    
    Args:
        num_cities: Number of cities for the TSP problem
        
    Returns:
        Tuple of (results, histories, coordinates, routes)
    """
    print(f"🚀 Running algorithm comparison with {num_cities} cities...")
    
    # Load data
    loader = load_transportation_data(sample_nodes=num_cities)
    node_ids = list(loader.graph.nodes())
    coordinates = [loader.get_node_coordinates(nid) for nid in node_ids]
    
    # Create distance calculator
    calculator = DistanceCalculator(coordinates)
    distance_matrix = calculator.get_distance_matrix()
    
    print(f"✅ Loaded {len(coordinates)} cities from Brazilian transportation network")
    
    all_results = {}
    all_histories = {}
    all_routes = {}
    
    # 1. GENETIC ALGORITHM (with detailed history tracking)
    print("\n🧬 Running Genetic Algorithm...")
    ga_config = GeneticConfig(
        population_size=50,
        generations=80,
        elite_size=10,
        mutation_rate=0.02,
        selection_method="tournament",
        crossover_method="order",
        mutation_method="swap"
    )
    
    start_time = time.time()
    ga = GeneticAlgorithm(distance_matrix, ga_config)
    ga_best = ga.run(verbose=False)
    ga_time = time.time() - start_time
    
    all_results['Genetic Algorithm'] = {
        'algorithm': 'Genetic Algorithm',
        'route': ga_best.route,
        'distance': ga_best.distance,
        'time': ga_time,
        'category': 'Evolutionary'
    }
    all_histories['Genetic Algorithm'] = ga.history
    all_routes['Genetic Algorithm'] = ga_best.route
    
    print(f"✅ GA completed: {ga_best.distance/1000:.2f} km in {ga_time:.3f}s")
    
    # 2. CONVENTIONAL ALGORITHMS
    print("\n🔍 Running Conventional Algorithms...")
    start_time = time.time()
    conventional_results = run_conventional_algorithms(distance_matrix, coordinates)
    conventional_time = time.time() - start_time
    
    for result in conventional_results:
        all_results[result.algorithm_name] = {
            'algorithm': result.algorithm_name,
            'route': result.route,
            'distance': result.distance,
            'time': result.execution_time,
            'category': 'Conventional'
        }
        all_routes[result.algorithm_name] = result.route
    
    print(f"✅ Conventional algorithms completed in {conventional_time:.3f}s")
    
    # 3. METAHEURISTIC ALGORITHMS
    print("\n🐜 Running Metaheuristic Algorithms...")
    meta_config = MetaheuristicConfig(
        population_size=30,
        max_iterations=40,
        num_ants=30
    )
    
    start_time = time.time()
    metaheuristic_results = run_metaheuristic_algorithms(distance_matrix, meta_config)
    meta_time = time.time() - start_time
    
    # Process PSO results
    pso_result = metaheuristic_results['PSO']
    all_results['Particle Swarm Optimization'] = {
        'algorithm': 'Particle Swarm Optimization',
        'route': pso_result['route'],
        'distance': pso_result['distance'],
        'time': meta_time / 2,
        'category': 'Metaheuristic'
    }
    all_routes['Particle Swarm Optimization'] = pso_result['route']
    all_histories['Particle Swarm Optimization'] = pso_result['history']
    
    # Process ACO results
    aco_result = metaheuristic_results['ACO']
    all_results['Ant Colony Optimization'] = {
        'algorithm': 'Ant Colony Optimization',
        'route': aco_result['route'],
        'distance': aco_result['distance'],
        'time': meta_time / 2,
        'category': 'Metaheuristic'
    }
    all_routes['Ant Colony Optimization'] = aco_result['route']
    all_histories['Ant Colony Optimization'] = aco_result['history']
    
    print(f"✅ Metaheuristic algorithms completed in {meta_time:.3f}s")
    
    return all_results, all_histories, coordinates, all_routes

def create_matplotlib_analysis(results: Dict, histories: Dict, 
                              coordinates: List, routes: Dict):
    """
    Create comprehensive matplotlib analysis plots.
    
    Args:
        results: Algorithm results
        histories: Algorithm histories
        coordinates: City coordinates  
        routes: Algorithm routes
    """
    print("\n📊 Creating comprehensive performance analysis...")
    
    plotter = ConvergencePlotter(figsize=(15, 10))
    
    # 1. Algorithm Performance Comparison
    print("   📈 Generating algorithm comparison plots...")
    plotter.plot_algorithm_comparison(results, "results/algorithm_comparison.png", show=True)
    
    # 2. Convergence Analysis
    print("   📉 Generating convergence analysis...")
    evolutionary_histories = {k: v for k, v in histories.items() 
                            if 'best_distances' in v or 'best_fitness' in v}
    if evolutionary_histories:
        plotter.plot_convergence_history(evolutionary_histories, "results/convergence_analysis.png", show=True)
    
    # 3. Route Comparison Maps
    print("   🗺️  Generating route comparison maps...")
    # Select best routes from each category for comparison
    best_routes = {}
    categories = {}
    
    for algo_name, result in results.items():
        category = result['category']
        if category not in categories or result['distance'] < categories[category]['distance']:
            categories[category] = {'name': algo_name, 'distance': result['distance']}
    
    for category_data in categories.values():
        algo_name = category_data['name']
        best_routes[algo_name] = routes[algo_name]
    
    plotter.plot_route_comparison(coordinates, best_routes, "results/route_comparison.png", show=True)
    
    # 4. Summary Report
    print("   📋 Generating summary report...")
    plotter.create_summary_report(results, histories, "results/summary_report")
    
    print("✅ Matplotlib analysis completed! Check 'results/' folder for plots.")

def display_final_summary(results: Dict):
    """
    Display final performance summary.
    
    Args:
        results: Algorithm results
    """
    print("\n" + "="*80)
    print("🏆 FINAL ALGORITHM PERFORMANCE SUMMARY")
    print("="*80)
    
    # Sort by distance (best first)
    sorted_results = sorted(results.items(), key=lambda x: x[1]['distance'])
    
    print(f"{'Rank':<4} {'Algorithm':<30} {'Distance (km)':<12} {'Time (s)':<10} {'Category':<15}")
    print("-" * 80)
    
    for i, (algo_name, result) in enumerate(sorted_results, 1):
        distance_km = result['distance'] / 1000
        time_s = result['time']
        category = result['category']
        
        # Add medal emoji for top 3
        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
        
        print(f"{medal} {i:<2} {algo_name:<30} {distance_km:<12.2f} {time_s:<10.4f} {category:<15}")
    
    # Category winners
    print("\n🏅 CATEGORY CHAMPIONS:")
    categories = {}
    for algo_name, result in results.items():
        cat = result['category']
        if cat not in categories or result['distance'] < categories[cat]['distance']:
            categories[cat] = {'name': algo_name, 'distance': result['distance']}
    
    for category, winner in categories.items():
        print(f"   {category}: {winner['name']} ({winner['distance']/1000:.2f} km)")
    
    # Best overall
    best_algo, best_result = sorted_results[0]
    print(f"\n🎯 OVERALL CHAMPION: {best_algo}")
    print(f"   Distance: {best_result['distance']/1000:.2f} km")
    print(f"   Time: {best_result['time']:.4f} seconds")
    print(f"   Category: {best_result['category']}")
    
    # Performance insights
    print(f"\n💡 PERFORMANCE INSIGHTS:")
    
    # Find genetic algorithm position
    ga_position = next((i for i, (name, _) in enumerate(sorted_results, 1) if 'Genetic' in name), None)
    if ga_position:
        print(f"   • Genetic Algorithm ranked #{ga_position} out of {len(results)} algorithms")
    
    # Calculate improvement range
    best_distance = sorted_results[0][1]['distance']
    worst_distance = sorted_results[-1][1]['distance']
    improvement_potential = (worst_distance - best_distance) / best_distance * 100
    print(f"   • Optimization potential: {improvement_potential:.1f}% improvement from worst to best")
    
    # Speed vs Quality analysis
    fastest_algo = min(results.items(), key=lambda x: x[1]['time'])
    print(f"   • Speed champion: {fastest_algo[0]} ({fastest_algo[1]['time']:.4f}s)")
    
    return sorted_results

def create_results_directory():
    """Create results directory for saving plots."""
    os.makedirs('results', exist_ok=True)
    print("📁 Created 'results/' directory for saving plots")

def main():
    """Run comprehensive matplotlib visualization demo."""
    print("🌟 FIAP Tech Challenge - Matplotlib Visualization Demo")
    print("=" * 80)
    
    try:
        # Create results directory
        create_results_directory()
        
        # Test with different problem sizes
        test_sizes = [8, 12]
        
        for size in test_sizes:
            print(f"\n{'🧪 TESTING WITH ' + str(size) + ' CITIES':=^80}")
            
            # Run algorithm comparison
            results, histories, coordinates, routes = run_algorithm_comparison(num_cities=size)
            
            # Display performance summary
            sorted_results = display_final_summary(results)
            
            # Create matplotlib analysis
            create_matplotlib_analysis(results, histories, coordinates, routes)
            
            print(f"\n✅ Analysis completed for {size} cities!")
            
            if size < max(test_sizes):
                input("\n⏭️  Press Enter to continue to next test size...")
        
        print("\n" + "="*80)
        print("🎊 MATPLOTLIB VISUALIZATION DEMO COMPLETED!")
        print("✅ All algorithm implementations tested and visualized")
        print("📊 Performance analysis plots generated and displayed")
        print("🗺️  Route comparison maps created")
        print("📈 Convergence analysis completed")
        print("💾 All plots saved in 'results/' folder")
        print("🚀 System ready for Tech Challenge demonstration")
        print("="*80)
        
        return True
        
    except Exception as e:
        print(f"\n❌ DEMO FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main()