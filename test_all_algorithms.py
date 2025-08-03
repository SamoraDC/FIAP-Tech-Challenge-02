"""
Comprehensive algorithm comparison test.
Tests all TSP optimization algorithms and compares their performance.
FIAP Tech Challenge Phase 2 - Multi-Algorithm Route Optimization.
"""

import sys
sys.path.append('src')

from utils.data_loader import load_transportation_data
from utils.distance_utils import DistanceCalculator
from algorithms.genetic_algorithm import GeneticAlgorithm, GeneticConfig
from algorithms.conventional_algorithms import run_conventional_algorithms
from algorithms.metaheuristic_algorithms import run_metaheuristic_algorithms, MetaheuristicConfig
import time
import pandas as pd

def run_comprehensive_comparison(num_cities: int = 12):
    """
    Run comprehensive comparison of all algorithms.
    
    Args:
        num_cities: Number of cities for the TSP problem
        
    Returns:
        Dictionary containing all results
    """
    print(f"🌟 FIAP Tech Challenge - Comprehensive Algorithm Comparison")
    print(f"📍 Testing with {num_cities} Brazilian cities")
    print("=" * 80)
    
    # Load data
    print("Loading Brazilian transportation data...")
    loader = load_transportation_data(sample_nodes=num_cities)
    node_ids = list(loader.graph.nodes())
    coordinates = [loader.get_node_coordinates(nid) for nid in node_ids]
    
    # Create distance calculator
    calculator = DistanceCalculator(coordinates)
    distance_matrix = calculator.get_distance_matrix()
    
    print(f"✅ Loaded {len(coordinates)} cities")
    print(f"✅ Distance matrix: {distance_matrix.shape}")
    print(f"✅ Graph connectivity: {len(loader.graph.edges)} edges")
    
    all_results = {}
    
    # 1. GENETIC ALGORITHM (Main Focus)
    print(f"\n🧬 GENETIC ALGORITHM (Main Focus)")
    print("=" * 50)
    
    ga_config = GeneticConfig(
        population_size=50,
        generations=100,
        elite_size=10,
        mutation_rate=0.02,
        selection_method="tournament",
        crossover_method="order",
        mutation_method="swap"
    )
    
    start_time = time.time()
    ga = GeneticAlgorithm(distance_matrix, ga_config)
    ga_best = ga.run(verbose=True)
    ga_time = time.time() - start_time
    
    all_results['Genetic Algorithm'] = {
        'algorithm': 'Genetic Algorithm',
        'route': ga_best.route,
        'distance': ga_best.distance,
        'time': ga_time,
        'convergence_rate': ga._calculate_convergence_rate(),
        'diversity': ga.history['diversity_scores'][-1],
        'category': 'Evolutionary'
    }
    
    # 2. CONVENTIONAL ALGORITHMS
    print(f"\n🔍 CONVENTIONAL ALGORITHMS")
    print("=" * 50)
    
    start_time = time.time()
    conventional_results = run_conventional_algorithms(distance_matrix, coordinates)
    conventional_time = time.time() - start_time
    
    for result in conventional_results:
        all_results[result.algorithm_name] = {
            'algorithm': result.algorithm_name,
            'route': result.route,
            'distance': result.distance,
            'time': result.execution_time,
            'nodes_explored': result.nodes_explored,
            'category': 'Conventional'
        }
    
    # 3. METAHEURISTIC ALGORITHMS
    print(f"\n🐜 METAHEURISTIC ALGORITHMS")
    print("=" * 50)
    
    meta_config = MetaheuristicConfig(
        population_size=30,
        max_iterations=50,
        num_ants=30
    )
    
    start_time = time.time()
    metaheuristic_results = run_metaheuristic_algorithms(distance_matrix, meta_config)
    meta_time = time.time() - start_time
    
    for algo_name, result in metaheuristic_results.items():
        all_results[result['algorithm']] = {
            'algorithm': result['algorithm'],
            'route': result['route'],
            'distance': result['distance'],
            'time': meta_time / 2,  # Approximate time per algorithm
            'category': 'Metaheuristic'
        }
    
    return all_results, distance_matrix, coordinates

def analyze_results(results: dict):
    """
    Analyze and compare algorithm results.
    
    Args:
        results: Dictionary of algorithm results
    """
    print(f"\n📊 PERFORMANCE ANALYSIS")
    print("=" * 80)
    
    # Create comparison table
    comparison_data = []
    
    for algo_name, result in results.items():
        comparison_data.append({
            'Algorithm': algo_name,
            'Category': result.get('category', 'Unknown'),
            'Distance (km)': round(result['distance'] / 1000, 2),
            'Time (s)': round(result['time'], 4),
            'Efficiency': round(result['distance'] / max(result['time'] * 1000, 0.001), 2)
        })
    
    # Sort by distance (best first)
    comparison_data.sort(key=lambda x: x['Distance (km)'])
    
    # Create DataFrame for nice display
    df = pd.DataFrame(comparison_data)
    
    print("🏆 ALGORITHM PERFORMANCE RANKING:")
    print(df.to_string(index=False))
    
    # Find best in each category
    print(f"\n🥇 CATEGORY WINNERS:")
    
    categories = {}
    for algo_name, result in results.items():
        cat = result.get('category', 'Unknown')
        if cat not in categories or result['distance'] < categories[cat]['distance']:
            categories[cat] = {'name': algo_name, 'distance': result['distance']}
    
    for category, winner in categories.items():
        print(f"  {category}: {winner['name']} ({winner['distance']/1000:.2f} km)")
    
    # Overall statistics
    distances = [r['distance'] for r in results.values()]
    times = [r['time'] for r in results.values()]
    
    best_distance = min(distances)
    worst_distance = max(distances)
    avg_distance = sum(distances) / len(distances)
    
    best_algo = min(results.items(), key=lambda x: x[1]['distance'])[0]
    fastest_algo = min(results.items(), key=lambda x: x[1]['time'])[0]
    
    print(f"\n📈 OVERALL STATISTICS:")
    print(f"  Best Distance: {best_distance/1000:.2f} km ({best_algo})")
    print(f"  Worst Distance: {worst_distance/1000:.2f} km")
    print(f"  Average Distance: {avg_distance/1000:.2f} km")
    print(f"  Distance Range: {(worst_distance-best_distance)/1000:.2f} km")
    print(f"  Fastest Algorithm: {fastest_algo} ({results[fastest_algo]['time']:.4f}s)")
    
    # Performance insights
    print(f"\n💡 INSIGHTS:")
    
    # Quality vs Speed analysis
    genetic_result = results.get('Genetic Algorithm')
    if genetic_result:
        ga_rank_distance = sorted(distances).index(genetic_result['distance']) + 1
        ga_rank_time = sorted(times, reverse=True).index(genetic_result['time']) + 1
        print(f"  • Genetic Algorithm: #{ga_rank_distance} in quality, #{ga_rank_time} in speed")
    
    # Find most efficient (best distance/time ratio)
    efficiencies = [(name, res['distance']/max(res['time'], 0.001)) for name, res in results.items()]
    most_efficient = min(efficiencies, key=lambda x: x[1])
    print(f"  • Most Efficient: {most_efficient[0]} (lowest distance/time ratio)")
    
    # Category performance
    cat_performance = {}
    for algo_name, result in results.items():
        cat = result.get('category', 'Unknown')
        if cat not in cat_performance:
            cat_performance[cat] = []
        cat_performance[cat].append(result['distance'])
    
    for category, distances_cat in cat_performance.items():
        avg_cat = sum(distances_cat) / len(distances_cat)
        print(f"  • {category} Average: {avg_cat/1000:.2f} km ({len(distances_cat)} algorithms)")
    
    return df

def validate_solutions(results: dict, distance_matrix):
    """
    Validate that all solutions are valid TSP routes.
    
    Args:
        results: Dictionary of algorithm results
        distance_matrix: Distance matrix for validation
    """
    print(f"\n✅ SOLUTION VALIDATION")
    print("=" * 50)
    
    num_cities = len(distance_matrix)
    all_valid = True
    
    for algo_name, result in results.items():
        route = result['route']
        distance = result['distance']
        
        # Check route validity
        valid = True
        issues = []
        
        # Check route length
        if len(route) != num_cities:
            valid = False
            issues.append(f"Wrong length: {len(route)} != {num_cities}")
        
        # Check all cities visited exactly once
        if len(set(route)) != num_cities:
            valid = False
            issues.append("Cities not visited exactly once")
        
        # Check city indices in valid range
        if min(route) < 0 or max(route) >= num_cities:
            valid = False
            issues.append("Invalid city indices")
        
        # Recalculate distance for verification using the same method
        calculated_distance = 0.0
        if len(route) > 0:
            for i in range(len(route)):
                next_i = (i + 1) % len(route)
                if route[i] < len(distance_matrix) and route[next_i] < len(distance_matrix):
                    calculated_distance += distance_matrix[route[i]][route[next_i]]
        
        # Check distance calculation (allow larger floating point differences)
        distance_diff = abs(calculated_distance - distance)
        if distance_diff > max(0.1, distance * 0.001):  # Allow 0.1% error or 0.1 units
            valid = False
            issues.append(f"Distance mismatch: {calculated_distance:.1f} vs {distance:.1f} (diff: {distance_diff:.1f})")
        
        # Print validation result
        status = "✅ VALID" if valid else "❌ INVALID"
        print(f"  {algo_name}: {status}")
        
        if not valid:
            all_valid = False
            for issue in issues:
                print(f"    - {issue}")
    
    if all_valid:
        print(f"\n🎉 ALL SOLUTIONS ARE VALID!")
    else:
        print(f"\n⚠️  SOME SOLUTIONS HAVE ISSUES!")
    
    return all_valid

def main():
    """Run comprehensive algorithm comparison."""
    try:
        # Test with different problem sizes
        test_sizes = [8, 12]
        
        for size in test_sizes:
            print(f"\n{'='*100}")
            print(f"🧪 TESTING WITH {size} CITIES")
            print(f"{'='*100}")
            
            # Run comparison
            results, distance_matrix, coordinates = run_comprehensive_comparison(size)
            
            # Analyze results
            df = analyze_results(results)
            
            # Validate solutions
            valid = validate_solutions(results, distance_matrix)
            
            if not valid:
                print("⚠️  Stopping due to invalid solutions")
                return False
        
        print(f"\n{'='*100}")
        print("🎊 COMPREHENSIVE TESTING COMPLETED SUCCESSFULLY!")
        print("🏆 All algorithms implemented and validated")
        print("📊 Performance analysis complete")
        print("🚀 System ready for demonstration and video")
        print(f"{'='*100}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ TESTING FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main()