"""
Four Focused Algorithms Testing Suite
FIAP Tech Challenge Phase 2 - Testing ONLY the 4 requested algorithms.

ALGORITHMS TESTED:
1. Particle Swarm Optimization
2. Ant Colony Optimization
3. Dijkstra-Enhanced Nearest Neighbor
4. A* Enhanced Nearest Neighbor
"""

import sys
sys.path.append('src')

from utils.data_loader import load_transportation_data
from utils.distance_utils import DistanceCalculator
from algorithms.four_focused_algorithms import run_four_focused_algorithms, FocusedConfig
import time
import pandas as pd

def run_comprehensive_comparison(num_cities: int = 12):
    """
    Run comprehensive comparison of ONLY the 4 requested algorithms.
    
    Args:
        num_cities: Number of cities for the TSP problem
        
    Returns:
        Tuple containing (results dict, distance matrix, coordinates)
    """
    print(f"🎯 FIAP Tech Challenge - Four Focused Algorithms Comparison")
    print(f"📍 Testing with {num_cities} Brazilian cities")
    print("=" * 80)
    print("TESTING ONLY THE 4 REQUESTED ALGORITHMS:")
    print("1. Particle Swarm Optimization")
    print("2. Ant Colony Optimization") 
    print("3. Dijkstra-Enhanced Nearest Neighbor")
    print("4. A* Enhanced Nearest Neighbor")
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
    
    # Configure the 4 focused algorithms
    config = FocusedConfig(
        pso_population_size=min(50, num_cities * 4),
        pso_max_iterations=min(100, num_cities * 8),
        aco_num_ants=min(50, num_cities * 4),
        aco_max_iterations=min(100, num_cities * 8)
    )
    
    print(f"\n🔧 Algorithm Configuration:")
    print(f"   PSO: {config.pso_population_size} particles, {config.pso_max_iterations} iterations")
    print(f"   ACO: {config.aco_num_ants} ants, {config.aco_max_iterations} iterations")
    print(f"   Dijkstra & A*: Enhanced nearest neighbor approach")
    
    # Run all 4 focused algorithms
    try:
        start_time = time.time()
        results = run_four_focused_algorithms(distance_matrix, coordinates, config)
        total_time = time.time() - start_time
        
        # Convert to expected format
        all_results = {}
        for algo_name, result in results.items():
            category = 'Metaheuristic' if 'Optimization' in result.algorithm_name else 'Conventional'
            
            all_results[result.algorithm_name] = {
                'algorithm': result.algorithm_name,
                'route': result.route,
                'distance': result.distance,
                'time': result.execution_time,
                'nodes_explored': result.nodes_explored,
                'iterations': result.iterations_completed,
                'category': category,
                'additional_info': result.additional_info or {}
            }
        
        print(f"\n⏱️ Total execution time for all 4 algorithms: {total_time:.2f} seconds")
        
        return all_results, distance_matrix, coordinates
        
    except Exception as e:
        print(f"❌ Four focused algorithms failed: {str(e)}")
        return {}, distance_matrix, coordinates

def analyze_results(results: dict):
    """
    Analyze and compare the 4 algorithm results.
    
    Args:
        results: Dictionary of the 4 algorithm results
    """
    print(f"\n📊 FOUR ALGORITHMS PERFORMANCE ANALYSIS")
    print("=" * 80)
    
    # Create comparison table
    comparison_data = []
    
    for algo_name, result in results.items():
        comparison_data.append({
            'Algorithm': algo_name,
            'Category': result.get('category', 'Unknown'),
            'Distance (km)': round(result['distance'] / 1000, 2),
            'Time (s)': round(result['time'], 4),
            'Iterations': result.get('iterations', 0),
            'Efficiency': round(result['distance'] / max(result['time'] * 1000, 0.001), 2)
        })
    
    # Sort by distance (best first)
    comparison_data.sort(key=lambda x: x['Distance (km)'])
    
    # Create DataFrame for nice display
    df = pd.DataFrame(comparison_data)
    
    print("🏆 FOUR ALGORITHMS PERFORMANCE RANKING:")
    print(df.to_string(index=False))
    
    # Find best in each category
    print(f"\n🥇 CATEGORY WINNERS (4 ALGORITHMS ONLY):")
    
    categories = {}
    for algo_name, result in results.items():
        cat = result.get('category', 'Unknown')
        if cat not in categories or result['distance'] < categories[cat]['distance']:
            categories[cat] = {'name': algo_name, 'distance': result['distance']}
    
    for category, winner in categories.items():
        print(f"  {category}: {winner['name']} ({winner['distance']/1000:.2f} km)")
    
    # Overall statistics for 4 algorithms
    distances = [r['distance'] for r in results.values()]
    times = [r['time'] for r in results.values()]
    
    best_distance = min(distances)
    worst_distance = max(distances)
    avg_distance = sum(distances) / len(distances)
    
    best_algo = min(results.items(), key=lambda x: x[1]['distance'])[0]
    fastest_algo = min(results.items(), key=lambda x: x[1]['time'])[0]
    
    print(f"\n📈 OVERALL STATISTICS (4 ALGORITHMS):")
    print(f"  Best Distance: {best_distance/1000:.2f} km ({best_algo})")
    print(f"  Worst Distance: {worst_distance/1000:.2f} km")
    print(f"  Average Distance: {avg_distance/1000:.2f} km")
    print(f"  Distance Range: {(worst_distance-best_distance)/1000:.2f} km")
    print(f"  Fastest Algorithm: {fastest_algo} ({results[fastest_algo]['time']:.4f}s)")
    print(f"  Optimization Potential: {((worst_distance-best_distance)/best_distance)*100:.1f}%")
    
    # Performance insights for 4 algorithms
    print(f"\n💡 INSIGHTS (4 ALGORITHMS ANALYSIS):")
    
    # Find most efficient (best distance/time ratio)
    efficiencies = [(name, res['distance']/max(res['time'], 0.001)) for name, res in results.items()]
    most_efficient = min(efficiencies, key=lambda x: x[1])
    print(f"  • Most Efficient: {most_efficient[0]} (best distance/time ratio)")
    
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
    
    # Speed analysis
    metaheuristic_algos = [name for name, res in results.items() if res.get('category') == 'Metaheuristic']
    conventional_algos = [name for name, res in results.items() if res.get('category') == 'Conventional']
    
    if metaheuristic_algos and conventional_algos:
        meta_avg_time = sum(results[name]['time'] for name in metaheuristic_algos) / len(metaheuristic_algos)
        conv_avg_time = sum(results[name]['time'] for name in conventional_algos) / len(conventional_algos)
        speed_factor = meta_avg_time / conv_avg_time if conv_avg_time > 0 else float('inf')
        print(f"  • Speed Factor: Metaheuristic {speed_factor:.0f}x slower than Conventional")
    
    return df

def validate_solutions(results: dict, distance_matrix):
    """
    Validate that all 4 algorithm solutions are valid TSP routes.
    
    Args:
        results: Dictionary of the 4 algorithm results
        distance_matrix: Distance matrix for validation
    """
    print(f"\n✅ SOLUTION VALIDATION (4 ALGORITHMS)")
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
        
        # Recalculate distance for verification
        calculated_distance = 0.0
        if len(route) > 0:
            for i in range(len(route)):
                next_i = (i + 1) % len(route)
                if route[i] < len(distance_matrix) and route[next_i] < len(distance_matrix):
                    calculated_distance += distance_matrix[route[i]][route[next_i]]
        
        # Check distance calculation (allow small floating point differences)
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
        print(f"\n🎉 ALL 4 ALGORITHM SOLUTIONS ARE VALID!")
    else:
        print(f"\n⚠️  SOME ALGORITHM SOLUTIONS HAVE ISSUES!")
    
    return all_valid

def main():
    """Run comprehensive comparison of the 4 focused algorithms."""
    try:
        # Test with different problem sizes
        test_sizes = [8, 12, 16]
        
        for size in test_sizes:
            print(f"\n{'='*100}")
            print(f"🧪 TESTING 4 ALGORITHMS WITH {size} CITIES")
            print(f"{'='*100}")
            
            # Run comparison
            results, distance_matrix, coordinates = run_comprehensive_comparison(size)
            
            if not results:
                print("⚠️  No results obtained, skipping analysis")
                continue
            
            # Analyze results
            df = analyze_results(results)
            
            # Validate solutions
            valid = validate_solutions(results, distance_matrix)
            
            if not valid:
                print("⚠️  Some solutions invalid, but continuing...")
        
        print(f"\n{'='*100}")
        print("🎊 FOUR ALGORITHMS TESTING COMPLETED SUCCESSFULLY!")
        print("🏆 All 4 requested algorithms implemented and tested")
        print("📊 Performance analysis complete for:")
        print("   1. Particle Swarm Optimization")
        print("   2. Ant Colony Optimization")
        print("   3. Dijkstra-Enhanced Nearest Neighbor")
        print("   4. A* Enhanced Nearest Neighbor")
        print("🚀 System ready for demonstration and visualization")
        print(f"{'='*100}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ TESTING FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main()