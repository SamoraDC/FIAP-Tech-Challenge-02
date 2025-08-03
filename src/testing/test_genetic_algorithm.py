"""
Test script for Genetic Algorithm implementation.
Validates TSP solving capability with different configurations.
"""

import sys
sys.path.append('src')

from utils.data_loader import load_transportation_data
from algorithms.genetic_algorithm import GeneticAlgorithm, GeneticConfig
import numpy as np
import time

def test_basic_genetic_algorithm():
    """Test basic genetic algorithm functionality."""
    print("=== Basic Genetic Algorithm Test ===")
    
    # Load small sample for quick testing
    loader = load_transportation_data(sample_nodes=10)
    
    # Get coordinates and create distance matrix
    node_ids = list(loader.graph.nodes())
    coordinates = [loader.get_node_coordinates(nid) for nid in node_ids]
    
    from utils.distance_utils import DistanceCalculator
    calculator = DistanceCalculator(coordinates)
    distance_matrix = calculator.get_distance_matrix()
    
    print(f"✓ Testing with {len(coordinates)} cities")
    print(f"✓ Distance matrix shape: {distance_matrix.shape}")
    
    # Configure genetic algorithm for quick test
    config = GeneticConfig(
        population_size=20,
        generations=50,
        elite_size=5,
        mutation_rate=0.02,
        selection_method="tournament",
        crossover_method="order",
        mutation_method="swap"
    )
    
    # Create and run genetic algorithm
    ga = GeneticAlgorithm(distance_matrix, config)
    
    print(f"✓ Algorithm initialized with {ga.num_cities} cities")
    
    # Run algorithm
    start_time = time.time()
    best_solution = ga.run(verbose=True)
    run_time = time.time() - start_time
    
    print(f"✓ Algorithm completed in {run_time:.2f} seconds")
    print(f"✓ Best route distance: {best_solution.distance:.1f} meters")
    print(f"✓ Best route length: {len(best_solution.route)} cities")
    
    # Validate solution
    assert len(best_solution.route) == ga.num_cities, "Route must visit all cities"
    assert len(set(best_solution.route)) == ga.num_cities, "Route must visit each city exactly once"
    assert min(best_solution.route) == 0, "Route should contain city 0"
    assert max(best_solution.route) == ga.num_cities - 1, "Route should contain highest city index"
    
    print("✅ Basic genetic algorithm test passed!")
    return ga, best_solution

def test_different_operators():
    """Test different selection, crossover, and mutation operators."""
    print("\n=== Testing Different Operators ===")
    
    # Load sample data
    loader = load_transportation_data(sample_nodes=8)
    node_ids = list(loader.graph.nodes())
    coordinates = [loader.get_node_coordinates(nid) for nid in node_ids]
    
    from utils.distance_utils import DistanceCalculator
    calculator = DistanceCalculator(coordinates)
    distance_matrix = calculator.get_distance_matrix()
    
    # Test different operator combinations
    test_configs = [
        {"name": "Tournament + Order + Swap", "selection": "tournament", "crossover": "order", "mutation": "swap"},
        {"name": "Roulette + Cycle + Insert", "selection": "roulette", "crossover": "cycle", "mutation": "insert"},
        {"name": "Rank + PMX + Invert", "selection": "rank", "crossover": "pmx", "mutation": "invert"},
    ]
    
    results = []
    
    for test_config in test_configs:
        print(f"\nTesting: {test_config['name']}")
        
        config = GeneticConfig(
            population_size=15,
            generations=30,
            elite_size=3,
            mutation_rate=0.05,
            selection_method=test_config["selection"],
            crossover_method=test_config["crossover"],
            mutation_method=test_config["mutation"]
        )
        
        ga = GeneticAlgorithm(distance_matrix, config)
        
        start_time = time.time()
        best_solution = ga.run(verbose=False)
        run_time = time.time() - start_time
        
        results.append({
            'name': test_config['name'],
            'distance': best_solution.distance,
            'time': run_time,
            'convergence': ga._calculate_convergence_rate()
        })
        
        print(f"  ✓ Distance: {best_solution.distance:.1f}m, Time: {run_time:.2f}s, Convergence: {ga._calculate_convergence_rate():.3f}")
    
    # Compare results
    print(f"\n📊 Operator Comparison:")
    best_result = min(results, key=lambda x: x['distance'])
    for result in results:
        marker = "🏆" if result == best_result else "  "
        print(f"{marker} {result['name']}: {result['distance']:.1f}m ({result['time']:.2f}s)")
    
    print("✅ Operator testing completed!")
    return results

def test_performance_scaling():
    """Test algorithm performance with different problem sizes."""
    print("\n=== Performance Scaling Test ===")
    
    city_counts = [5, 8, 12]
    results = []
    
    for city_count in city_counts:
        print(f"\nTesting with {city_count} cities:")
        
        # Load data
        loader = load_transportation_data(sample_nodes=city_count)
        node_ids = list(loader.graph.nodes())
        coordinates = [loader.get_node_coordinates(nid) for nid in node_ids]
        
        from utils.distance_utils import DistanceCalculator
        calculator = DistanceCalculator(coordinates)
        distance_matrix = calculator.get_distance_matrix()
        
        # Adjust GA parameters based on problem size
        generations = max(30, city_count * 3)
        population = max(15, city_count * 2)
        
        config = GeneticConfig(
            population_size=population,
            generations=generations,
            elite_size=max(3, population // 5),
            mutation_rate=0.03,
            selection_method="tournament",
            crossover_method="order",
            mutation_method="swap"
        )
        
        ga = GeneticAlgorithm(distance_matrix, config)
        
        start_time = time.time()
        best_solution = ga.run(verbose=False)
        run_time = time.time() - start_time
        
        # Calculate statistics
        stats = ga.get_statistics()
        
        result = {
            'cities': len(coordinates),
            'distance': best_solution.distance,
            'time': run_time,
            'generations': generations,
            'convergence': stats['convergence_rate'],
            'final_diversity': stats['final_diversity']
        }
        results.append(result)
        
        print(f"  ✓ Cities: {result['cities']}, Distance: {result['distance']:.1f}m")
        print(f"  ✓ Time: {result['time']:.2f}s, Convergence: {result['convergence']:.3f}")
        print(f"  ✓ Final Diversity: {result['final_diversity']:.3f}")
    
    # Performance analysis
    print(f"\n📈 Performance Analysis:")
    for result in results:
        time_per_city = result['time'] / result['cities']
        print(f"  {result['cities']} cities: {result['time']:.2f}s total, {time_per_city:.3f}s per city")
    
    print("✅ Performance scaling test completed!")
    return results

def test_algorithm_convergence():
    """Test algorithm convergence behavior."""
    print("\n=== Convergence Test ===")
    
    # Load medium-sized problem
    loader = load_transportation_data(sample_nodes=10)
    node_ids = list(loader.graph.nodes())
    coordinates = [loader.get_node_coordinates(nid) for nid in node_ids]
    
    from utils.distance_utils import DistanceCalculator
    calculator = DistanceCalculator(coordinates)
    distance_matrix = calculator.get_distance_matrix()
    
    # Run algorithm with detailed tracking
    config = GeneticConfig(
        population_size=30,
        generations=100,
        elite_size=6,
        mutation_rate=0.02,
        selection_method="tournament",
        crossover_method="order",
        mutation_method="swap"
    )
    
    ga = GeneticAlgorithm(distance_matrix, config)
    
    print("Running convergence analysis...")
    best_solution = ga.run(verbose=False)
    
    # Analyze convergence
    history = ga.history
    initial_distance = history['best_distances'][0]
    final_distance = history['best_distances'][-1]
    improvement = initial_distance - final_distance
    
    print(f"✓ Initial best: {initial_distance:.1f}m")
    print(f"✓ Final best: {final_distance:.1f}m")
    print(f"✓ Total improvement: {improvement:.1f}m ({improvement/initial_distance*100:.1f}%)")
    print(f"✓ Convergence rate: {ga._calculate_convergence_rate():.3f}")
    
    # Check for premature convergence
    final_10_generations = history['best_distances'][-10:]
    improvement_in_final_10 = max(final_10_generations) - min(final_10_generations)
    
    if improvement_in_final_10 < 0.01 * initial_distance:
        print("⚠️  Algorithm may have converged prematurely")
    else:
        print("✓ Algorithm showed continued improvement")
    
    # Diversity analysis
    initial_diversity = history['diversity_scores'][0]
    final_diversity = history['diversity_scores'][-1]
    
    print(f"✓ Initial diversity: {initial_diversity:.3f}")
    print(f"✓ Final diversity: {final_diversity:.3f}")
    
    if final_diversity < 0.1:
        print("⚠️  Low final diversity - population may have lost variation")
    else:
        print("✓ Good final diversity maintained")
    
    print("✅ Convergence analysis completed!")
    return ga, history

def main():
    """Run all genetic algorithm tests."""
    print("FIAP Tech Challenge - Genetic Algorithm Tests")
    print("=" * 60)
    
    try:
        # Basic functionality test
        ga, best_solution = test_basic_genetic_algorithm()
        
        # Operator comparison test
        operator_results = test_different_operators()
        
        # Performance scaling test
        scaling_results = test_performance_scaling()
        
        # Convergence analysis
        convergence_ga, history = test_algorithm_convergence()
        
        print("\n" + "=" * 60)
        print("🎉 ALL GENETIC ALGORITHM TESTS PASSED!")
        print("🧬 Genetic Algorithm implementation is fully functional")
        print("🔬 Ready for comparison with other optimization methods")
        
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main()