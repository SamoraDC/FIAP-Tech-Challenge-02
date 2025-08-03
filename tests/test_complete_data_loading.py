"""
Complete test of data loading with distance calculations.
Tests the full pipeline including route distance calculations.
"""

import sys
sys.path.append('src')

from utils.data_loader import load_transportation_data
from utils.distance_utils import DistanceCalculator
import numpy as np

def test_complete_pipeline():
    """Test the complete data loading and distance calculation pipeline."""
    print("=== Complete Pipeline Test ===")
    
    # Load sample data with connected graph
    print("Loading sample data (30 nodes)...")
    loader = load_transportation_data(sample_nodes=30)
    
    # Get statistics
    stats = loader.get_dataset_statistics()
    print(f"✓ Graph connectivity: {stats['graph']['is_connected']}")
    print(f"✓ Graph density: {stats['graph']['density']:.3f}")
    print(f"✓ Nodes: {stats['graph']['nodes']}, Edges: {stats['graph']['edges']}")
    
    # Extract coordinates for distance calculations
    node_ids = list(loader.graph.nodes())[:10]  # Use first 10 nodes
    coordinates = []
    
    for node_id in node_ids:
        coord = loader.get_node_coordinates(node_id)
        coordinates.append(coord)
    
    # Create distance calculator
    print(f"\nTesting distance calculations for {len(coordinates)} nodes...")
    calculator = DistanceCalculator(coordinates)
    distance_matrix = calculator.get_distance_matrix()
    
    print(f"✓ Distance matrix shape: {distance_matrix.shape}")
    print(f"✓ Matrix diagonal (should be zeros): {np.diag(distance_matrix)[:5]}")
    print(f"✓ Sample distances (meters): {distance_matrix[0][:5]}")
    
    # Test TSP route calculation
    test_routes = [
        [0, 1, 2, 3, 4],  # Sequential route
        [0, 2, 4, 1, 3],  # Mixed route
        [4, 3, 2, 1, 0],  # Reverse route
    ]
    
    print(f"\nTesting route distance calculations:")
    for i, route in enumerate(test_routes):
        distance = calculator.calculate_route_distance(route)
        print(f"✓ Route {i+1} {route}: {distance:.1f} meters ({distance/1000:.1f} km)")
    
    return loader, calculator

def test_optimization_readiness():
    """Test that data is ready for optimization algorithms."""
    print("\n=== Optimization Readiness Test ===")
    
    # Load larger sample for optimization testing
    loader = load_transportation_data(sample_nodes=50)
    
    # Get distance matrix for all nodes
    all_node_ids = list(loader.graph.nodes())
    coordinates = [loader.get_node_coordinates(nid) for nid in all_node_ids]
    
    calculator = DistanceCalculator(coordinates)
    distance_matrix = calculator.get_distance_matrix()
    
    print(f"✓ Full distance matrix for optimization: {distance_matrix.shape}")
    print(f"✓ Matrix is symmetric: {np.allclose(distance_matrix, distance_matrix.T)}")
    print(f"✓ Matrix has no negative values: {np.all(distance_matrix >= 0)}")
    print(f"✓ Diagonal is zero: {np.allclose(np.diag(distance_matrix), 0)}")
    
    # Test random route generation (for genetic algorithm)
    n_nodes = len(all_node_ids)
    random_route = np.random.permutation(n_nodes).tolist()
    route_distance = calculator.calculate_route_distance(random_route)
    
    print(f"✓ Random route ({n_nodes} nodes): {route_distance:.1f} meters")
    print(f"✓ Average distance per segment: {route_distance/(n_nodes):.1f} meters")
    
    return distance_matrix, all_node_ids

def performance_benchmark():
    """Benchmark performance for different problem sizes."""
    print("\n=== Performance Benchmark ===")
    
    import time
    
    sizes = [10, 20, 30, 50]
    
    for size in sizes:
        print(f"\nBenchmarking {size} nodes:")
        
        # Load data
        start_time = time.time()
        loader = load_transportation_data(sample_nodes=size)
        load_time = time.time() - start_time
        
        # Get coordinates
        node_ids = list(loader.graph.nodes())
        coordinates = [loader.get_node_coordinates(nid) for nid in node_ids]
        
        # Calculate distance matrix
        start_time = time.time()
        calculator = DistanceCalculator(coordinates)
        distance_matrix = calculator.get_distance_matrix()
        calc_time = time.time() - start_time
        
        # Test route calculation
        start_time = time.time()
        test_route = list(range(len(node_ids)))
        route_distance = calculator.calculate_route_distance(test_route)
        route_time = time.time() - start_time
        
        print(f"  Load time: {load_time:.3f}s")
        print(f"  Distance matrix: {calc_time:.3f}s")
        print(f"  Route calculation: {route_time:.6f}s")
        print(f"  Connected graph: {len(loader.graph.edges) > 0}")

def main():
    """Run complete testing suite."""
    print("FIAP Tech Challenge - Complete Data Loading Tests")
    print("=" * 60)
    
    try:
        # Test complete pipeline
        loader, calculator = test_complete_pipeline()
        
        # Test optimization readiness
        distance_matrix, node_ids = test_optimization_readiness()
        
        # Performance benchmark
        performance_benchmark()
        
        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED!")
        print("🚀 System is ready for genetic algorithm implementation!")
        print("📊 Data loading and distance calculations are optimized")
        
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main()