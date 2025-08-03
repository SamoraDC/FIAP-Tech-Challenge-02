"""
Test script for data loading functionality.
Validates that the transportation data can be loaded and processed correctly.
"""

import sys
sys.path.append('src')

from utils.data_loader import TransportationDataLoader, load_transportation_data
from utils.distance_utils import DistanceCalculator, haversine_distance
import time

def test_basic_loading():
    """Test basic data loading functionality."""
    print("=== Testing Basic Data Loading ===")
    
    # Initialize loader
    loader = TransportationDataLoader()
    
    # Test nodes loading
    print("\n1. Loading nodes...")
    nodes_df = loader.load_nodes()
    print(f"✓ Successfully loaded {len(nodes_df)} nodes")
    
    # Test sample edges loading
    print("\n2. Loading sample edges...")
    edges_df = loader.load_edges(sample_size=1000)
    print(f"✓ Successfully loaded {len(edges_df)} edges")
    
    # Test graph creation with sample
    print("\n3. Creating sample graph...")
    graph = loader.create_graph(use_sample=True, sample_nodes=20)
    print(f"✓ Successfully created graph with {len(graph.nodes)} nodes and {len(graph.edges)} edges")
    
    return loader

def test_distance_calculations():
    """Test distance calculation utilities."""
    print("\n=== Testing Distance Calculations ===")
    
    # Test coordinates (Brasília area)
    coord1 = (-47.8825, -15.7942)  # Brasília center
    coord2 = (-47.8519, -15.7801)  # Nearby location
    
    # Test Haversine distance
    distance = haversine_distance(coord1, coord2)
    print(f"✓ Haversine distance between test points: {distance:.1f} meters")
    
    # Test distance calculator
    test_coords = [coord1, coord2, (-47.9000, -15.8000)]
    calculator = DistanceCalculator(test_coords)
    distance_matrix = calculator.get_distance_matrix()
    print(f"✓ Distance matrix shape: {distance_matrix.shape}")
    print(f"✓ Sample distances: {distance_matrix[0]}")
    
    return calculator

def test_route_calculation():
    """Test route distance calculation."""
    print("\n=== Testing Route Calculation ===")
    
    # Load sample data
    loader = load_transportation_data(sample_nodes=10)
    
    # Get coordinates for sample nodes
    sample_nodes = list(loader.graph.nodes())[:5]
    coordinates = []
    for node_id in sample_nodes:
        coord = loader.get_node_coordinates(node_id)
        coordinates.append(coord)
    
    # Create calculator
    calculator = DistanceCalculator(coordinates)
    
    # Test route [0, 1, 2, 3, 4]
    test_route = [0, 1, 2, 3, 4]
    route_distance = calculator.calculate_route_distance(test_route)
    print(f"✓ Route distance for {test_route}: {route_distance:.1f} meters")
    
    return calculator, test_route

def test_statistics():
    """Test dataset statistics generation."""
    print("\n=== Testing Dataset Statistics ===")
    
    loader = load_transportation_data(sample_nodes=50)
    stats = loader.get_dataset_statistics()
    
    print("✓ Dataset Statistics:")
    for category, data in stats.items():
        print(f"  {category.upper()}:")
        for key, value in data.items():
            if isinstance(value, float):
                print(f"    {key}: {value:.3f}")
            else:
                print(f"    {key}: {value}")
    
    return stats

def performance_test():
    """Test performance with different dataset sizes."""
    print("\n=== Performance Testing ===")
    
    sizes_to_test = [10, 25, 50]
    
    for size in sizes_to_test:
        print(f"\nTesting with {size} nodes:")
        
        start_time = time.time()
        loader = load_transportation_data(sample_nodes=size)
        load_time = time.time() - start_time
        
        start_time = time.time()
        stats = loader.get_dataset_statistics()
        stats_time = time.time() - start_time
        
        print(f"  ✓ Load time: {load_time:.3f}s")
        print(f"  ✓ Stats time: {stats_time:.3f}s")
        print(f"  ✓ Graph density: {stats['graph']['density']:.3f}")
        print(f"  ✓ Connected: {stats['graph']['is_connected']}")

def main():
    """Run all tests."""
    print("FIAP Tech Challenge - Data Loading Tests")
    print("=" * 50)
    
    try:
        # Test basic loading
        loader = test_basic_loading()
        
        # Test distance calculations
        calculator = test_distance_calculations()
        
        # Test route calculations
        route_calc, test_route = test_route_calculation()
        
        # Test statistics
        stats = test_statistics()
        
        # Performance testing
        performance_test()
        
        print("\n" + "=" * 50)
        print("✅ ALL TESTS PASSED SUCCESSFULLY!")
        print("✅ Data loading system is ready for algorithm implementation")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    main()