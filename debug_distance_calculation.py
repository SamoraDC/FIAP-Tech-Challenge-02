"""
Debug script to identify distance calculation inconsistencies.
"""

import sys
sys.path.append('src')

from utils.data_loader import load_transportation_data
from utils.distance_utils import DistanceCalculator
from algorithms.genetic_algorithm import GeneticAlgorithm, GeneticConfig
from algorithms.conventional_algorithms import GreedyAlgorithms
import numpy as np

def debug_distance_calculations():
    """Debug distance calculation methods."""
    print("🔍 Debugging Distance Calculations")
    print("=" * 50)
    
    # Load small dataset
    loader = load_transportation_data(sample_nodes=5)
    node_ids = list(loader.graph.nodes())
    coordinates = [loader.get_node_coordinates(nid) for nid in node_ids]
    
    # Create distance calculator
    calculator = DistanceCalculator(coordinates)
    distance_matrix = calculator.get_distance_matrix()
    
    print(f"Cities: {len(coordinates)}")
    print(f"Distance matrix shape: {distance_matrix.shape}")
    print(f"Sample distances:\n{distance_matrix[:3, :3]}")
    
    # Test route
    test_route = [0, 1, 2, 3, 4]
    
    # Method 1: Manual calculation
    manual_distance = 0.0
    for i in range(len(test_route)):
        next_i = (i + 1) % len(test_route)
        manual_distance += distance_matrix[test_route[i]][test_route[next_i]]
    
    print(f"\nTest route: {test_route}")
    print(f"Manual calculation: {manual_distance:.1f}")
    
    # Method 2: Genetic Algorithm calculation
    ga = GeneticAlgorithm(distance_matrix)
    ga_distance = ga.calculate_distance(test_route)
    print(f"GA calculation: {ga_distance:.1f}")
    
    # Method 3: Greedy algorithm calculation
    greedy = GreedyAlgorithms(distance_matrix)
    # Test nearest neighbor
    nn_result = greedy.nearest_neighbor(start=0)
    print(f"NN route: {nn_result.route}")
    print(f"NN reported distance: {nn_result.distance:.1f}")
    
    # Recalculate NN distance manually
    nn_manual = 0.0
    for i in range(len(nn_result.route)):
        next_i = (i + 1) % len(nn_result.route)
        nn_manual += distance_matrix[nn_result.route[i]][nn_result.route[next_i]]
    
    print(f"NN manual recalc: {nn_manual:.1f}")
    
    # Check if they match
    print(f"\nComparisons:")
    print(f"Manual vs GA: {abs(manual_distance - ga_distance) < 0.1}")
    print(f"NN reported vs manual: {abs(nn_result.distance - nn_manual) < 0.1}")
    
    return distance_matrix, test_route

if __name__ == "__main__":
    debug_distance_calculations()