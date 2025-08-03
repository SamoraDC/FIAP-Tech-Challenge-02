"""
Focused Four Algorithms Testing Suite
FIAP Tech Challenge Phase 2 - Testing ONLY the 4 requested algorithms on full dataset.

ALGORITHMS TESTED:
1. Particle Swarm Optimization
2. Ant Colony Optimization  
3. Dijkstra
4. A*
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.data_loader import load_transportation_data
from utils.distance_utils import DistanceCalculator
from algorithms.metaheuristic_algorithms import run_metaheuristic_algorithms, MetaheuristicConfig
from algorithms.conventional_algorithms import run_conventional_algorithms

import time
import statistics
import numpy as np
import pandas as pd
import json
import os
import psutil
import gc
from typing import Dict, List, Tuple
from datetime import datetime

class FocusedFourAlgorithmsTester:
    """
    Testing system for ONLY the 4 requested algorithms:
    - Particle Swarm Optimization
    - Ant Colony Optimization  
    - Dijkstra
    - A*
    """
    
    def __init__(self):
        """Initialize the focused tester."""
        self.results = {}
        self.start_time = None
        
    def get_system_metrics(self) -> Dict:
        """Get current system resource usage."""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            'memory_usage_mb': memory_info.rss / 1024 / 1024,
            'memory_percent': process.memory_percent(),
            'cpu_percent': process.cpu_percent(),
            'timestamp': datetime.now().isoformat()
        }
    
    def run_focused_testing(self, problem_sizes: List[int] = None, max_size: int = 100):
        """
        Run testing for ONLY the 4 requested algorithms.
        
        Args:
            problem_sizes: List of problem sizes to test
            max_size: Maximum problem size to test
        """
        if problem_sizes is None:
            problem_sizes = [15, 25, 40, 60, 80]
        
        # Ensure we don't exceed max_size
        problem_sizes = [size for size in problem_sizes if size <= max_size]
        
        print("🎯 FOCUSED FOUR ALGORITHMS TESTING")
        print("=" * 80)
        print("TESTING ONLY THE 4 REQUESTED ALGORITHMS:")
        print("1. Particle Swarm Optimization")
        print("2. Ant Colony Optimization")
        print("3. Dijkstra")
        print("4. A*")
        print("=" * 80)
        print(f"Problem sizes: {problem_sizes}")
        print(f"Maximum size: {max_size} nodes from 1000-node dataset")
        print(f"Dataset: Brazilian Transportation Network")
        
        self.start_time = time.time()
        all_results = {}
        
        for size in problem_sizes:
            print(f"\n{'='*20} TESTING {size} CITIES - 4 ALGORITHMS ONLY {'='*20}")
            
            # Record system metrics before test
            pre_metrics = self.get_system_metrics()
            
            try:
                # Run the 4 requested algorithms only
                size_results = self.run_four_algorithms_only(size)
                
                # Record system metrics after test
                post_metrics = self.get_system_metrics()
                
                all_results[f'{size}_cities'] = {
                    'algorithm_results': size_results,
                    'problem_size': size,
                    'system_metrics': {
                        'pre_test': pre_metrics,
                        'post_test': post_metrics,
                        'memory_delta_mb': post_metrics['memory_usage_mb'] - pre_metrics['memory_usage_mb']
                    }
                }
                
                # Display summary
                self.display_focused_summary(size, size_results)
                
                # Force garbage collection
                gc.collect()
                
            except Exception as e:
                print(f"❌ Error testing {size} cities: {str(e)}")
                all_results[f'{size}_cities'] = {
                    'error': str(e),
                    'problem_size': size
                }
        
        self.results = all_results
        total_time = time.time() - self.start_time
        
        print(f"\n✅ FOCUSED 4-ALGORITHM TESTING COMPLETED in {total_time:.2f} seconds")
        return all_results
    
    def run_four_algorithms_only(self, num_cities: int) -> Dict:
        """
        Run ONLY the 4 requested algorithms.
        
        Args:
            num_cities: Number of cities to sample from the full dataset
            
        Returns:
            Dictionary containing the 4 algorithm results
        """
        print(f"📊 Loading {num_cities} cities from full 1000-node dataset...")
        
        # Load full dataset and sample
        loader = load_transportation_data(sample_nodes=num_cities)
        node_ids = list(loader.graph.nodes())
        coordinates = [loader.get_node_coordinates(nid) for nid in node_ids]
        
        print(f"✅ Loaded {len(coordinates)} cities from Brazilian network")
        print(f"   Graph connectivity: {loader.graph.number_of_edges()} edges")
        
        # Create distance calculator
        calculator = DistanceCalculator(coordinates)
        distance_matrix = calculator.get_distance_matrix()
        
        results = {}
        
        # 1. METAHEURISTIC ALGORITHMS (PSO and ACO)
        print("\n🐜 Running Metaheuristic Algorithms (PSO & ACO)...")
        try:
            meta_config = MetaheuristicConfig(
                population_size=min(100, max(30, num_cities * 2)),
                max_iterations=min(300, max(50, num_cities * 6)),
                num_ants=min(100, max(30, num_cities * 2))
            )
            
            start_time = time.time()
            start_memory = self.get_system_metrics()['memory_usage_mb']
            
            metaheuristic_results = run_metaheuristic_algorithms(distance_matrix, meta_config)
            
            end_time = time.time()
            end_memory = self.get_system_metrics()['memory_usage_mb']
            meta_time = end_time - start_time
            
            # Process PSO results
            pso_result = metaheuristic_results['PSO']
            results['Particle Swarm Optimization'] = {
                'distance': pso_result['distance'],
                'time': meta_time / 2,
                'route': pso_result['route'],
                'iterations_completed': len(pso_result['history']['best_fitness']),
                'final_diversity': pso_result['history']['diversity'][-1] if pso_result['history']['diversity'] else 0,
                'memory_usage_mb': (end_memory - start_memory) / 2,
                'category': 'Metaheuristic',
                'swarm_size': meta_config.population_size
            }
            
            # Process ACO results
            aco_result = metaheuristic_results['ACO']
            results['Ant Colony Optimization'] = {
                'distance': aco_result['distance'],
                'time': meta_time / 2,
                'route': aco_result['route'],
                'iterations_completed': len(aco_result['history']['best_distances']),
                'memory_usage_mb': (end_memory - start_memory) / 2,
                'category': 'Metaheuristic',
                'num_ants': meta_config.num_ants
            }
            
            print(f"✅ PSO: {pso_result['distance']/1000:.2f} km")
            print(f"✅ ACO: {aco_result['distance']/1000:.2f} km")
            
        except Exception as e:
            print(f"❌ Metaheuristic algorithms failed: {str(e)}")
        
        # 2. CONVENTIONAL ALGORITHMS (Dijkstra and A*)
        print("\n🔍 Running Conventional Algorithms (Dijkstra & A*)...")
        try:
            start_time = time.time()
            start_memory = self.get_system_metrics()['memory_usage_mb']
            
            conventional_results = run_conventional_algorithms(distance_matrix, coordinates)
            
            end_time = time.time()
            end_memory = self.get_system_metrics()['memory_usage_mb']
            conv_time = end_time - start_time
            
            # Filter to get only Dijkstra and A* results
            for result in conventional_results:
                if 'Dijkstra' in result.algorithm_name or 'A*' in result.algorithm_name:
                    results[result.algorithm_name] = {
                        'distance': result.distance,
                        'time': result.execution_time,
                        'route': result.route,
                        'nodes_explored': result.nodes_explored,
                        'memory_usage_mb': (end_memory - start_memory) / 2,  # Split between Dijkstra and A*
                        'category': 'Conventional'
                    }
                    print(f"✅ {result.algorithm_name}: {result.distance/1000:.2f} km")
            
        except Exception as e:
            print(f"❌ Conventional algorithms failed: {str(e)}")
        
        return results
    
    def display_focused_summary(self, size: int, results: Dict):
        """
        Display summary for ONLY the 4 requested algorithms.
        
        Args:
            size: Problem size
            results: Algorithm results
        """
        print(f"\n📊 4-ALGORITHM FOCUSED SUMMARY FOR {size} CITIES:")
        print("-" * 70)
        
        # Filter successful results
        successful_results = {k: v for k, v in results.items() if 'error' not in v}
        
        if not successful_results:
            print("❌ No successful algorithm results")
            return
        
        # Sort by distance
        sorted_results = sorted(successful_results.items(), key=lambda x: x[1]['distance'])
        
        print(f"{'Rank':<4} {'Algorithm':<30} {'Distance (km)':<12} {'Time (s)':<10} {'Category':<15}")
        print("-" * 80)
        
        for i, (algo_name, result) in enumerate(sorted_results, 1):
            distance_km = result['distance'] / 1000
            time_s = result['time']
            category = result['category']
            
            # Add medal emoji for top performers
            medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "4️⃣" if i == 4 else "  "
            
            print(f"{medal} {i:<2} {algo_name:<30} {distance_km:<12.2f} {time_s:<10.3f} {category:<15}")
        
        # Performance insights for the 4 algorithms
        best_distance = sorted_results[0][1]['distance']
        worst_distance = sorted_results[-1][1]['distance']
        improvement = (worst_distance - best_distance) / best_distance * 100
        
        print(f"\n💡 4-ALGORITHM PERFORMANCE INSIGHTS:")
        print(f"   • Best vs Worst (among 4): {improvement:.1f}% optimization potential")
        print(f"   • Best Algorithm: {sorted_results[0][0]}")
        print(f"   • Fastest: {min(successful_results.items(), key=lambda x: x[1]['time'])[0]}")
    
    def export_focused_results(self, output_dir: str = "results/focused_four_algorithms"):
        """Export results for the 4 algorithms to files."""
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Export raw results to JSON
        json_path = os.path.join(output_dir, 'four_algorithms_results.json')
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Create focused CSV
        csv_data = []
        
        for size_key, data in self.results.items():
            if 'algorithm_results' not in data:
                continue
                
            size = data['problem_size']
            
            for algo_name, result in data['algorithm_results'].items():
                if 'error' in result:
                    continue
                
                csv_data.append({
                    'problem_size': size,
                    'algorithm': algo_name,
                    'category': result['category'],
                    'distance_km': result['distance'] / 1000,
                    'time_seconds': result['time'],
                    'memory_mb': result.get('memory_usage_mb', 0),
                    'route_length': len(result.get('route', [])),
                    'iterations': result.get('iterations_completed', 0),
                    'nodes_explored': result.get('nodes_explored', 0),
                    'swarm_size': result.get('swarm_size', 'N/A'),
                    'num_ants': result.get('num_ants', 'N/A')
                })
        
        # Save CSV
        csv_path = os.path.join(output_dir, 'four_algorithms_comparison.csv')
        df = pd.DataFrame(csv_data)
        df.to_csv(csv_path, index=False)
        
        print(f"💾 Focused 4-algorithm results exported:")
        print(f"  • Raw results: {json_path}")
        print(f"  • CSV summary: {csv_path}")

def run_focused_four_algorithms_analysis(problem_sizes: List[int] = None):
    """
    Main function to run the focused 4-algorithm analysis.
    
    Args:
        problem_sizes: List of problem sizes to test (default: [15, 25, 40, 60, 80])
    """
    if problem_sizes is None:
        problem_sizes = [15, 25, 40, 60, 80]
    
    print("🌟 FIAP Tech Challenge - Focused 4-Algorithm Testing")
    print("=" * 80)
    print("🎯 Testing ONLY the 4 requested algorithms:")
    print("   1. Particle Swarm Optimization")
    print("   2. Ant Colony Optimization")
    print("   3. Dijkstra")
    print("   4. A*")
    print("📊 Dataset: 1,000 nodes, ~500,000 edges")
    
    try:
        # Initialize focused tester
        tester = FocusedFourAlgorithmsTester()
        
        print(f"\n🎯 Test Configuration:")
        print(f"   Problem sizes: {problem_sizes}")
        print(f"   Data source: Full 1000-node Brazilian transportation network")
        print(f"   Algorithms: ONLY the 4 specifically requested")
        
        # Run focused testing
        results = tester.run_focused_testing(problem_sizes, max_size=100)
        
        # Export results
        tester.export_focused_results()
        
        print("\n" + "="*80)
        print("🎊 FOCUSED 4-ALGORITHM TESTING COMPLETED!")
        print("✅ Only the 4 requested algorithms tested")
        print("📊 Particle Swarm Optimization, Ant Colony Optimization, Dijkstra, A*")
        print("💾 Results exported to 'results/focused_four_algorithms/'")
        print("🚀 Ready for specific 4-algorithm documentation")
        print("="*80)
        
        return True
        
    except Exception as e:
        print(f"\n❌ TESTING FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Run focused testing for the 4 requested algorithms only
    run_focused_four_algorithms_analysis()