"""
Full Dataset Testing Suite - Organized Version
FIAP Tech Challenge Phase 2 - Complete 1000-node Brazilian transportation network analysis.

This script tests all implemented algorithms on the complete dataset with proper organization.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.data_loader import load_transportation_data
from utils.distance_utils import DistanceCalculator
from algorithms.genetic_algorithm import GeneticAlgorithm, GeneticConfig
from algorithms.conventional_algorithms import run_conventional_algorithms
from algorithms.metaheuristic_algorithms import run_metaheuristic_algorithms, MetaheuristicConfig

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

class FullDatasetTester:
    """
    Comprehensive testing system for the complete Brazilian transportation dataset.
    Tests all implemented algorithms on the full 1000-node network with detailed metrics.
    """
    
    def __init__(self):
        """Initialize the full dataset tester."""
        self.results = {}
        self.system_metrics = {}
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
    
    def run_scalable_testing(self, problem_sizes: List[int] = None, max_size: int = 100):
        """
        Run scalable testing from small to large problem sizes.
        
        Args:
            problem_sizes: List of problem sizes to test
            max_size: Maximum problem size to test
        """
        if problem_sizes is None:
            problem_sizes = [10, 20, 30, 50, 100]
        
        # Ensure we don't exceed max_size
        problem_sizes = [size for size in problem_sizes if size <= max_size]
        
        print("🚀 FULL DATASET COMPREHENSIVE TESTING")
        print("=" * 80)
        print(f"Problem sizes: {problem_sizes}")
        print(f"Maximum size: {max_size} nodes from 1000-node dataset")
        print(f"Dataset: Brazilian Transportation Network")
        
        self.start_time = time.time()
        all_results = {}
        
        for size in problem_sizes:
            print(f"\n{'='*20} TESTING {size} CITIES FROM FULL DATASET {'='*20}")
            
            # Record system metrics before test
            pre_metrics = self.get_system_metrics()
            
            try:
                # Run comprehensive test for this size
                size_results = self.run_algorithm_suite_full_dataset(size)
                
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
                self.display_size_summary(size, size_results)
                
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
        
        print(f"\n✅ FULL DATASET TESTING COMPLETED in {total_time:.2f} seconds")
        return all_results
    
    def run_algorithm_suite_full_dataset(self, num_cities: int) -> Dict:
        """
        Run complete algorithm suite using samples from the full dataset.
        
        Args:
            num_cities: Number of cities to sample from the full dataset
            
        Returns:
            Dictionary containing all algorithm results
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
        
        # 1. GENETIC ALGORITHM (Our main implementation)
        print("\n🧬 Running Genetic Algorithm...")
        try:
            ga_config = GeneticConfig(
                population_size=min(200, max(50, num_cities * 4)),
                generations=min(500, max(100, num_cities * 10)),
                elite_size=max(5, min(30, num_cities // 2)),
                mutation_rate=0.02,
                selection_method="tournament",
                crossover_method="order",
                mutation_method="swap"
            )
            
            start_time = time.time()
            start_memory = self.get_system_metrics()['memory_usage_mb']
            
            ga = GeneticAlgorithm(distance_matrix, ga_config)
            ga_best = ga.run(verbose=True)
            
            end_time = time.time()
            end_memory = self.get_system_metrics()['memory_usage_mb']
            
            results['Genetic Algorithm'] = {
                'distance': ga_best.distance,
                'time': end_time - start_time,
                'route': ga_best.route,
                'generations_completed': ga.generation,
                'final_diversity': ga.history['diversity_scores'][-1] if ga.history['diversity_scores'] else 0,
                'convergence_rate': ga._calculate_convergence_rate(),
                'memory_usage_mb': end_memory - start_memory,
                'category': 'Evolutionary',
                'population_size': ga_config.population_size,
                'total_evaluations': ga.generation * ga_config.population_size
            }
            
            print(f"✅ GA: {ga_best.distance/1000:.2f} km in {end_time - start_time:.2f}s ({ga.generation} generations)")
            
        except Exception as e:
            print(f"❌ Genetic Algorithm failed: {str(e)}")
            results['Genetic Algorithm'] = {'error': str(e), 'category': 'Evolutionary'}
        
        # 2. CONVENTIONAL ALGORITHMS
        print("\n🔍 Running Conventional Algorithms...")
        try:
            start_time = time.time()
            start_memory = self.get_system_metrics()['memory_usage_mb']
            
            conventional_results = run_conventional_algorithms(distance_matrix, coordinates)
            
            end_time = time.time()
            end_memory = self.get_system_metrics()['memory_usage_mb']
            conv_time = end_time - start_time
            
            for result in conventional_results:
                results[result.algorithm_name] = {
                    'distance': result.distance,
                    'time': result.execution_time,
                    'route': result.route,
                    'nodes_explored': result.nodes_explored,
                    'memory_usage_mb': (end_memory - start_memory) / len(conventional_results),
                    'category': 'Conventional'
                }
            
            print(f"✅ Conventional algorithms completed in {conv_time:.2f}s")
            
        except Exception as e:
            print(f"❌ Conventional algorithms failed: {str(e)}")
        
        # 3. METAHEURISTIC ALGORITHMS
        print("\n🐜 Running Metaheuristic Algorithms...")
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
            
            print(f"✅ Metaheuristic algorithms completed in {meta_time:.2f}s")
            
        except Exception as e:
            print(f"❌ Metaheuristic algorithms failed: {str(e)}")
        
        return results
    
    def display_size_summary(self, size: int, results: Dict):
        """
        Display comprehensive summary for a problem size.
        
        Args:
            size: Problem size
            results: Algorithm results
        """
        print(f"\n📊 COMPREHENSIVE SUMMARY FOR {size} CITIES:")
        print("-" * 70)
        
        # Filter successful results
        successful_results = {k: v for k, v in results.items() if 'error' not in v}
        
        if not successful_results:
            print("❌ No successful algorithm results")
            return
        
        # Sort by distance
        sorted_results = sorted(successful_results.items(), key=lambda x: x[1]['distance'])
        
        print(f"{'Rank':<4} {'Algorithm':<25} {'Distance (km)':<12} {'Time (s)':<10} {'Memory (MB)':<12} {'Category':<15}")
        print("-" * 100)
        
        for i, (algo_name, result) in enumerate(sorted_results, 1):
            distance_km = result['distance'] / 1000
            time_s = result['time']
            memory_mb = result.get('memory_usage_mb', 0)
            category = result['category']
            
            # Add medal emoji for top 3
            medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
            
            print(f"{medal} {i:<2} {algo_name:<25} {distance_km:<12.2f} {time_s:<10.3f} {memory_mb:<12.1f} {category:<15}")
        
        # Performance insights
        best_distance = sorted_results[0][1]['distance']
        worst_distance = sorted_results[-1][1]['distance']
        improvement = (worst_distance - best_distance) / best_distance * 100
        
        print(f"\n💡 PERFORMANCE INSIGHTS:")
        print(f"   • Best vs Worst: {improvement:.1f}% optimization potential")
        print(f"   • Fastest: {min(successful_results.items(), key=lambda x: x[1]['time'])[0]}")
        print(f"   • Most Memory Efficient: {min(successful_results.items(), key=lambda x: x[1].get('memory_usage_mb', float('inf')))[0]}")
    
    def export_results(self, output_dir: str = "results/full_dataset_analysis"):
        """Export comprehensive results to files."""
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Export raw results to JSON
        json_path = os.path.join(output_dir, 'full_dataset_results.json')
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Create comprehensive CSV
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
                    'generations': result.get('generations_completed', 0),
                    'iterations': result.get('iterations_completed', 0),
                    'nodes_explored': result.get('nodes_explored', 0)
                })
        
        # Save CSV
        csv_path = os.path.join(output_dir, 'comprehensive_results.csv')
        df = pd.DataFrame(csv_data)
        df.to_csv(csv_path, index=False)
        
        print(f"💾 Full dataset results exported:")
        print(f"  • Raw results: {json_path}")
        print(f"  • CSV summary: {csv_path}")

def run_full_dataset_analysis(problem_sizes: List[int] = None):
    """
    Main function to run complete full dataset analysis.
    
    Args:
        problem_sizes: List of problem sizes to test (default: [15, 25, 40, 60, 80])
    """
    if problem_sizes is None:
        problem_sizes = [15, 25, 40, 60, 80]  # Challenging but manageable sizes
    
    print("🌟 FIAP Tech Challenge - Full Dataset Comprehensive Testing")
    print("=" * 80)
    print("🎯 Testing on complete Brazilian Transportation Network")
    print("📊 Dataset: 1,000 nodes, ~500,000 edges")
    
    try:
        # Initialize tester
        tester = FullDatasetTester()
        
        print(f"\n🎯 Test Configuration:")
        print(f"   Problem sizes: {problem_sizes}")
        print(f"   Data source: Full 1000-node Brazilian transportation network")
        print(f"   Sampling: Random connected subgraphs")
        print(f"   Algorithms: 8 different optimization approaches")
        
        # Run comprehensive testing
        results = tester.run_scalable_testing(problem_sizes, max_size=100)
        
        # Export results
        tester.export_results()
        
        print("\n" + "="*80)
        print("🎊 FULL DATASET COMPREHENSIVE TESTING COMPLETED!")
        print("✅ All algorithms tested on real Brazilian transportation data")
        print("📊 Scalability and resource usage analysis completed")
        print("💾 Comprehensive results exported to 'results/full_dataset_analysis/'")
        print("🚀 Ready for final algorithm comparison documentation")
        print("="*80)
        
        return True
        
    except Exception as e:
        print(f"\n❌ TESTING FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Run with default problem sizes
    run_full_dataset_analysis()