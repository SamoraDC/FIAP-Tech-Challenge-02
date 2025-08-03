"""
Complete Four Algorithms Dataset Testing
FIAP Tech Challenge Phase 2 - Comprehensive testing of ONLY the 4 requested algorithms on full dataset.

ALGORITHMS TESTED:
1. Particle Swarm Optimization  
2. Ant Colony Optimization
3. Dijkstra-Enhanced Nearest Neighbor
4. A* Enhanced Nearest Neighbor
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.data_loader import load_transportation_data
from utils.distance_utils import DistanceCalculator
from algorithms.four_focused_algorithms import run_four_focused_algorithms, FocusedConfig

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

class CompleteFourAlgorithmsDatasetTester:
    """
    Complete dataset testing system for ONLY the 4 requested algorithms.
    Tests on full 1000-node Brazilian transportation network.
    """
    
    def __init__(self):
        """Initialize the complete dataset tester."""
        self.results = {}
        self.start_time = None
        self.full_dataset_stats = {}
        
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
    
    def run_complete_dataset_testing(self, problem_sizes: List[int] = None, 
                                    max_iterations_scale: float = 1.0):
        """
        Run comprehensive testing on the complete dataset with varying problem sizes.
        
        Args:
            problem_sizes: List of problem sizes to test from full dataset
            max_iterations_scale: Scale factor for algorithm iterations
        """
        if problem_sizes is None:
            problem_sizes = [20, 30, 50, 80, 120, 150, 200]  # Larger sizes from full dataset
        
        print("🌟 COMPLETE FOUR ALGORITHMS DATASET TESTING")
        print("=" * 80)
        print("TESTING ONLY THE 4 REQUESTED ALGORITHMS ON FULL DATASET:")
        print("1. Particle Swarm Optimization")
        print("2. Ant Colony Optimization")
        print("3. Dijkstra-Enhanced Nearest Neighbor")
        print("4. A* Enhanced Nearest Neighbor")
        print("=" * 80)
        print(f"Problem sizes: {problem_sizes}")
        print(f"Full dataset: 1,000 nodes, ~500,000 edges")
        print(f"Source: Brazilian Transportation Network")
        print(f"Iterations scale: {max_iterations_scale}x")
        
        self.start_time = time.time()
        all_results = {}
        
        # Load full dataset statistics first
        print("\n📊 Loading full dataset statistics...")
        self._load_full_dataset_stats()
        
        for size in problem_sizes:
            print(f"\n{'='*25} TESTING {size} CITIES FROM FULL DATASET {'='*25}")
            
            # Record system metrics before test
            pre_metrics = self.get_system_metrics()
            
            try:
                # Run the 4 algorithms on this problem size
                size_results = self.run_four_algorithms_complete_dataset(size, max_iterations_scale)
                
                # Record system metrics after test
                post_metrics = self.get_system_metrics()
                
                all_results[f'{size}_cities'] = {
                    'algorithm_results': size_results,
                    'problem_size': size,
                    'dataset_coverage': (size / 1000) * 100,  # Percentage of full dataset
                    'system_metrics': {
                        'pre_test': pre_metrics,
                        'post_test': post_metrics,
                        'memory_delta_mb': post_metrics['memory_usage_mb'] - pre_metrics['memory_usage_mb']
                    }
                }
                
                # Display summary
                self.display_complete_summary(size, size_results)
                
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
        
        print(f"\n✅ COMPLETE 4-ALGORITHM DATASET TESTING COMPLETED in {total_time/60:.2f} minutes")
        return all_results
    
    def _load_full_dataset_stats(self):
        """Load and cache full dataset statistics."""
        try:
            loader = load_transportation_data()  # Load full dataset
            
            self.full_dataset_stats = {
                'total_nodes': loader.nodes_df.shape[0],
                'total_edges': loader.edges_df.shape[0] if hasattr(loader, 'edges_df') else 0,
                'geographic_bounds': {
                    'min_longitude': float(loader.nodes_df['longitude'].min()),
                    'max_longitude': float(loader.nodes_df['longitude'].max()),
                    'min_latitude': float(loader.nodes_df['latitude'].min()),
                    'max_latitude': float(loader.nodes_df['latitude'].max())
                },
                'coordinate_range': {
                    'longitude_span': float(loader.nodes_df['longitude'].max() - loader.nodes_df['longitude'].min()),
                    'latitude_span': float(loader.nodes_df['latitude'].max() - loader.nodes_df['latitude'].min())
                }
            }
            
            print(f"✅ Full dataset loaded: {self.full_dataset_stats['total_nodes']} nodes")
            print(f"   Geographic bounds: Longitude [{self.full_dataset_stats['geographic_bounds']['min_longitude']:.3f}, {self.full_dataset_stats['geographic_bounds']['max_longitude']:.3f}]")
            print(f"                     Latitude  [{self.full_dataset_stats['geographic_bounds']['min_latitude']:.3f}, {self.full_dataset_stats['geographic_bounds']['max_latitude']:.3f}]")
            
        except Exception as e:
            print(f"⚠️ Could not load full dataset stats: {str(e)}")
            self.full_dataset_stats = {'total_nodes': 1000, 'total_edges': 500000}
    
    def run_four_algorithms_complete_dataset(self, num_cities: int, 
                                           iterations_scale: float = 1.0) -> Dict:
        """
        Run ONLY the 4 requested algorithms on a sample from the complete dataset.
        
        Args:
            num_cities: Number of cities to sample from the full 1000-node dataset
            iterations_scale: Scale factor for algorithm iterations
            
        Returns:
            Dictionary containing the 4 algorithm results
        """
        print(f"📊 Sampling {num_cities} cities from full {self.full_dataset_stats.get('total_nodes', 1000)}-node dataset...")
        
        # Load full dataset and sample strategically
        loader = load_transportation_data(sample_nodes=num_cities)
        node_ids = list(loader.graph.nodes())
        coordinates = [loader.get_node_coordinates(nid) for nid in node_ids]
        
        actual_nodes = len(coordinates)
        print(f"✅ Sampled {actual_nodes} cities from Brazilian transportation network")
        print(f"   Dataset coverage: {(actual_nodes / self.full_dataset_stats.get('total_nodes', 1000)) * 100:.1f}%")
        print(f"   Graph connectivity: {loader.graph.number_of_edges()} edges")
        
        # Create distance calculator
        calculator = DistanceCalculator(coordinates)
        distance_matrix = calculator.get_distance_matrix()
        
        print(f"   Distance matrix: {distance_matrix.shape[0]}×{distance_matrix.shape[1]}")
        
        # Configure algorithms based on problem size and scale
        config = FocusedConfig(
            # PSO Configuration - scaled for larger problems
            pso_population_size=min(200, max(50, int(num_cities * 2 * iterations_scale))),
            pso_max_iterations=min(500, max(100, int(num_cities * 3 * iterations_scale))),
            pso_w=0.9,
            pso_c1=2.0,
            pso_c2=2.0,
            
            # ACO Configuration - scaled for larger problems  
            aco_num_ants=min(200, max(50, int(num_cities * 2 * iterations_scale))),
            aco_max_iterations=min(500, max(100, int(num_cities * 3 * iterations_scale))),
            aco_alpha=1.0,
            aco_beta=2.0,
            aco_rho=0.1,
            aco_q=100.0
        )
        
        print(f"🔧 Algorithm Configuration:")
        print(f"   PSO: {config.pso_population_size} particles, {config.pso_max_iterations} iterations")
        print(f"   ACO: {config.aco_num_ants} ants, {config.aco_max_iterations} iterations")
        
        # Run all 4 focused algorithms
        start_time = time.time()
        results = run_four_focused_algorithms(distance_matrix, coordinates, config)
        total_time = time.time() - start_time
        
        # Add execution metadata
        for algo_name, result in results.items():
            result.additional_info = result.additional_info or {}
            result.additional_info.update({
                'dataset_coverage_percent': (actual_nodes / self.full_dataset_stats.get('total_nodes', 1000)) * 100,
                'sampled_nodes': actual_nodes,
                'total_execution_time': total_time,
                'iterations_scale_used': iterations_scale,
                'problem_complexity': self._calculate_problem_complexity(num_cities)
            })
        
        print(f"\n⏱️ Total execution time for {num_cities} cities: {total_time:.2f} seconds")
        
        return results
    
    def _calculate_problem_complexity(self, num_cities: int) -> str:
        """Calculate problem complexity category."""
        if num_cities <= 30:
            return "Small"
        elif num_cities <= 80:
            return "Medium"
        elif num_cities <= 150:
            return "Large"
        else:
            return "Very Large"
    
    def display_complete_summary(self, size: int, results: Dict):
        """
        Display comprehensive summary for the 4 algorithms.
        
        Args:
            size: Problem size
            results: Algorithm results dict
        """
        print(f"\n📊 COMPLETE DATASET SUMMARY FOR {size} CITIES:")
        print("-" * 80)
        
        # Filter successful results
        successful_results = {k: v for k, v in results.items() if hasattr(v, 'distance')}
        
        if not successful_results:
            print("❌ No successful algorithm results")
            return
        
        # Sort by distance
        sorted_results = sorted(successful_results.items(), key=lambda x: x[1].distance)
        
        print(f"{'Rank':<4} {'Algorithm':<35} {'Distance (km)':<12} {'Time (s)':<10} {'Iterations':<12}")
        print("-" * 90)
        
        for i, (algo_name, result) in enumerate(sorted_results, 1):
            distance_km = result.distance / 1000
            time_s = result.execution_time
            iterations = result.iterations_completed
            
            # Add medal emoji for top performers
            medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "4️⃣" if i == 4 else "  "
            
            print(f"{medal} {i:<2} {algo_name:<35} {distance_km:<12.2f} {time_s:<10.2f} {iterations:<12}")
        
        # Performance insights
        best_distance = sorted_results[0][1].distance
        worst_distance = sorted_results[-1][1].distance
        improvement = (worst_distance - best_distance) / best_distance * 100
        
        total_execution = sum(result.execution_time for _, result in successful_results.items())
        
        print(f"\n💡 COMPLETE DATASET PERFORMANCE INSIGHTS:")
        print(f"   • Problem size: {size} cities ({(size/1000)*100:.1f}% of full dataset)")
        print(f"   • Best vs Worst optimization: {improvement:.1f}% improvement potential")
        print(f"   • Champion algorithm: {sorted_results[0][0]}")
        print(f"   • Fastest algorithm: {min(successful_results.items(), key=lambda x: x[1].execution_time)[0]}")
        print(f"   • Total computation time: {total_execution:.2f} seconds")
        print(f"   • Complexity category: {self._calculate_problem_complexity(size)}")
    
    def export_complete_results(self, output_dir: str = "results/complete_four_algorithms_dataset"):
        """Export comprehensive results to files."""
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Export raw results to JSON
        json_path = os.path.join(output_dir, 'complete_dataset_results.json')
        
        # Convert results to serializable format
        serializable_results = {}
        for size_key, data in self.results.items():
            if 'algorithm_results' in data:
                serializable_data = data.copy()
                serializable_data['algorithm_results'] = {}
                
                for algo_name, result in data['algorithm_results'].items():
                    serializable_data['algorithm_results'][algo_name] = {
                        'algorithm_name': result.algorithm_name,
                        'route': result.route,
                        'distance': result.distance,
                        'execution_time': result.execution_time,
                        'nodes_explored': result.nodes_explored,
                        'iterations_completed': result.iterations_completed,
                        'additional_info': result.additional_info
                    }
                
                serializable_results[size_key] = serializable_data
            else:
                serializable_results[size_key] = data
        
        with open(json_path, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        # Create comprehensive CSV
        csv_data = []
        
        for size_key, data in self.results.items():
            if 'algorithm_results' not in data:
                continue
                
            size = data['problem_size']
            coverage = data.get('dataset_coverage', 0)
            
            for algo_name, result in data['algorithm_results'].items():
                csv_data.append({
                    'problem_size': size,
                    'dataset_coverage_percent': coverage,
                    'algorithm': result.algorithm_name,
                    'distance_km': result.distance / 1000,
                    'execution_time_seconds': result.execution_time,
                    'nodes_explored': result.nodes_explored,
                    'iterations_completed': result.iterations_completed,
                    'route_length': len(result.route),
                    'complexity_category': self._calculate_problem_complexity(size),
                    'memory_delta_mb': data['system_metrics']['memory_delta_mb'],
                    'scale_factor': result.additional_info.get('iterations_scale_used', 1.0) if result.additional_info else 1.0
                })
        
        # Save comprehensive CSV
        csv_path = os.path.join(output_dir, 'complete_dataset_analysis.csv')
        df = pd.DataFrame(csv_data)
        df.to_csv(csv_path, index=False)
        
        # Create summary statistics
        summary_stats = self._generate_summary_statistics(df)
        stats_path = os.path.join(output_dir, 'summary_statistics.json')
        with open(stats_path, 'w') as f:
            json.dump(summary_stats, f, indent=2)
        
        print(f"💾 Complete dataset results exported:")
        print(f"  • Raw results: {json_path}")
        print(f"  • Analysis CSV: {csv_path}")
        print(f"  • Summary statistics: {stats_path}")
    
    def _generate_summary_statistics(self, df: pd.DataFrame) -> Dict:
        """Generate comprehensive summary statistics."""
        stats = {
            'overall_performance': {},
            'algorithm_rankings': {},
            'scalability_analysis': {},
            'execution_time_analysis': {}
        }
        
        # Overall performance by algorithm
        for algo in df['algorithm'].unique():
            algo_data = df[df['algorithm'] == algo]
            stats['overall_performance'][algo] = {
                'avg_distance_km': float(algo_data['distance_km'].mean()),
                'std_distance_km': float(algo_data['distance_km'].std()),
                'min_distance_km': float(algo_data['distance_km'].min()),
                'max_distance_km': float(algo_data['distance_km'].max()),
                'avg_execution_time': float(algo_data['execution_time_seconds'].mean()),
                'total_tests': int(len(algo_data))
            }
        
        # Algorithm rankings by problem size
        for size in sorted(df['problem_size'].unique()):
            size_data = df[df['problem_size'] == size]
            ranked = size_data.sort_values('distance_km')
            stats['algorithm_rankings'][f'{size}_cities'] = ranked['algorithm'].tolist()
        
        # Scalability analysis
        for algo in df['algorithm'].unique():
            algo_data = df[df['algorithm'] == algo].sort_values('problem_size')
            if len(algo_data) > 1:
                time_growth = []
                for i in range(1, len(algo_data)):
                    prev_time = algo_data.iloc[i-1]['execution_time_seconds']
                    curr_time = algo_data.iloc[i]['execution_time_seconds']
                    prev_size = algo_data.iloc[i-1]['problem_size']
                    curr_size = algo_data.iloc[i]['problem_size']
                    
                    if prev_time > 0 and prev_size > 0:
                        size_ratio = curr_size / prev_size
                        time_ratio = curr_time / prev_time
                        growth_rate = time_ratio / size_ratio
                        time_growth.append(growth_rate)
                
                if time_growth:
                    stats['scalability_analysis'][algo] = {
                        'avg_growth_rate': float(np.mean(time_growth)),
                        'scalability_category': 'Excellent' if np.mean(time_growth) < 1.5 else 'Good' if np.mean(time_growth) < 2.5 else 'Moderate'
                    }
        
        return stats

def run_complete_four_algorithms_dataset_analysis(problem_sizes: List[int] = None, 
                                                 iterations_scale: float = 1.0):
    """
    Main function to run complete dataset analysis for the 4 algorithms.
    
    Args:
        problem_sizes: List of problem sizes to test (default: varied from small to large)
        iterations_scale: Scale factor for algorithm iterations
    """
    if problem_sizes is None:
        problem_sizes = [20, 30, 50, 80, 120, 150, 200]  # Comprehensive range
    
    print("🌟 FIAP Tech Challenge - Complete Four Algorithms Dataset Analysis")
    print("=" * 80)
    print("🎯 Testing ONLY the 4 requested algorithms on FULL DATASET:")
    print("   1. Particle Swarm Optimization")
    print("   2. Ant Colony Optimization")
    print("   3. Dijkstra-Enhanced Nearest Neighbor")
    print("   4. A* Enhanced Nearest Neighbor")
    print("📊 Source: Complete 1,000-node Brazilian Transportation Network")
    
    try:
        # Initialize complete dataset tester
        tester = CompleteFourAlgorithmsDatasetTester()
        
        print(f"\n🎯 Test Configuration:")
        print(f"   Problem sizes: {problem_sizes}")
        print(f"   Iterations scale: {iterations_scale}x")
        print(f"   Data source: Complete 1000-node transportation network")
        print(f"   Algorithms: ONLY the 4 specifically requested")
        print(f"   Expected runtime: {len(problem_sizes) * 2:.0f}-{len(problem_sizes) * 5:.0f} minutes")
        
        # Run complete dataset testing
        results = tester.run_complete_dataset_testing(problem_sizes, iterations_scale)
        
        # Export results
        tester.export_complete_results()
        
        print("\n" + "="*80)
        print("🎊 COMPLETE FOUR ALGORITHMS DATASET ANALYSIS COMPLETED!")
        print("✅ Comprehensive testing on full Brazilian transportation network")
        print("📊 All 4 requested algorithms tested across multiple problem sizes")
        print("💾 Results exported to 'results/complete_four_algorithms_dataset/'")
        print("📈 Ready for detailed performance analysis and documentation")
        print("="*80)
        
        return True
        
    except Exception as e:
        print(f"\n❌ COMPLETE DATASET TESTING FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Run complete dataset testing for the 4 requested algorithms
    run_complete_four_algorithms_dataset_analysis()