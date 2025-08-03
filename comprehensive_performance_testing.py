"""
Comprehensive Performance Testing Suite
FIAP Tech Challenge Phase 2 - Statistical analysis with varying problem sizes.
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
import statistics
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import json
import os

class PerformanceTester:
    """
    Comprehensive performance testing system for TSP algorithms.
    Tests multiple problem sizes with statistical analysis.
    """
    
    def __init__(self):
        """Initialize the performance tester."""
        self.test_results = {}
        self.statistical_summary = {}
        
    def run_algorithm_suite(self, num_cities: int, runs: int = 3) -> Dict:
        """
        Run complete algorithm suite for a given problem size.
        
        Args:
            num_cities: Number of cities for TSP problem
            runs: Number of runs for statistical analysis
            
        Returns:
            Dictionary containing all algorithm results
        """
        print(f"\n🧪 Testing {num_cities} cities with {runs} runs for statistical analysis...")
        
        # Load data once for all runs
        loader = load_transportation_data(sample_nodes=num_cities)
        node_ids = list(loader.graph.nodes())
        coordinates = [loader.get_node_coordinates(nid) for nid in node_ids]
        
        calculator = DistanceCalculator(coordinates)
        distance_matrix = calculator.get_distance_matrix()
        
        print(f"✅ Data loaded: {len(coordinates)} cities")
        
        all_runs_results = {}
        
        for run in range(runs):
            print(f"   Run {run + 1}/{runs}...")
            run_results = {}
            
            # 1. GENETIC ALGORITHM
            ga_config = GeneticConfig(
                population_size=min(100, num_cities * 5),
                generations=min(200, num_cities * 8),
                elite_size=max(5, min(20, num_cities)),
                mutation_rate=0.02,
                selection_method="tournament",
                crossover_method="order",
                mutation_method="swap"
            )
            
            start_time = time.time()
            ga = GeneticAlgorithm(distance_matrix, ga_config)
            ga_best = ga.run(verbose=False)
            ga_time = time.time() - start_time
            
            run_results['Genetic Algorithm'] = {
                'distance': ga_best.distance,
                'time': ga_time,
                'convergence_rate': ga._calculate_convergence_rate(),
                'final_diversity': ga.history['diversity_scores'][-1] if ga.history['diversity_scores'] else 0,
                'generations': ga.generation,
                'category': 'Evolutionary'
            }
            
            # 2. CONVENTIONAL ALGORITHMS
            start_time = time.time()
            conventional_results = run_conventional_algorithms(distance_matrix, coordinates)
            conventional_time = time.time() - start_time
            
            for result in conventional_results:
                run_results[result.algorithm_name] = {
                    'distance': result.distance,
                    'time': result.execution_time,
                    'nodes_explored': result.nodes_explored,
                    'category': 'Conventional'
                }
            
            # 3. METAHEURISTIC ALGORITHMS
            meta_config = MetaheuristicConfig(
                population_size=min(50, num_cities * 3),
                max_iterations=min(100, num_cities * 4),
                num_ants=min(50, num_cities * 3)
            )
            
            start_time = time.time()
            metaheuristic_results = run_metaheuristic_algorithms(distance_matrix, meta_config)
            meta_time = time.time() - start_time
            
            # PSO results
            pso_result = metaheuristic_results['PSO']
            run_results['Particle Swarm Optimization'] = {
                'distance': pso_result['distance'],
                'time': meta_time / 2,
                'final_diversity': pso_result['history']['diversity'][-1] if pso_result['history']['diversity'] else 0,
                'category': 'Metaheuristic'
            }
            
            # ACO results
            aco_result = metaheuristic_results['ACO']
            run_results['Ant Colony Optimization'] = {
                'distance': aco_result['distance'],
                'time': meta_time / 2,
                'category': 'Metaheuristic'
            }
            
            all_runs_results[f'run_{run + 1}'] = run_results
        
        print(f"✅ Completed {runs} runs for {num_cities} cities")
        return all_runs_results
    
    def calculate_statistics(self, all_runs: Dict) -> Dict:
        """
        Calculate statistical measures across multiple runs.
        
        Args:
            all_runs: Results from multiple runs
            
        Returns:
            Statistical summary
        """
        algorithms = set()
        for run_data in all_runs.values():
            algorithms.update(run_data.keys())
        
        stats = {}
        
        for algorithm in algorithms:
            # Collect data from all runs
            distances = []
            times = []
            
            for run_data in all_runs.values():
                if algorithm in run_data:
                    distances.append(run_data[algorithm]['distance'])
                    times.append(run_data[algorithm]['time'])
            
            if distances:  # Only calculate if we have data
                stats[algorithm] = {
                    'distance_mean': statistics.mean(distances),
                    'distance_std': statistics.stdev(distances) if len(distances) > 1 else 0,
                    'distance_min': min(distances),
                    'distance_max': max(distances),
                    'time_mean': statistics.mean(times),
                    'time_std': statistics.stdev(times) if len(times) > 1 else 0,
                    'time_min': min(times),
                    'time_max': max(times),
                    'runs': len(distances),
                    'category': all_runs[list(all_runs.keys())[0]][algorithm]['category']
                }
        
        return stats
    
    def run_scalability_test(self, problem_sizes: List[int] = None, runs_per_size: int = 3):
        """
        Run scalability test across different problem sizes.
        
        Args:
            problem_sizes: List of problem sizes to test
            runs_per_size: Number of runs per problem size
        """
        if problem_sizes is None:
            problem_sizes = [8, 12, 16, 20]
        
        print("🚀 COMPREHENSIVE SCALABILITY TESTING")
        print("=" * 80)
        print(f"Problem sizes: {problem_sizes}")
        print(f"Runs per size: {runs_per_size}")
        print(f"Total tests: {len(problem_sizes) * runs_per_size}")
        
        all_results = {}
        
        for size in problem_sizes:
            print(f"\n{'='*20} TESTING {size} CITIES {'='*20}")
            
            # Run algorithm suite for this problem size
            size_results = self.run_algorithm_suite(size, runs_per_size)
            
            # Calculate statistics
            size_stats = self.calculate_statistics(size_results)
            
            all_results[f'{size}_cities'] = {
                'raw_results': size_results,
                'statistics': size_stats,
                'problem_size': size
            }
            
            # Display summary for this size
            self.display_size_summary(size, size_stats)
        
        self.test_results = all_results
        return all_results
    
    def display_size_summary(self, size: int, stats: Dict):
        """
        Display summary statistics for a problem size.
        
        Args:
            size: Problem size
            stats: Statistical summary
        """
        print(f"\n📊 SUMMARY FOR {size} CITIES:")
        print("-" * 50)
        
        # Sort by mean distance
        sorted_algorithms = sorted(stats.items(), key=lambda x: x[1]['distance_mean'])
        
        print(f"{'Algorithm':<25} {'Avg Dist (km)':<12} {'Std Dev':<8} {'Avg Time (s)':<12}")
        print("-" * 65)
        
        for i, (algo_name, stat) in enumerate(sorted_algorithms[:5]):  # Top 5
            distance_km = stat['distance_mean'] / 1000
            std_dev = stat['distance_std']
            avg_time = stat['time_mean']
            
            rank_emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "  "
            
            print(f"{rank_emoji} {algo_name[:23]:<23} {distance_km:<12.2f} {std_dev:<8.0f} {avg_time:<12.4f}")
    
    def generate_comprehensive_report(self, save_to_file: bool = True):
        """
        Generate comprehensive performance report.
        
        Args:
            save_to_file: Whether to save results to files
        """
        if not self.test_results:
            print("❌ No test results available. Run scalability test first.")
            return
        
        print("\n📈 GENERATING COMPREHENSIVE PERFORMANCE REPORT")
        print("=" * 80)
        
        # Create results directory
        os.makedirs('results/comprehensive_testing', exist_ok=True)
        
        # 1. Scalability Analysis
        self.analyze_scalability()
        
        # 2. Algorithm Category Analysis
        self.analyze_by_category()
        
        # 3. Statistical Significance Analysis
        self.analyze_statistical_significance()
        
        # 4. Performance Consistency Analysis
        self.analyze_consistency()
        
        # 5. Export results
        if save_to_file:
            self.export_results()
        
        print("\n✅ Comprehensive performance report completed!")
    
    def analyze_scalability(self):
        """Analyze how algorithms scale with problem size."""
        print("\n🔍 SCALABILITY ANALYSIS:")
        print("-" * 30)
        
        # Extract scalability data
        sizes = []
        ga_times = []
        ga_distances = []
        
        for size_key, data in self.test_results.items():
            size = data['problem_size']
            sizes.append(size)
            
            if 'Genetic Algorithm' in data['statistics']:
                ga_stat = data['statistics']['Genetic Algorithm']
                ga_times.append(ga_stat['time_mean'])
                ga_distances.append(ga_stat['distance_mean'])
        
        if len(sizes) > 1:
            # Calculate time complexity growth
            time_growth_rates = []
            for i in range(1, len(sizes)):
                prev_time = ga_times[i-1]
                curr_time = ga_times[i]
                size_ratio = sizes[i] / sizes[i-1]
                time_ratio = curr_time / prev_time if prev_time > 0 else 0
                
                growth_rate = time_ratio / size_ratio
                time_growth_rates.append(growth_rate)
            
            avg_growth = statistics.mean(time_growth_rates) if time_growth_rates else 0
            
            print(f"• Genetic Algorithm time complexity growth: {avg_growth:.2f}x per city increase")
            
            if avg_growth < 1.5:
                print("  → Excellent scalability (sub-linear growth)")
            elif avg_growth < 2.5:
                print("  → Good scalability (near-linear growth)")
            else:
                print("  → Moderate scalability (super-linear growth)")
    
    def analyze_by_category(self):
        """Analyze performance by algorithm category."""
        print("\n📊 CATEGORY PERFORMANCE ANALYSIS:")
        print("-" * 35)
        
        category_performance = {}
        
        for size_key, data in self.test_results.items():
            size = data['problem_size']
            
            for algo_name, stats in data['statistics'].items():
                category = stats['category']
                
                if category not in category_performance:
                    category_performance[category] = {'distances': [], 'times': [], 'sizes': []}
                
                category_performance[category]['distances'].append(stats['distance_mean'])
                category_performance[category]['times'].append(stats['time_mean'])
                category_performance[category]['sizes'].append(size)
        
        for category, perf_data in category_performance.items():
            avg_distance = statistics.mean(perf_data['distances']) / 1000
            avg_time = statistics.mean(perf_data['times'])
            
            print(f"• {category}:")
            print(f"  Average distance: {avg_distance:.2f} km")
            print(f"  Average time: {avg_time:.4f} seconds")
    
    def analyze_statistical_significance(self):
        """Analyze statistical significance of results."""
        print("\n📈 STATISTICAL SIGNIFICANCE ANALYSIS:")
        print("-" * 40)
        
        for size_key, data in self.test_results.items():
            size = data['problem_size']
            print(f"\n{size} cities:")
            
            # Find algorithms with lowest standard deviation (most consistent)
            consistent_algos = sorted(
                data['statistics'].items(),
                key=lambda x: x[1]['distance_std']
            )[:3]
            
            print("  Most consistent algorithms:")
            for i, (algo_name, stats) in enumerate(consistent_algos, 1):
                std_pct = (stats['distance_std'] / stats['distance_mean']) * 100
                print(f"    {i}. {algo_name[:25]}: {std_pct:.1f}% variation")
    
    def analyze_consistency(self):
        """Analyze performance consistency across problem sizes."""
        print("\n🎯 CONSISTENCY ANALYSIS:")
        print("-" * 25)
        
        # Track how often each algorithm ranks in top 3
        algorithm_rankings = {}
        
        for size_key, data in self.test_results.items():
            size = data['problem_size']
            
            # Rank algorithms by mean distance
            ranked_algos = sorted(
                data['statistics'].items(),
                key=lambda x: x[1]['distance_mean']
            )
            
            for rank, (algo_name, _) in enumerate(ranked_algos[:3], 1):
                if algo_name not in algorithm_rankings:
                    algorithm_rankings[algo_name] = []
                algorithm_rankings[algo_name].append(rank)
        
        print("Top performers across problem sizes:")
        consistent_performers = []
        
        for algo_name, rankings in algorithm_rankings.items():
            avg_rank = statistics.mean(rankings)
            consistency_score = len(rankings)  # How often in top 3
            
            consistent_performers.append((algo_name, avg_rank, consistency_score))
        
        # Sort by average rank
        consistent_performers.sort(key=lambda x: x[1])
        
        for i, (algo_name, avg_rank, appearances) in enumerate(consistent_performers[:5], 1):
            print(f"  {i}. {algo_name[:25]}: Avg rank {avg_rank:.1f}, {appearances}/{len(self.test_results)} tests")
    
    def export_results(self):
        """Export results to files."""
        # Export raw results to JSON
        with open('results/comprehensive_testing/raw_results.json', 'w') as f:
            json.dump(self.test_results, f, indent=2)
        
        # Export summary statistics to CSV
        summary_data = []
        
        for size_key, data in self.test_results.items():
            size = data['problem_size']
            
            for algo_name, stats in data['statistics'].items():
                summary_data.append({
                    'problem_size': size,
                    'algorithm': algo_name,
                    'category': stats['category'],
                    'distance_mean_km': stats['distance_mean'] / 1000,
                    'distance_std': stats['distance_std'],
                    'time_mean': stats['time_mean'],
                    'time_std': stats['time_std'],
                    'runs': stats['runs']
                })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv('results/comprehensive_testing/summary_statistics.csv', index=False)
        
        print("💾 Results exported:")
        print("  • Raw results: results/comprehensive_testing/raw_results.json")
        print("  • Summary: results/comprehensive_testing/summary_statistics.csv")

def main():
    """Run comprehensive performance testing."""
    print("🌟 FIAP Tech Challenge - Comprehensive Performance Testing")
    print("=" * 80)
    
    try:
        # Initialize tester
        tester = PerformanceTester()
        
        # Define test parameters
        problem_sizes = [8, 12, 16]  # Start with manageable sizes
        runs_per_size = 3  # Statistical significance
        
        print(f"🎯 Test Configuration:")
        print(f"   Problem sizes: {problem_sizes}")
        print(f"   Runs per size: {runs_per_size}")
        print(f"   Total algorithm executions: {len(problem_sizes) * runs_per_size * 8}")
        
        # Run scalability test
        results = tester.run_scalability_test(problem_sizes, runs_per_size)
        
        # Generate comprehensive report
        tester.generate_comprehensive_report(save_to_file=True)
        
        print("\n" + "="*80)
        print("🎊 COMPREHENSIVE PERFORMANCE TESTING COMPLETED!")
        print("✅ Statistical analysis across multiple problem sizes")
        print("📊 Scalability and consistency analysis performed")
        print("💾 Detailed results exported to 'results/comprehensive_testing/'")
        print("🚀 Ready for final Tech Challenge documentation")
        print("="*80)
        
        return True
        
    except Exception as e:
        print(f"\n❌ TESTING FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main()