"""
Algorithm convergence and performance plotting utilities.
FIAP Tech Challenge Phase 2 - Algorithm performance analysis visualization.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import numpy as np
from typing import List, Dict, Tuple, Optional
import seaborn as sns

# Set style
try:
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
except:
    # Fallback if seaborn style not available
    plt.style.use('default')
    plt.rcParams['figure.facecolor'] = 'white'

class ConvergencePlotter:
    """
    Utility class for plotting algorithm convergence and performance comparisons.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize the convergence plotter.
        
        Args:
            figsize: Figure size for plots
        """
        self.figsize = figsize
        self.colors = {
            'Genetic Algorithm': '#FF1493',  # Deep pink
            'Particle Swarm Optimization': '#FF8C00',  # Dark orange
            'Ant Colony Optimization': '#32CD32',  # Lime green
            'Nearest Neighbor': '#4169E1',  # Royal blue
            'Cheapest Insertion': '#8A2BE2',  # Blue violet
            'Farthest Insertion': '#DC143C',  # Crimson
            'Dijkstra-Enhanced Nearest Neighbor': '#20B2AA',  # Light sea green
            'A* Enhanced Nearest Neighbor': '#FF6347'  # Tomato
        }
    
    def plot_convergence_history(self, algorithm_histories: Dict[str, Dict], 
                                save_path: str = None, show: bool = True):
        """
        Plot convergence history for multiple algorithms.
        
        Args:
            algorithm_histories: Dictionary with algorithm names and their history
            save_path: Optional path to save the plot
            show: Whether to display the plot
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=self.figsize)
        fig.suptitle('FIAP Tech Challenge - Algorithm Convergence Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Best fitness over generations
        ax1.set_title('Best Distance Over Generations')
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('Best Distance (km)')
        
        for algo_name, history in algorithm_histories.items():
            if 'best_distances' in history or 'best_fitness' in history:
                best_data = history.get('best_distances', history.get('best_fitness', []))
                if best_data:
                    generations = range(len(best_data))
                    distances_km = [d / 1000 for d in best_data]  # Convert to km
                    color = self.colors.get(algo_name, '#666666')
                    ax1.plot(generations, distances_km, label=algo_name, color=color, linewidth=2)
        
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Average fitness over generations
        ax2.set_title('Average Distance Over Generations')
        ax2.set_xlabel('Generation')
        ax2.set_ylabel('Average Distance (km)')
        
        for algo_name, history in algorithm_histories.items():
            if 'average_distances' in history or 'average_fitness' in history:
                avg_data = history.get('average_distances', history.get('average_fitness', []))
                if avg_data:
                    generations = range(len(avg_data))
                    distances_km = [d / 1000 for d in avg_data]
                    color = self.colors.get(algo_name, '#666666')
                    ax2.plot(generations, distances_km, label=algo_name, color=color, linewidth=2, alpha=0.7)
        
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Diversity over generations (if available)
        ax3.set_title('Population Diversity Over Generations')
        ax3.set_xlabel('Generation')
        ax3.set_ylabel('Diversity Score')
        
        diversity_plotted = False
        for algo_name, history in algorithm_histories.items():
            if 'diversity_scores' in history or 'diversity' in history:
                diversity_data = history.get('diversity_scores', history.get('diversity', []))
                if diversity_data:
                    generations = range(len(diversity_data))
                    color = self.colors.get(algo_name, '#666666')
                    ax3.plot(generations, diversity_data, label=algo_name, color=color, linewidth=2)
                    diversity_plotted = True
        
        if diversity_plotted:
            ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'No diversity data available', 
                    ha='center', va='center', transform=ax3.transAxes)
        
        # Plot 4: Generation times (if available)
        ax4.set_title('Generation Processing Time')
        ax4.set_xlabel('Generation')
        ax4.set_ylabel('Time (seconds)')
        
        time_plotted = False
        for algo_name, history in algorithm_histories.items():
            if 'generation_times' in history:
                time_data = history['generation_times']
                if time_data:
                    generations = range(len(time_data))
                    color = self.colors.get(algo_name, '#666666')
                    ax4.plot(generations, time_data, label=algo_name, color=color, linewidth=2)
                    time_plotted = True
        
        if time_plotted:
            ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'No timing data available', 
                    ha='center', va='center', transform=ax4.transAxes)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Convergence plot saved to: {save_path}")
        
        if show:
            plt.show()
    
    def plot_algorithm_comparison(self, results: Dict[str, Dict], 
                                save_path: str = None, show: bool = True):
        """
        Plot comprehensive algorithm comparison.
        
        Args:
            results: Dictionary with algorithm results
            save_path: Optional path to save the plot
            show: Whether to display the plot
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=self.figsize)
        fig.suptitle('FIAP Tech Challenge - Algorithm Performance Comparison', fontsize=16, fontweight='bold')
        
        # Extract data
        algorithms = list(results.keys())
        distances = [results[algo]['distance'] / 1000 for algo in algorithms]  # Convert to km
        times = [results[algo]['time'] for algo in algorithms]
        categories = [results[algo].get('category', 'Unknown') for algo in algorithms]
        
        # Plot 1: Distance comparison bar chart
        colors_list = [self.colors.get(algo, '#666666') for algo in algorithms]
        bars1 = ax1.bar(range(len(algorithms)), distances, color=colors_list, alpha=0.7)
        ax1.set_title('Route Distance Comparison')
        ax1.set_xlabel('Algorithm')
        ax1.set_ylabel('Distance (km)')
        ax1.set_xticks(range(len(algorithms)))
        ax1.set_xticklabels([algo[:15] for algo in algorithms], rotation=45, ha='right')
        
        # Add value labels on bars
        for i, (bar, distance) in enumerate(zip(bars1, distances)):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{distance:.1f}', ha='center', va='bottom', fontsize=8)
        
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Plot 2: Execution time comparison
        bars2 = ax2.bar(range(len(algorithms)), times, color=colors_list, alpha=0.7)
        ax2.set_title('Execution Time Comparison')
        ax2.set_xlabel('Algorithm')
        ax2.set_ylabel('Time (seconds)')
        ax2.set_xticks(range(len(algorithms)))
        ax2.set_xticklabels([algo[:15] for algo in algorithms], rotation=45, ha='right')
        
        # Add value labels on bars
        for i, (bar, time_val) in enumerate(zip(bars2, times)):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{time_val:.3f}', ha='center', va='bottom', fontsize=8)
        
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Plot 3: Efficiency scatter plot (Distance vs Time)
        category_colors = {'Evolutionary': '#FF1493', 'Conventional': '#4169E1', 'Metaheuristic': '#32CD32'}
        plotted_categories = set()
        for i, (distance, time_val, category) in enumerate(zip(distances, times, categories)):
            color = category_colors.get(category, '#666666')
            label = category if category not in plotted_categories else ""
            if label:
                plotted_categories.add(category)
            ax3.scatter(time_val, distance, c=color, s=100, alpha=0.7, label=label)
            ax3.annotate(algorithms[i][:10], (time_val, distance), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax3.set_title('Algorithm Efficiency (Distance vs Time)')
        ax3.set_xlabel('Execution Time (seconds)')
        ax3.set_ylabel('Distance (km)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Category performance box plot
        category_data = {}
        for algo, result in results.items():
            cat = result.get('category', 'Unknown')
            if cat not in category_data:
                category_data[cat] = []
            category_data[cat].append(result['distance'] / 1000)
        
        if len(category_data) > 1:
            categories_list = list(category_data.keys())
            data_list = [category_data[cat] for cat in categories_list]
            
            box_plot = ax4.boxplot(data_list, labels=categories_list, patch_artist=True)
            
            # Color the boxes
            for patch, cat in zip(box_plot['boxes'], categories_list):
                patch.set_facecolor(category_colors.get(cat, '#666666'))
                patch.set_alpha(0.7)
            
            ax4.set_title('Performance by Algorithm Category')
            ax4.set_ylabel('Distance (km)')
            ax4.grid(True, alpha=0.3, axis='y')
        else:
            ax4.text(0.5, 0.5, 'Insufficient categories for comparison', 
                    ha='center', va='center', transform=ax4.transAxes)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Comparison plot saved to: {save_path}")
        
        if show:
            plt.show()
    
    def plot_route_comparison(self, coordinates: List[Tuple[float, float]], 
                            routes: Dict[str, List[int]], 
                            save_path: str = None, show: bool = True):
        """
        Plot route comparison on a map.
        
        Args:
            coordinates: List of city coordinates
            routes: Dictionary of algorithm routes
            save_path: Optional path to save the plot
            show: Whether to display the plot
        """
        n_routes = len(routes)
        if n_routes == 0:
            return
        
        # Determine subplot layout
        cols = min(3, n_routes)
        rows = (n_routes + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
        if n_routes == 1:
            axes = [axes]
        elif rows == 1:
            axes = [axes] if cols == 1 else axes
        else:
            axes = axes.flatten()
        
        fig.suptitle('FIAP Tech Challenge - Route Comparison', fontsize=16, fontweight='bold')
        
        # Extract coordinates
        lons, lats = zip(*coordinates)
        
        for i, (algo_name, route) in enumerate(routes.items()):
            ax = axes[i]
            
            # Plot cities
            ax.scatter(lons, lats, c='blue', s=50, alpha=0.7, zorder=3)
            
            # Add city labels
            for j, (lon, lat) in enumerate(coordinates):
                ax.annotate(str(j), (lon, lat), xytext=(3, 3), 
                           textcoords='offset points', fontsize=8)
            
            # Plot route
            if len(route) > 1:
                route_lons = [coordinates[city][0] for city in route] + [coordinates[route[0]][0]]
                route_lats = [coordinates[city][1] for city in route] + [coordinates[route[0]][1]]
                
                color = self.colors.get(algo_name, '#666666')
                ax.plot(route_lons, route_lats, color=color, linewidth=2, alpha=0.8, zorder=2)
                
                # Add arrows to show direction
                for k in range(len(route)):
                    start_idx = route[k]
                    end_idx = route[(k + 1) % len(route)]
                    
                    start_lon, start_lat = coordinates[start_idx]
                    end_lon, end_lat = coordinates[end_idx]
                    
                    # Arrow in the middle of the segment
                    mid_lon = (start_lon + end_lon) / 2
                    mid_lat = (start_lat + end_lat) / 2
                    
                    dx = end_lon - start_lon
                    dy = end_lat - start_lat
                    
                    ax.arrow(mid_lon, mid_lat, dx*0.1, dy*0.1, 
                            head_width=0.001, head_length=0.001, 
                            fc=color, ec=color, alpha=0.6)
            
            ax.set_title(f'{algo_name[:20]}', fontsize=10)
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal', adjustable='box')
        
        # Hide unused subplots
        for i in range(n_routes, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Route comparison plot saved to: {save_path}")
        
        if show:
            plt.show()
    
    def create_summary_report(self, results: Dict[str, Dict], 
                            algorithm_histories: Dict[str, Dict] = None,
                            save_path: str = None):
        """
        Create a comprehensive summary report with multiple plots.
        
        Args:
            results: Algorithm results
            algorithm_histories: Algorithm convergence histories
            save_path: Base path for saving plots
        """
        print("📊 Generating comprehensive algorithm analysis report...")
        
        # 1. Algorithm comparison
        comparison_path = f"{save_path}_comparison.png" if save_path else None
        self.plot_algorithm_comparison(results, comparison_path, show=False)
        
        # 2. Convergence analysis (if data available)
        if algorithm_histories:
            convergence_path = f"{save_path}_convergence.png" if save_path else None
            self.plot_convergence_history(algorithm_histories, convergence_path, show=False)
        
        print("✅ Algorithm analysis report completed!")

def create_performance_dashboard(results: Dict, histories: Dict = None, 
                               coordinates: List = None, routes: Dict = None):
    """
    Create a comprehensive performance dashboard.
    
    Args:
        results: Algorithm results
        histories: Algorithm histories
        coordinates: City coordinates
        routes: Algorithm routes
    """
    plotter = ConvergencePlotter(figsize=(15, 10))
    
    # Create all plots
    if results:
        plotter.plot_algorithm_comparison(results, "algorithm_comparison.png")
    
    if histories:
        plotter.plot_convergence_history(histories, "convergence_analysis.png")
    
    if coordinates and routes:
        plotter.plot_route_comparison(coordinates, routes, "route_comparison.png")
    
    print("📈 Performance dashboard created successfully!")