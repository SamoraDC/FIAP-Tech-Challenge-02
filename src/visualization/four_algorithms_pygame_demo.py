"""
Four Algorithms Pygame Visualization Demo
FIAP Tech Challenge Phase 2 - Real-time visualization of ONLY the 4 requested algorithms.

ALGORITHMS VISUALIZED:
1. Particle Swarm Optimization
2. Ant Colony Optimization  
3. Dijkstra-Enhanced Nearest Neighbor
4. A* Enhanced Nearest Neighbor
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pygame
import math
import time
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import numpy as np

from utils.data_loader import load_transportation_data
from utils.distance_utils import DistanceCalculator
from algorithms.four_focused_algorithms import run_four_focused_algorithms, FocusedConfig

# Initialize Pygame
pygame.init()

@dataclass
class FourAlgorithmsVisualizationConfig:
    """Configuration for the 4 algorithms visualizer."""
    window_width: int = 1400
    window_height: int = 900
    map_margin: int = 60
    panel_width: int = 350
    city_radius: int = 8
    route_width: int = 4
    font_size: int = 18
    title_font_size: int = 26
    
    # Colors for the 4 algorithms
    colors: Dict[str, Tuple[int, int, int]] = None
    
    def __post_init__(self):
        if self.colors is None:
            self.colors = {
                'Particle Swarm Optimization': (255, 100, 100),    # Red
                'Ant Colony Optimization': (100, 255, 100),        # Green  
                'Dijkstra-Enhanced Nearest Neighbor': (100, 100, 255),  # Blue
                'A* Enhanced Nearest Neighbor': (255, 255, 100),   # Yellow
                'cities': (80, 80, 80),                           # Dark gray
                'background': (240, 240, 240),                    # Light gray
                'panel': (220, 220, 220),                         # Panel gray
                'text': (40, 40, 40),                            # Dark text
                'grid': (200, 200, 200),                         # Light grid
                'highlight': (255, 150, 0)                       # Orange highlight
            }

class FourAlgorithmsTSPVisualizer:
    """
    Real-time visualization system for ONLY the 4 requested TSP algorithms.
    Shows routes, performance comparisons, and interactive controls.
    """
    
    def __init__(self, config: FourAlgorithmsVisualizationConfig = None):
        """Initialize the focused visualizer."""
        self.config = config or FourAlgorithmsVisualizationConfig()
        
        # Pygame setup
        self.screen = pygame.display.set_mode((self.config.window_width, self.config.window_height))
        pygame.display.set_caption("Four Algorithms TSP Visualization - FIAP Tech Challenge")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, self.config.font_size)
        self.title_font = pygame.font.Font(None, self.config.title_font_size)
        
        # Data storage
        self.coordinates: List[Tuple[float, float]] = []
        self.city_names: List[str] = []
        self.algorithm_results: Dict[str, any] = {}
        
        # Visualization state
        self.visible_algorithms = set()
        self.highlighted_algorithm = None
        self.show_distance_info = True
        self.show_performance_stats = True
        self.animation_progress = 0.0
        
        # Map bounds
        self.min_lon = self.max_lon = self.min_lat = self.max_lat = 0
        self.map_width = self.config.window_width - self.config.panel_width - 2 * self.config.map_margin
        self.map_height = self.config.window_height - 2 * self.config.map_margin
    
    def set_data(self, coordinates: List[Tuple[float, float]], city_names: List[str] = None):
        """
        Set the city coordinates and names for visualization.
        
        Args:
            coordinates: List of (longitude, latitude) tuples
            city_names: Optional list of city names
        """
        self.coordinates = coordinates
        self.city_names = city_names or [f"City {i}" for i in range(len(coordinates))]
        
        # Calculate map bounds
        if coordinates:
            lons, lats = zip(*coordinates)
            self.min_lon, self.max_lon = min(lons), max(lons)
            self.min_lat, self.max_lat = min(lats), max(lats)
        
        print(f"📍 Visualization data set: {len(coordinates)} cities")
        print(f"   Geographic bounds: Longitude [{self.min_lon:.3f}, {self.max_lon:.3f}]")
        print(f"                     Latitude  [{self.min_lat:.3f}, {self.max_lat:.3f}]")
    
    def add_algorithm_results(self, results: Dict[str, any]):
        """
        Add results from the 4 algorithms.
        
        Args:
            results: Dictionary containing results from all 4 algorithms
        """
        self.algorithm_results = results
        self.visible_algorithms = set(results.keys())
        
        print(f"🎯 Added results for {len(results)} algorithms:")
        for algo_name, result in results.items():
            distance_km = result.distance / 1000
            time_s = result.execution_time
            print(f"   • {algo_name}: {distance_km:.2f} km, {time_s:.3f}s")
    
    def lonlat_to_screen(self, lon: float, lat: float) -> Tuple[int, int]:
        """Convert longitude/latitude to screen coordinates."""
        if self.max_lon == self.min_lon or self.max_lat == self.min_lat:
            return (self.config.map_margin + self.map_width // 2, 
                   self.config.map_margin + self.map_height // 2)
        
        x = self.config.map_margin + int((lon - self.min_lon) / (self.max_lon - self.min_lon) * self.map_width)
        y = self.config.map_margin + int((1 - (lat - self.min_lat) / (self.max_lat - self.min_lat)) * self.map_height)
        
        return (x, y)
    
    def draw_coordinate_grid(self):
        """Draw coordinate grid for geographic reference."""
        grid_color = self.config.colors['grid']
        
        # Vertical lines (longitude)
        for i in range(5):
            x = self.config.map_margin + int(i * self.map_width / 4)
            pygame.draw.line(self.screen, grid_color, 
                           (x, self.config.map_margin), 
                           (x, self.config.map_margin + self.map_height), 1)
        
        # Horizontal lines (latitude)
        for i in range(5):
            y = self.config.map_margin + int(i * self.map_height / 4)
            pygame.draw.line(self.screen, grid_color,
                           (self.config.map_margin, y),
                           (self.config.map_margin + self.map_width, y), 1)
    
    def draw_cities(self):
        """Draw all cities as dots."""
        city_color = self.config.colors['cities']
        
        for i, (lon, lat) in enumerate(self.coordinates):
            x, y = self.lonlat_to_screen(lon, lat)
            pygame.draw.circle(self.screen, city_color, (x, y), self.config.city_radius)
            pygame.draw.circle(self.screen, (255, 255, 255), (x, y), self.config.city_radius - 2)
            
            # Draw city number
            if self.config.city_radius >= 6:
                text = self.font.render(str(i), True, (0, 0, 0))
                text_rect = text.get_rect(center=(x, y))
                self.screen.blit(text, text_rect)
    
    def draw_route(self, route: List[int], color: Tuple[int, int, int], 
                   width: int = None, alpha: float = 1.0, animate: bool = False):
        """
        Draw a TSP route with optional animation.
        
        Args:
            route: List of city indices
            color: RGB color tuple
            width: Line width (uses config default if None)
            alpha: Transparency (0.0 to 1.0)
            animate: Whether to animate route drawing
        """
        if not route or len(route) < 2:
            return
        
        width = width or self.config.route_width
        
        # Create surface for alpha blending if needed
        if alpha < 1.0:
            route_surface = pygame.Surface((self.config.window_width, self.config.window_height))
            route_surface.set_alpha(int(alpha * 255))
            draw_surface = route_surface
        else:
            draw_surface = self.screen
        
        # Convert route to screen coordinates
        points = []
        for city_idx in route:
            if 0 <= city_idx < len(self.coordinates):
                lon, lat = self.coordinates[city_idx]
                x, y = self.lonlat_to_screen(lon, lat)
                points.append((x, y))
        
        # Add return to start for closed loop
        if points:
            points.append(points[0])
        
        # Draw route segments
        num_segments = len(points) - 1
        segments_to_draw = num_segments
        
        if animate:
            segments_to_draw = int(self.animation_progress * num_segments)
        
        for i in range(segments_to_draw):
            if i + 1 < len(points):
                pygame.draw.line(draw_surface, color, points[i], points[i + 1], width)
        
        # Draw partially completed segment if animating
        if animate and segments_to_draw < num_segments and segments_to_draw + 1 < len(points):
            partial_progress = (self.animation_progress * num_segments) - segments_to_draw
            start_point = points[segments_to_draw]
            end_point = points[segments_to_draw + 1]
            
            partial_x = start_point[0] + (end_point[0] - start_point[0]) * partial_progress
            partial_y = start_point[1] + (end_point[1] - start_point[1]) * partial_progress
            
            pygame.draw.line(draw_surface, color, start_point, (int(partial_x), int(partial_y)), width)
        
        # Blit alpha surface to main screen if needed
        if alpha < 1.0:
            self.screen.blit(route_surface, (0, 0))
        
        # Draw direction arrows on route
        if not animate or segments_to_draw > 3:
            self.draw_route_arrows(points[:segments_to_draw + 1], color)
    
    def draw_route_arrows(self, points: List[Tuple[int, int]], color: Tuple[int, int, int]):
        """Draw arrows showing route direction."""
        arrow_color = color
        arrow_size = 8
        
        for i in range(0, len(points) - 1, max(1, len(points) // 8)):  # Show arrows at intervals
            if i + 1 < len(points):
                start_x, start_y = points[i]
                end_x, end_y = points[i + 1]
                
                # Calculate arrow direction
                dx = end_x - start_x
                dy = end_y - start_y
                length = math.sqrt(dx**2 + dy**2)
                
                if length > 0:
                    # Normalize
                    dx /= length
                    dy /= length
                    
                    # Arrow position (middle of segment)
                    arrow_x = start_x + 0.7 * (end_x - start_x)
                    arrow_y = start_y + 0.7 * (end_y - start_y)
                    
                    # Arrow points
                    arrow_points = [
                        (arrow_x, arrow_y),
                        (arrow_x - arrow_size * dx - arrow_size * dy * 0.5, 
                         arrow_y - arrow_size * dy + arrow_size * dx * 0.5),
                        (arrow_x - arrow_size * dx + arrow_size * dy * 0.5, 
                         arrow_y - arrow_size * dy - arrow_size * dx * 0.5)
                    ]
                    
                    pygame.draw.polygon(self.screen, arrow_color, arrow_points)
    
    def draw_info_panel(self):
        """Draw information panel with algorithm details."""
        panel_x = self.config.window_width - self.config.panel_width
        panel_color = self.config.colors['panel']
        text_color = self.config.colors['text']
        
        # Draw panel background
        pygame.draw.rect(self.screen, panel_color, 
                        (panel_x, 0, self.config.panel_width, self.config.window_height))
        
        # Panel border
        pygame.draw.line(self.screen, (180, 180, 180), 
                        (panel_x, 0), (panel_x, self.config.window_height), 2)
        
        y_offset = 20
        
        # Title
        title_text = self.title_font.render("4 Algorithms TSP", True, text_color)
        self.screen.blit(title_text, (panel_x + 10, y_offset))
        y_offset += 40
        
        # Algorithm list and controls
        if self.show_performance_stats:
            stats_text = self.font.render("Performance Stats:", True, text_color)
            self.screen.blit(stats_text, (panel_x + 10, y_offset))
            y_offset += 30
            
            # Sort algorithms by distance for ranking
            sorted_algos = sorted(self.algorithm_results.items(), 
                                key=lambda x: x[1].distance)
            
            for i, (algo_name, result) in enumerate(sorted_algos, 1):
                color = self.config.colors.get(algo_name, (128, 128, 128))
                distance_km = result.distance / 1000
                time_s = result.execution_time
                
                # Rank indicator
                rank_emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "4️⃣"
                
                # Algorithm visibility checkbox
                checkbox = "☑" if algo_name in self.visible_algorithms else "☐"
                
                # Algorithm info
                algo_short = algo_name.replace("Optimization", "Opt").replace("Enhanced", "Enh")[:20]
                info_text = f"{checkbox} {rank_emoji} {algo_short}"
                text_surface = self.font.render(info_text, True, color)
                self.screen.blit(text_surface, (panel_x + 10, y_offset))
                y_offset += 20
                
                # Distance and time
                dist_text = f"   {distance_km:.2f} km, {time_s:.3f}s"
                dist_surface = self.font.render(dist_text, True, text_color)
                self.screen.blit(dist_surface, (panel_x + 15, y_offset))
                y_offset += 25
        
        # Controls
        y_offset += 20
        controls_text = self.font.render("Controls:", True, text_color)
        self.screen.blit(controls_text, (panel_x + 10, y_offset))
        y_offset += 25
        
        controls = [
            "1-4: Toggle algorithms",
            "A: Toggle all routes",
            "S: Toggle stats",
            "C: Clear all routes",
            "SPACE: Animation",
            "ESC: Exit"
        ]
        
        for control in controls:
            control_surface = self.font.render(control, True, text_color)
            self.screen.blit(control_surface, (panel_x + 10, y_offset))
            y_offset += 18
        
        # Dataset info
        y_offset += 30
        dataset_text = self.font.render("Dataset Info:", True, text_color)
        self.screen.blit(dataset_text, (panel_x + 10, y_offset))
        y_offset += 25
        
        info_lines = [
            f"Cities: {len(self.coordinates)}",
            f"Source: Brazilian Network",
            f"Algorithms: 4 focused",
            f"Real-time visualization"
        ]
        
        for info in info_lines:
            info_surface = self.font.render(info, True, text_color)
            self.screen.blit(info_surface, (panel_x + 10, y_offset))
            y_offset += 18
    
    def draw_title_bar(self):
        """Draw title bar with current status."""
        title_color = self.config.colors['text']
        
        # Main title
        title = "FIAP Tech Challenge - Four Algorithms TSP Visualization"
        title_surface = self.title_font.render(title, True, title_color)
        title_rect = title_surface.get_rect(center=(self.config.window_width // 2, 25))
        self.screen.blit(title_surface, title_rect)
        
        # Status line
        visible_count = len(self.visible_algorithms)
        status = f"Showing {visible_count}/4 algorithms | {len(self.coordinates)} cities"
        status_surface = self.font.render(status, True, title_color)
        status_rect = status_surface.get_rect(center=(self.config.window_width // 2, 50))
        self.screen.blit(status_surface, status_rect)
    
    def handle_event(self, event):
        """Handle pygame events."""
        if event.type == pygame.KEYDOWN:
            algo_names = list(self.algorithm_results.keys())
            
            if event.key == pygame.K_1 and len(algo_names) > 0:
                # Toggle first algorithm
                algo = algo_names[0]
                if algo in self.visible_algorithms:
                    self.visible_algorithms.remove(algo)
                else:
                    self.visible_algorithms.add(algo)
                    
            elif event.key == pygame.K_2 and len(algo_names) > 1:
                # Toggle second algorithm
                algo = algo_names[1]
                if algo in self.visible_algorithms:
                    self.visible_algorithms.remove(algo)
                else:
                    self.visible_algorithms.add(algo)
                    
            elif event.key == pygame.K_3 and len(algo_names) > 2:
                # Toggle third algorithm
                algo = algo_names[2]
                if algo in self.visible_algorithms:
                    self.visible_algorithms.remove(algo)
                else:
                    self.visible_algorithms.add(algo)
                    
            elif event.key == pygame.K_4 and len(algo_names) > 3:
                # Toggle fourth algorithm
                algo = algo_names[3]
                if algo in self.visible_algorithms:
                    self.visible_algorithms.remove(algo)
                else:
                    self.visible_algorithms.add(algo)
                    
            elif event.key == pygame.K_a:
                # Toggle all algorithms
                if len(self.visible_algorithms) == len(self.algorithm_results):
                    self.visible_algorithms.clear()
                else:
                    self.visible_algorithms = set(self.algorithm_results.keys())
                    
            elif event.key == pygame.K_c:
                # Clear all routes
                self.visible_algorithms.clear()
                
            elif event.key == pygame.K_s:
                # Toggle stats
                self.show_performance_stats = not self.show_performance_stats
                
            elif event.key == pygame.K_SPACE:
                # Toggle animation
                self.animation_progress = 0.0
                
            elif event.key == pygame.K_ESCAPE:
                return False  # Signal to quit
        
        return True  # Continue running
    
    def render(self):
        """Render the complete visualization."""
        # Clear screen
        self.screen.fill(self.config.colors['background'])
        
        # Draw coordinate grid
        self.draw_coordinate_grid()
        
        # Draw cities
        self.draw_cities()
        
        # Draw routes for visible algorithms
        for algo_name in self.visible_algorithms:
            if algo_name in self.algorithm_results:
                result = self.algorithm_results[algo_name]
                color = self.config.colors.get(algo_name, (128, 128, 128))
                
                # Highlight selected algorithm
                width = self.config.route_width
                alpha = 1.0
                
                if self.highlighted_algorithm == algo_name:
                    width += 2
                    color = self.config.colors['highlight']
                
                self.draw_route(result.route, color, width, alpha)
        
        # Draw UI elements
        self.draw_title_bar()
        self.draw_info_panel()
        
        # Update display
        pygame.display.flip()
    
    def run_visualization(self, target_fps: int = 60):
        """
        Run the interactive visualization loop.
        
        Args:
            target_fps: Target frames per second
        """
        print("🎮 Starting Four Algorithms Pygame Visualization...")
        print("   Controls:")
        print("   - Keys 1-4: Toggle individual algorithms")
        print("   - A: Toggle all algorithms")
        print("   - C: Clear all routes")
        print("   - S: Toggle performance stats")
        print("   - SPACE: Reset animation")
        print("   - ESC: Exit")
        
        running = True
        
        while running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif not self.handle_event(event):
                    running = False
            
            # Update animation
            self.animation_progress += 0.005
            if self.animation_progress > 1.0:
                self.animation_progress = 0.0
            
            # Render frame
            self.render()
            
            # Control frame rate
            self.clock.tick(target_fps)
        
        pygame.quit()
        print("🎮 Visualization closed.")

def create_four_algorithms_demo(num_cities: int = 25):
    """
    Create and run a demonstration of the 4 algorithms with visualization.
    
    Args:
        num_cities: Number of cities to include in the demo
    """
    print("🎯 Creating Four Algorithms Demo...")
    print("=" * 60)
    
    # Load data
    print(f"📊 Loading {num_cities} cities from Brazilian transportation network...")
    loader = load_transportation_data(sample_nodes=num_cities)
    node_ids = list(loader.graph.nodes())
    coordinates = [loader.get_node_coordinates(nid) for nid in node_ids]
    
    print(f"✅ Loaded {len(coordinates)} cities")
    
    # Calculate distance matrix
    calculator = DistanceCalculator(coordinates)
    distance_matrix = calculator.get_distance_matrix()
    
    # Configure algorithms for demo
    config = FocusedConfig(
        pso_population_size=min(50, num_cities * 2),
        pso_max_iterations=min(100, num_cities * 3),
        aco_num_ants=min(50, num_cities * 2),
        aco_max_iterations=min(100, num_cities * 3)
    )
    
    # Run algorithms
    print("\n🚀 Running the 4 focused algorithms...")
    results = run_four_focused_algorithms(distance_matrix, coordinates, config)
    
    # Create and run visualization
    print("\n🎮 Launching Pygame visualization...")
    visualizer = FourAlgorithmsTSPVisualizer()
    visualizer.set_data(coordinates)
    visualizer.add_algorithm_results(results)
    
    print("\n" + "="*60)
    print("🎊 FOUR ALGORITHMS DEMO READY!")
    print("Use keyboard controls to interact with the visualization.")
    print("="*60)
    
    # Run interactive visualization
    visualizer.run_visualization()

if __name__ == "__main__":
    # Run the four algorithms demo with pygame visualization
    create_four_algorithms_demo(num_cities=30)