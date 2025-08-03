"""
TSP Visualization System using Pygame
FIAP Tech Challenge Phase 2 - Interactive route optimization visualization.
"""

import pygame
import sys
import math
from typing import List, Tuple, Dict, Optional
import numpy as np
from dataclasses import dataclass

# Color constants
COLORS = {
    'background': (240, 248, 255),  # Alice blue
    'city': (30, 144, 255),         # Dodger blue
    'city_highlight': (255, 69, 0), # Red orange
    'route_ga': (255, 20, 147),     # Deep pink
    'route_conventional': (50, 205, 50),  # Lime green
    'route_meta': (255, 165, 0),    # Orange
    'text': (25, 25, 112),          # Midnight blue
    'panel': (245, 245, 245),       # White smoke
    'button': (70, 130, 180),       # Steel blue
    'button_hover': (100, 149, 237), # Cornflower blue
    'grid': (220, 220, 220),        # Light gray
}

@dataclass
class VisualizationConfig:
    """Configuration for the TSP visualizer."""
    window_width: int = 1200
    window_height: int = 800
    map_margin: int = 50
    panel_width: int = 300
    city_radius: int = 6
    route_width: int = 3
    font_size: int = 16
    title_font_size: int = 24

class TSPVisualizer:
    """
    Main visualization class for TSP algorithms.
    Displays Brazilian cities, routes, and algorithm performance.
    """
    
    def __init__(self, config: VisualizationConfig = None):
        """
        Initialize the TSP visualizer.
        
        Args:
            config: Visualization configuration
        """
        self.config = config or VisualizationConfig()
        
        # Initialize Pygame
        pygame.init()
        self.screen = pygame.display.set_mode((self.config.window_width, self.config.window_height))
        pygame.display.set_caption("FIAP Tech Challenge - TSP Algorithm Visualization")
        
        # Fonts
        self.font = pygame.font.Font(None, self.config.font_size)
        self.title_font = pygame.font.Font(None, self.config.title_font_size)
        
        # Data storage
        self.cities: List[Tuple[float, float]] = []
        self.city_names: List[str] = []
        self.routes: Dict[str, List[int]] = {}
        self.algorithm_results: Dict[str, Dict] = {}
        self.selected_algorithms: List[str] = []
        
        # Coordinate transformation
        self.min_lon = self.max_lon = 0
        self.min_lat = self.max_lat = 0
        self.map_width = 0
        self.map_height = 0
        
        # UI elements
        self.buttons: List[Dict] = []
        self.current_algorithm = ""
        self.show_all_routes = False
        
        # Animation
        self.clock = pygame.time.Clock()
        self.animation_progress = 0.0
        self.animation_speed = 0.02
        
    def set_data(self, coordinates: List[Tuple[float, float]], 
                 city_names: List[str] = None):
        """
        Set the city data for visualization.
        
        Args:
            coordinates: List of (longitude, latitude) tuples
            city_names: Optional list of city names
        """
        self.cities = coordinates
        self.city_names = city_names or [f"City {i}" for i in range(len(coordinates))]
        
        # Calculate coordinate bounds for transformation
        if coordinates:
            lons, lats = zip(*coordinates)
            self.min_lon, self.max_lon = min(lons), max(lons)
            self.min_lat, self.max_lat = min(lats), max(lats)
            
            # Calculate map dimensions
            self.map_width = self.config.window_width - self.config.panel_width - 2 * self.config.map_margin
            self.map_height = self.config.window_height - 2 * self.config.map_margin
    
    def add_algorithm_result(self, algorithm_name: str, route: List[int], 
                           distance: float, time: float, 
                           additional_info: Dict = None):
        """
        Add algorithm result for visualization.
        
        Args:
            algorithm_name: Name of the algorithm
            route: Route as list of city indices
            distance: Total route distance
            time: Execution time
            additional_info: Additional algorithm information
        """
        self.routes[algorithm_name] = route
        result_data = {
            'distance': distance,
            'time': time,
            'route': route
        }
        if additional_info:
            result_data.update(additional_info)
        self.algorithm_results[algorithm_name] = result_data
        
        if algorithm_name not in self.selected_algorithms:
            self.selected_algorithms.append(algorithm_name)
    
    def lonlat_to_screen(self, lon: float, lat: float) -> Tuple[int, int]:
        """
        Convert longitude/latitude to screen coordinates.
        
        Args:
            lon: Longitude
            lat: Latitude
            
        Returns:
            Screen coordinates (x, y)
        """
        if self.max_lon == self.min_lon or self.max_lat == self.min_lat:
            return (self.config.map_margin, self.config.map_margin)
        
        # Normalize coordinates
        x_norm = (lon - self.min_lon) / (self.max_lon - self.min_lon)
        y_norm = (lat - self.min_lat) / (self.max_lat - self.min_lat)
        
        # Convert to screen coordinates (flip Y axis)
        x = self.config.map_margin + x_norm * self.map_width
        y = self.config.map_margin + (1 - y_norm) * self.map_height
        
        return (int(x), int(y))
    
    def draw_coordinate_grid(self):
        """Draw coordinate grid and labels."""
        if not self.cities:
            return
        
        # Draw grid lines
        num_lines = 5
        
        for i in range(num_lines + 1):
            # Vertical lines
            x = self.config.map_margin + i * self.map_width / num_lines
            pygame.draw.line(self.screen, COLORS['grid'], 
                           (x, self.config.map_margin), 
                           (x, self.config.map_margin + self.map_height), 1)
            
            # Horizontal lines
            y = self.config.map_margin + i * self.map_height / num_lines
            pygame.draw.line(self.screen, COLORS['grid'], 
                           (self.config.map_margin, y), 
                           (self.config.map_margin + self.map_width, y), 1)
        
        # Draw coordinate labels
        for i in range(num_lines + 1):
            # Longitude labels
            lon = self.min_lon + i * (self.max_lon - self.min_lon) / num_lines
            x = self.config.map_margin + i * self.map_width / num_lines
            text = self.font.render(f"{lon:.2f}°", True, COLORS['text'])
            self.screen.blit(text, (x - text.get_width()//2, 
                                  self.config.map_margin + self.map_height + 5))
            
            # Latitude labels
            lat = self.min_lat + i * (self.max_lat - self.min_lat) / num_lines
            y = self.config.map_margin + (num_lines - i) * self.map_height / num_lines
            text = self.font.render(f"{lat:.2f}°", True, COLORS['text'])
            self.screen.blit(text, (5, y - text.get_height()//2))
    
    def draw_cities(self):
        """Draw cities as circles on the map."""
        for i, (lon, lat) in enumerate(self.cities):
            x, y = self.lonlat_to_screen(lon, lat)
            
            # Draw city circle
            pygame.draw.circle(self.screen, COLORS['city'], (x, y), self.config.city_radius)
            pygame.draw.circle(self.screen, COLORS['text'], (x, y), self.config.city_radius, 2)
            
            # Draw city index
            text = self.font.render(str(i), True, COLORS['text'])
            text_rect = text.get_rect(center=(x, y - self.config.city_radius - 15))
            self.screen.blit(text, text_rect)
    
    def draw_route(self, route: List[int], color: Tuple[int, int, int], 
                   width: int = None, alpha: float = 1.0):
        """
        Draw a TSP route on the map.
        
        Args:
            route: List of city indices
            color: Route color
            width: Line width
            alpha: Transparency (0.0 to 1.0)
        """
        if len(route) < 2:
            return
        
        width = width or self.config.route_width
        
        # Create a surface for alpha blending
        if alpha < 1.0:
            route_surface = pygame.Surface((self.config.window_width, self.config.window_height), pygame.SRCALPHA)
            surface = route_surface
        else:
            surface = self.screen
        
        # Draw route segments
        points = []
        for city_idx in route:
            if city_idx < len(self.cities):
                lon, lat = self.cities[city_idx]
                x, y = self.lonlat_to_screen(lon, lat)
                points.append((x, y))
        
        # Draw lines between consecutive cities
        for i in range(len(points)):
            start = points[i]
            end = points[(i + 1) % len(points)]  # Connect back to start
            pygame.draw.line(surface, color, start, end, width)
        
        # Draw route direction arrows
        self.draw_route_arrows(points, color, surface)
        
        # Blit alpha surface if needed
        if alpha < 1.0:
            surface.set_alpha(int(255 * alpha))
            self.screen.blit(surface, (0, 0))
    
    def draw_route_arrows(self, points: List[Tuple[int, int]], 
                         color: Tuple[int, int, int], surface):
        """Draw directional arrows on route segments."""
        arrow_size = 8
        
        for i in range(len(points)):
            start = points[i]
            end = points[(i + 1) % len(points)]
            
            # Calculate arrow position (middle of segment)
            mid_x = (start[0] + end[0]) // 2
            mid_y = (start[1] + end[1]) // 2
            
            # Calculate arrow direction
            dx = end[0] - start[0]
            dy = end[1] - start[1]
            length = math.sqrt(dx*dx + dy*dy)
            
            if length > 0:
                # Normalize direction
                dx /= length
                dy /= length
                
                # Arrow points
                arrow_points = [
                    (mid_x + dx * arrow_size, mid_y + dy * arrow_size),
                    (mid_x - dx * arrow_size//2 - dy * arrow_size//2, 
                     mid_y - dy * arrow_size//2 + dx * arrow_size//2),
                    (mid_x - dx * arrow_size//2 + dy * arrow_size//2, 
                     mid_y - dy * arrow_size//2 - dx * arrow_size//2)
                ]
                
                pygame.draw.polygon(surface, color, arrow_points)
    
    def draw_info_panel(self):
        """Draw the information panel with algorithm results."""
        panel_x = self.config.window_width - self.config.panel_width
        panel_rect = pygame.Rect(panel_x, 0, self.config.panel_width, self.config.window_height)
        pygame.draw.rect(self.screen, COLORS['panel'], panel_rect)
        pygame.draw.line(self.screen, COLORS['text'], 
                        (panel_x, 0), (panel_x, self.config.window_height), 2)
        
        y_offset = 20
        
        # Title
        title = self.title_font.render("Algorithm Results", True, COLORS['text'])
        self.screen.blit(title, (panel_x + 10, y_offset))
        y_offset += 40
        
        # Algorithm results
        if self.algorithm_results:
            for i, (algo_name, result) in enumerate(self.algorithm_results.items()):
                # Algorithm name
                name_color = self.get_algorithm_color(algo_name)
                algo_text = self.font.render(algo_name[:25], True, name_color)
                self.screen.blit(algo_text, (panel_x + 10, y_offset))
                y_offset += 25
                
                # Distance
                dist_text = f"Distance: {result['distance']/1000:.2f} km"
                dist_surface = self.font.render(dist_text, True, COLORS['text'])
                self.screen.blit(dist_surface, (panel_x + 15, y_offset))
                y_offset += 20
                
                # Time
                time_text = f"Time: {result['time']:.4f} s"
                time_surface = self.font.render(time_text, True, COLORS['text'])
                self.screen.blit(time_surface, (panel_x + 15, y_offset))
                y_offset += 30
        
        # Instructions
        y_offset += 20
        instructions = [
            "Controls:",
            "1-9: Show algorithm route",
            "A: Show all routes",
            "C: Clear routes",
            "ESC: Exit"
        ]
        
        for instruction in instructions:
            text = self.font.render(instruction, True, COLORS['text'])
            self.screen.blit(text, (panel_x + 10, y_offset))
            y_offset += 20
    
    def get_algorithm_color(self, algorithm_name: str) -> Tuple[int, int, int]:
        """Get color for an algorithm based on its category."""
        if "Genetic" in algorithm_name:
            return COLORS['route_ga']
        elif any(word in algorithm_name for word in ["PSO", "Particle", "ACO", "Ant"]):
            return COLORS['route_meta']
        else:
            return COLORS['route_conventional']
    
    def draw_title_bar(self):
        """Draw the main title bar."""
        title = "FIAP Tech Challenge - TSP Algorithm Visualization"
        title_surface = self.title_font.render(title, True, COLORS['text'])
        title_rect = title_surface.get_rect(center=(self.config.window_width // 2, 25))
        self.screen.blit(title_surface, title_rect)
        
        # Draw subtitle with city count
        if self.cities:
            subtitle = f"Brazilian Transportation Network - {len(self.cities)} Cities"
            subtitle_surface = self.font.render(subtitle, True, COLORS['text'])
            subtitle_rect = subtitle_surface.get_rect(center=(self.config.window_width // 2, 50))
            self.screen.blit(subtitle_surface, subtitle_rect)
    
    def handle_event(self, event):
        """
        Handle pygame events.
        
        Args:
            event: Pygame event
            
        Returns:
            True if should continue, False if should quit
        """
        if event.type == pygame.QUIT:
            return False
        
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                return False
            
            elif event.key == pygame.K_c:
                # Clear all routes
                self.current_algorithm = ""
                self.show_all_routes = False
            
            elif event.key == pygame.K_a:
                # Show all routes
                self.show_all_routes = not self.show_all_routes
                self.current_algorithm = ""
            
            elif event.key >= pygame.K_1 and event.key <= pygame.K_9:
                # Show specific algorithm (1-9)
                algo_index = event.key - pygame.K_1
                if algo_index < len(self.selected_algorithms):
                    self.current_algorithm = self.selected_algorithms[algo_index]
                    self.show_all_routes = False
        
        return True
    
    def render(self):
        """Render the complete visualization."""
        # Clear screen
        self.screen.fill(COLORS['background'])
        
        # Draw coordinate grid
        self.draw_coordinate_grid()
        
        # Draw routes
        if self.show_all_routes:
            # Draw all routes with transparency
            for i, (algo_name, route) in enumerate(self.routes.items()):
                color = self.get_algorithm_color(algo_name)
                alpha = 0.6
                self.draw_route(route, color, alpha=alpha)
        
        elif self.current_algorithm and self.current_algorithm in self.routes:
            # Draw selected route
            route = self.routes[self.current_algorithm]
            color = self.get_algorithm_color(self.current_algorithm)
            self.draw_route(route, color)
        
        # Draw cities (always on top)
        self.draw_cities()
        
        # Draw UI elements
        self.draw_title_bar()
        self.draw_info_panel()
        
        # Update display
        pygame.display.flip()
    
    def run_visualization(self, target_fps: int = 60):
        """
        Run the main visualization loop.
        
        Args:
            target_fps: Target frames per second
        """
        running = True
        
        print("🎮 TSP Visualization Started!")
        print("Controls:")
        print("  1-9: Show specific algorithm route")
        print("  A: Toggle all routes")
        print("  C: Clear routes")
        print("  ESC: Exit")
        
        while running:
            # Handle events
            for event in pygame.event.get():
                if not self.handle_event(event):
                    running = False
                    break
            
            # Render frame
            self.render()
            
            # Control frame rate
            self.clock.tick(target_fps)
        
        # Cleanup
        pygame.quit()
        print("👋 Visualization ended")

def create_demo_visualization():
    """Create a demonstration of the TSP visualizer."""
    # This function will be called from a test script
    # It sets up sample data and runs the visualizer
    pass

if __name__ == "__main__":
    # Basic test
    visualizer = TSPVisualizer()
    
    # Add some sample Brazilian cities (Brasília region)
    sample_cities = [
        (-47.8825, -15.7942),  # Brasília center
        (-47.9297, -15.7801),  # Brasília west
        (-47.8519, -15.8200),  # Brasília south
        (-47.8100, -15.7500),  # Brasília east
        (-47.9000, -15.7600),  # Brasília northwest
    ]
    
    visualizer.set_data(sample_cities)
    
    # Add a sample route
    sample_route = [0, 1, 2, 3, 4]
    visualizer.add_algorithm_result("Sample Route", sample_route, 50000, 0.1)
    
    # Run visualization
    visualizer.run_visualization()