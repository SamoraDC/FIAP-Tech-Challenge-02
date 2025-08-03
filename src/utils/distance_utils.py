"""
Distance calculation utilities for the FIAP Tech Challenge Phase 2.
Provides various distance metrics for geographical coordinates and optimization.
"""

import numpy as np
import math
from typing import Tuple, List, Union

def euclidean_distance(coord1: Tuple[float, float], coord2: Tuple[float, float]) -> float:
    """
    Calculate Euclidean distance between two coordinates.
    
    Args:
        coord1: (longitude, latitude) of first point
        coord2: (longitude, latitude) of second point
        
    Returns:
        Euclidean distance (suitable for small geographical areas)
    """
    lon1, lat1 = coord1
    lon2, lat2 = coord2
    
    return math.sqrt((lon2 - lon1)**2 + (lat2 - lat1)**2)

def haversine_distance(coord1: Tuple[float, float], coord2: Tuple[float, float]) -> float:
    """
    Calculate Haversine distance between two coordinates.
    More accurate for geographical distances on Earth's surface.
    
    Args:
        coord1: (longitude, latitude) of first point
        coord2: (longitude, latitude) of second point
        
    Returns:
        Distance in meters
    """
    lon1, lat1 = coord1
    lon2, lat2 = coord2
    
    # Earth's radius in meters
    R = 6371000
    
    # Convert latitude and longitude from degrees to radians
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)
    
    # Haversine formula
    a = (math.sin(delta_lat / 2) * math.sin(delta_lat / 2) +
         math.cos(lat1_rad) * math.cos(lat2_rad) *
         math.sin(delta_lon / 2) * math.sin(delta_lon / 2))
    
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    return R * c

def manhattan_distance(coord1: Tuple[float, float], coord2: Tuple[float, float]) -> float:
    """
    Calculate Manhattan distance between two coordinates.
    
    Args:
        coord1: (longitude, latitude) of first point
        coord2: (longitude, latitude) of second point
        
    Returns:
        Manhattan distance
    """
    lon1, lat1 = coord1
    lon2, lat2 = coord2
    
    return abs(lon2 - lon1) + abs(lat2 - lat1)

def calculate_distance_matrix(coordinates: List[Tuple[float, float]], 
                            method: str = "haversine") -> np.ndarray:
    """
    Calculate distance matrix for a list of coordinates.
    
    Args:
        coordinates: List of (longitude, latitude) tuples
        method: Distance calculation method ("euclidean", "haversine", "manhattan")
        
    Returns:
        Symmetric distance matrix
    """
    n = len(coordinates)
    distance_matrix = np.zeros((n, n))
    
    # Choose distance function
    if method == "euclidean":
        dist_func = euclidean_distance
    elif method == "haversine":
        dist_func = haversine_distance
    elif method == "manhattan":
        dist_func = manhattan_distance
    else:
        raise ValueError(f"Unknown distance method: {method}")
    
    # Calculate upper triangle of matrix (symmetric)
    for i in range(n):
        for j in range(i + 1, n):
            distance = dist_func(coordinates[i], coordinates[j])
            distance_matrix[i][j] = distance
            distance_matrix[j][i] = distance  # Symmetric
    
    return distance_matrix

def validate_coordinates(coordinates: List[Tuple[float, float]]) -> bool:
    """
    Validate that coordinates are reasonable for Brazilian locations.
    
    Args:
        coordinates: List of (longitude, latitude) tuples
        
    Returns:
        True if coordinates appear valid
    """
    for lon, lat in coordinates:
        # Brazil longitude range: approximately -74 to -34 degrees
        # Brazil latitude range: approximately -34 to 5 degrees
        if not (-75 <= lon <= -30):
            return False
        if not (-35 <= lat <= 6):
            return False
    
    return True

def calculate_route_distance(route: List[int], distance_matrix: np.ndarray) -> float:
    """
    Calculate total distance for a route given a distance matrix.
    
    Args:
        route: List of node indices representing the route
        distance_matrix: Precomputed distance matrix
        
    Returns:
        Total route distance
    """
    if len(route) < 2:
        return 0.0
    
    total_distance = 0.0
    for i in range(len(route) - 1):
        total_distance += distance_matrix[route[i]][route[i + 1]]
    
    # Add return to start for TSP (closed loop)
    total_distance += distance_matrix[route[-1]][route[0]]
    
    return total_distance

def calculate_route_distance_coords(route_coords: List[Tuple[float, float]], 
                                  method: str = "haversine",
                                  closed_loop: bool = True) -> float:
    """
    Calculate total distance for a route given coordinates.
    
    Args:
        route_coords: List of (longitude, latitude) tuples in route order
        method: Distance calculation method
        closed_loop: Whether to add return trip to start
        
    Returns:
        Total route distance
    """
    if len(route_coords) < 2:
        return 0.0
    
    # Choose distance function
    if method == "euclidean":
        dist_func = euclidean_distance
    elif method == "haversine":
        dist_func = haversine_distance
    elif method == "manhattan":
        dist_func = manhattan_distance
    else:
        raise ValueError(f"Unknown distance method: {method}")
    
    total_distance = 0.0
    
    # Calculate distance between consecutive points
    for i in range(len(route_coords) - 1):
        total_distance += dist_func(route_coords[i], route_coords[i + 1])
    
    # Add return to start if closed loop (TSP)
    if closed_loop:
        total_distance += dist_func(route_coords[-1], route_coords[0])
    
    return total_distance

def get_nearest_neighbors(target_coord: Tuple[float, float], 
                         coordinates: List[Tuple[float, float]], 
                         k: int = 5,
                         method: str = "haversine") -> List[Tuple[int, float]]:
    """
    Find k nearest neighbors to a target coordinate.
    
    Args:
        target_coord: Target (longitude, latitude)
        coordinates: List of available coordinates
        k: Number of nearest neighbors to return
        method: Distance calculation method
        
    Returns:
        List of (index, distance) tuples for k nearest neighbors
    """
    # Choose distance function
    if method == "euclidean":
        dist_func = euclidean_distance
    elif method == "haversine":
        dist_func = haversine_distance
    elif method == "manhattan":
        dist_func = manhattan_distance
    else:
        raise ValueError(f"Unknown distance method: {method}")
    
    # Calculate distances to all coordinates
    distances = []
    for i, coord in enumerate(coordinates):
        distance = dist_func(target_coord, coord)
        distances.append((i, distance))
    
    # Sort by distance and return k nearest
    distances.sort(key=lambda x: x[1])
    return distances[:k]

class DistanceCalculator:
    """
    Centralized distance calculator for optimization algorithms.
    Caches distance matrices for efficient repeated calculations.
    """
    
    def __init__(self, coordinates: List[Tuple[float, float]], method: str = "haversine"):
        """
        Initialize calculator with coordinates.
        
        Args:
            coordinates: List of (longitude, latitude) tuples
            method: Distance calculation method
        """
        self.coordinates = coordinates
        self.method = method
        self.distance_matrix = None
        self._validate_inputs()
        
    def _validate_inputs(self):
        """Validate input coordinates."""
        if not self.coordinates:
            raise ValueError("Coordinates list cannot be empty")
        
        if not validate_coordinates(self.coordinates):
            print("Warning: Some coordinates may be outside Brazil bounds")
    
    def get_distance_matrix(self, force_recalculate: bool = False) -> np.ndarray:
        """
        Get or calculate distance matrix.
        
        Args:
            force_recalculate: Force recalculation even if matrix exists
            
        Returns:
            Distance matrix
        """
        if self.distance_matrix is None or force_recalculate:
            print(f"Calculating distance matrix using {self.method} method...")
            self.distance_matrix = calculate_distance_matrix(self.coordinates, self.method)
            print(f"Distance matrix calculated: {self.distance_matrix.shape}")
        
        return self.distance_matrix
    
    def get_distance(self, idx1: int, idx2: int) -> float:
        """
        Get distance between two coordinate indices.
        
        Args:
            idx1: First coordinate index
            idx2: Second coordinate index
            
        Returns:
            Distance between coordinates
        """
        if self.distance_matrix is None:
            self.get_distance_matrix()
        
        return self.distance_matrix[idx1][idx2]
    
    def calculate_route_distance(self, route: List[int], closed_loop: bool = True) -> float:
        """
        Calculate total distance for a route.
        
        Args:
            route: List of coordinate indices representing the route
            closed_loop: Whether to add return trip to start
            
        Returns:
            Total route distance
        """
        if self.distance_matrix is None:
            self.get_distance_matrix()
        
        if closed_loop:
            return calculate_route_distance(route, self.distance_matrix)
        else:
            # Open route (no return to start)
            if len(route) < 2:
                return 0.0
            
            total_distance = 0.0
            for i in range(len(route) - 1):
                total_distance += self.distance_matrix[route[i]][route[i + 1]]
            
            return total_distance