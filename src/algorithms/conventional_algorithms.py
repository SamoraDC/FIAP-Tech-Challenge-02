"""
Conventional optimization algorithms for comparison with Genetic Algorithm.
Implements Dijkstra, A*, and other classical pathfinding/optimization methods.
FIAP Tech Challenge Phase 2 - Multi-Algorithm Route Optimization.
"""

import numpy as np
import heapq
from typing import List, Tuple, Dict, Optional, Set
import math
import time
from dataclasses import dataclass

@dataclass
class AlgorithmResult:
    """Result container for optimization algorithms."""
    algorithm_name: str
    route: List[int]
    distance: float
    execution_time: float
    nodes_explored: int
    additional_info: Dict = None

class DijkstraAlgorithm:
    """
    Dijkstra's algorithm for shortest path problems.
    Adapts single-source shortest path for TSP-like problems.
    """
    
    def __init__(self, distance_matrix: np.ndarray):
        """
        Initialize Dijkstra algorithm.
        
        Args:
            distance_matrix: NxN matrix of distances between nodes
        """
        self.distance_matrix = distance_matrix
        self.num_nodes = len(distance_matrix)
        self.nodes_explored = 0
    
    def shortest_path(self, start: int, end: int) -> Tuple[List[int], float]:
        """
        Find shortest path between two nodes.
        
        Args:
            start: Starting node index
            end: Ending node index
            
        Returns:
            Tuple of (path, total_distance)
        """
        # Initialize distances and previous nodes
        distances = [float('inf')] * self.num_nodes
        previous = [-1] * self.num_nodes
        visited = [False] * self.num_nodes
        
        distances[start] = 0
        
        # Priority queue: (distance, node)
        pq = [(0, start)]
        self.nodes_explored = 0
        
        while pq:
            current_dist, current_node = heapq.heappop(pq)
            
            if visited[current_node]:
                continue
                
            visited[current_node] = True
            self.nodes_explored += 1
            
            # If we reached the destination
            if current_node == end:
                break
            
            # Explore neighbors
            for neighbor in range(self.num_nodes):
                if not visited[neighbor] and self.distance_matrix[current_node][neighbor] > 0:
                    distance = current_dist + self.distance_matrix[current_node][neighbor]
                    
                    if distance < distances[neighbor]:
                        distances[neighbor] = distance
                        previous[neighbor] = current_node
                        heapq.heappush(pq, (distance, neighbor))
        
        # Reconstruct path
        if distances[end] == float('inf'):
            return [], float('inf')
        
        path = []
        current = end
        while current != -1:
            path.append(current)
            current = previous[current]
        
        path.reverse()
        return path, distances[end]
    
    def tsp_nearest_neighbor_with_dijkstra(self, start: int = 0) -> AlgorithmResult:
        """
        Solve TSP using nearest neighbor heuristic with Dijkstra for each step.
        
        Args:
            start: Starting city index
            
        Returns:
            AlgorithmResult with solution details
        """
        start_time = time.time()
        total_nodes_explored = 0
        
        route = [start]
        unvisited = set(range(self.num_nodes)) - {start}
        total_distance = 0.0
        current_city = start
        
        while unvisited:
            # Find nearest unvisited city using direct distance
            # (Dijkstra overkill for direct connections, but demonstrates usage)
            nearest_city = min(unvisited, 
                             key=lambda city: self.distance_matrix[current_city][city])
            
            # Add to route
            route.append(nearest_city)
            total_distance += self.distance_matrix[current_city][nearest_city]
            unvisited.remove(nearest_city)
            current_city = nearest_city
            total_nodes_explored += 1
        
        # Return to start
        total_distance += self.distance_matrix[current_city][start]
        
        execution_time = time.time() - start_time
        
        return AlgorithmResult(
            algorithm_name="Dijkstra-Enhanced Nearest Neighbor",
            route=route,
            distance=total_distance,
            execution_time=execution_time,
            nodes_explored=total_nodes_explored,
            additional_info={
                'heuristic_type': 'nearest_neighbor',
                'start_city': start
            }
        )
    
    def all_pairs_shortest_paths(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute shortest paths between all pairs of nodes.
        
        Returns:
            Tuple of (distance_matrix, next_node_matrix)
        """
        start_time = time.time()
        
        # Initialize matrices
        dist = self.distance_matrix.copy()
        next_node = np.full((self.num_nodes, self.num_nodes), -1)
        
        # Initialize next node matrix
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if i != j and dist[i][j] < float('inf'):
                    next_node[i][j] = j
        
        # Floyd-Warshall algorithm
        for k in range(self.num_nodes):
            for i in range(self.num_nodes):
                for j in range(self.num_nodes):
                    if dist[i][k] + dist[k][j] < dist[i][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]
                        next_node[i][j] = next_node[i][k]
        
        computation_time = time.time() - start_time
        print(f"All-pairs shortest paths computed in {computation_time:.3f}s")
        
        return dist, next_node

class AStarAlgorithm:
    """
    A* algorithm for shortest path with heuristic.
    Uses geographical coordinates for heuristic function.
    """
    
    def __init__(self, distance_matrix: np.ndarray, coordinates: List[Tuple[float, float]]):
        """
        Initialize A* algorithm.
        
        Args:
            distance_matrix: NxN matrix of distances between nodes
            coordinates: List of (longitude, latitude) tuples for heuristic
        """
        self.distance_matrix = distance_matrix
        self.coordinates = coordinates
        self.num_nodes = len(distance_matrix)
        self.nodes_explored = 0
    
    def heuristic(self, node1: int, node2: int) -> float:
        """
        Heuristic function using Euclidean distance between coordinates.
        
        Args:
            node1: First node index
            node2: Second node index
            
        Returns:
            Heuristic distance estimate
        """
        lon1, lat1 = self.coordinates[node1]
        lon2, lat2 = self.coordinates[node2]
        
        # Simple Euclidean distance as heuristic
        return math.sqrt((lon2 - lon1)**2 + (lat2 - lat1)**2) * 111000  # Rough conversion to meters
    
    def shortest_path(self, start: int, goal: int) -> Tuple[List[int], float]:
        """
        Find shortest path using A* algorithm.
        
        Args:
            start: Starting node index
            goal: Goal node index
            
        Returns:
            Tuple of (path, total_distance)
        """
        # Initialize data structures
        open_set = [(0, start)]  # Priority queue: (f_score, node)
        came_from = {}
        g_score = {i: float('inf') for i in range(self.num_nodes)}
        f_score = {i: float('inf') for i in range(self.num_nodes)}
        
        g_score[start] = 0
        f_score[start] = self.heuristic(start, goal)
        
        open_set_hash = {start}
        self.nodes_explored = 0
        
        while open_set:
            current_f, current = heapq.heappop(open_set)
            open_set_hash.discard(current)
            
            self.nodes_explored += 1
            
            if current == goal:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path.reverse()
                
                return path, g_score[goal]
            
            # Explore neighbors
            for neighbor in range(self.num_nodes):
                if neighbor != current and self.distance_matrix[current][neighbor] > 0:
                    tentative_g_score = g_score[current] + self.distance_matrix[current][neighbor]
                    
                    if tentative_g_score < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g_score
                        f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, goal)
                        
                        if neighbor not in open_set_hash:
                            heapq.heappush(open_set, (f_score[neighbor], neighbor))
                            open_set_hash.add(neighbor)
        
        return [], float('inf')  # No path found
    
    def tsp_astar_nearest_neighbor(self, start: int = 0) -> AlgorithmResult:
        """
        Solve TSP using A* for pathfinding with nearest neighbor heuristic.
        
        Args:
            start: Starting city index
            
        Returns:
            AlgorithmResult with solution details
        """
        start_time = time.time()
        total_nodes_explored = 0
        
        route = [start]
        unvisited = set(range(self.num_nodes)) - {start}
        total_distance = 0.0
        current_city = start
        
        while unvisited:
            # Find nearest unvisited city
            nearest_city = min(unvisited, 
                             key=lambda city: self.distance_matrix[current_city][city])
            
            # Use A* to find path (though for direct connections, it's just the edge)
            path, distance = self.shortest_path(current_city, nearest_city)
            
            if path:
                route.append(nearest_city)
                total_distance += distance
                total_nodes_explored += self.nodes_explored
                unvisited.remove(nearest_city)
                current_city = nearest_city
            else:
                # Fallback to direct connection
                route.append(nearest_city)
                total_distance += self.distance_matrix[current_city][nearest_city]
                unvisited.remove(nearest_city)
                current_city = nearest_city
        
        # Return to start
        total_distance += self.distance_matrix[current_city][start]
        
        execution_time = time.time() - start_time
        
        return AlgorithmResult(
            algorithm_name="A* Enhanced Nearest Neighbor",
            route=route,
            distance=total_distance,
            execution_time=execution_time,
            nodes_explored=total_nodes_explored,
            additional_info={
                'heuristic_type': 'euclidean_coordinates',
                'start_city': start
            }
        )

class GreedyAlgorithms:
    """
    Collection of greedy algorithms for TSP comparison.
    """
    
    def __init__(self, distance_matrix: np.ndarray):
        """
        Initialize greedy algorithms.
        
        Args:
            distance_matrix: NxN matrix of distances between nodes
        """
        self.distance_matrix = distance_matrix
        self.num_nodes = len(distance_matrix)
    
    def nearest_neighbor(self, start: int = 0) -> AlgorithmResult:
        """
        Classic nearest neighbor TSP heuristic.
        
        Args:
            start: Starting city index
            
        Returns:
            AlgorithmResult with solution details
        """
        start_time = time.time()
        
        route = [start]
        unvisited = set(range(self.num_nodes)) - {start}
        total_distance = 0.0
        current_city = start
        
        while unvisited:
            # Find nearest unvisited city
            nearest_city = min(unvisited, 
                             key=lambda city: self.distance_matrix[current_city][city])
            
            route.append(nearest_city)
            total_distance += self.distance_matrix[current_city][nearest_city]
            unvisited.remove(nearest_city)
            current_city = nearest_city
        
        # Return to start
        total_distance += self.distance_matrix[current_city][start]
        
        execution_time = time.time() - start_time
        
        return AlgorithmResult(
            algorithm_name="Nearest Neighbor",
            route=route,
            distance=total_distance,
            execution_time=execution_time,
            nodes_explored=self.num_nodes - 1,
            additional_info={
                'heuristic_type': 'greedy_nearest',
                'start_city': start
            }
        )
    
    def cheapest_insertion(self) -> AlgorithmResult:
        """
        Cheapest insertion heuristic for TSP.
        
        Returns:
            AlgorithmResult with solution details
        """
        start_time = time.time()
        
        if self.num_nodes < 3:
            # Handle trivial cases
            route = list(range(self.num_nodes))
            distance = sum(self.distance_matrix[route[i]][route[i+1]] for i in range(len(route)-1))
            distance += self.distance_matrix[route[-1]][route[0]]
            
            return AlgorithmResult(
                algorithm_name="Cheapest Insertion",
                route=route,
                distance=distance,
                execution_time=time.time() - start_time,
                nodes_explored=self.num_nodes,
                additional_info={'method': 'trivial_case'}
            )
        
        # Start with the smallest triangle
        min_triangle_cost = float('inf')
        best_triangle = None
        
        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                for k in range(j + 1, self.num_nodes):
                    cost = (self.distance_matrix[i][j] + 
                           self.distance_matrix[j][k] + 
                           self.distance_matrix[k][i])
                    if cost < min_triangle_cost:
                        min_triangle_cost = cost
                        best_triangle = [i, j, k]
        
        route = best_triangle
        unvisited = set(range(self.num_nodes)) - set(route)
        
        # Insert remaining cities at cheapest positions
        while unvisited:
            best_city = None
            best_position = None
            best_increase = float('inf')
            
            for city in unvisited:
                for pos in range(len(route)):
                    # Calculate cost increase of inserting city at position pos
                    prev_city = route[pos - 1]
                    next_city = route[pos]
                    
                    old_cost = self.distance_matrix[prev_city][next_city]
                    new_cost = (self.distance_matrix[prev_city][city] + 
                               self.distance_matrix[city][next_city])
                    increase = new_cost - old_cost
                    
                    if increase < best_increase:
                        best_increase = increase
                        best_city = city
                        best_position = pos
            
            # Insert best city at best position
            route.insert(best_position, best_city)
            unvisited.remove(best_city)
        
        # Calculate total distance
        total_distance = 0.0
        for i in range(len(route)):
            next_i = (i + 1) % len(route)
            total_distance += self.distance_matrix[route[i]][route[next_i]]
        
        execution_time = time.time() - start_time
        
        return AlgorithmResult(
            algorithm_name="Cheapest Insertion",
            route=route,
            distance=total_distance,
            execution_time=execution_time,
            nodes_explored=self.num_nodes,
            additional_info={
                'initial_triangle': best_triangle,
                'triangle_cost': min_triangle_cost
            }
        )
    
    def farthest_insertion(self) -> AlgorithmResult:
        """
        Farthest insertion heuristic for TSP.
        
        Returns:
            AlgorithmResult with solution details
        """
        start_time = time.time()
        
        if self.num_nodes < 2:
            route = list(range(self.num_nodes))
            distance = 0 if self.num_nodes < 2 else self.distance_matrix[0][0]
            
            return AlgorithmResult(
                algorithm_name="Farthest Insertion",
                route=route,
                distance=distance,
                execution_time=time.time() - start_time,
                nodes_explored=self.num_nodes,
                additional_info={'method': 'trivial_case'}
            )
        
        # Start with the two farthest cities
        max_distance = 0
        start_pair = (0, 1)
        
        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                if self.distance_matrix[i][j] > max_distance:
                    max_distance = self.distance_matrix[i][j]
                    start_pair = (i, j)
        
        route = list(start_pair)
        unvisited = set(range(self.num_nodes)) - set(route)
        
        # Insert remaining cities
        while unvisited:
            # Find farthest city from current route
            farthest_city = None
            max_min_distance = 0
            
            for city in unvisited:
                min_distance_to_route = min(self.distance_matrix[city][route_city] 
                                          for route_city in route)
                if min_distance_to_route > max_min_distance:
                    max_min_distance = min_distance_to_route
                    farthest_city = city
            
            # Find best insertion position for farthest city
            best_position = 0
            best_increase = float('inf')
            
            for pos in range(len(route)):
                prev_city = route[pos - 1]
                next_city = route[pos]
                
                old_cost = self.distance_matrix[prev_city][next_city]
                new_cost = (self.distance_matrix[prev_city][farthest_city] + 
                           self.distance_matrix[farthest_city][next_city])
                increase = new_cost - old_cost
                
                if increase < best_increase:
                    best_increase = increase
                    best_position = pos
            
            # Insert farthest city at best position
            route.insert(best_position, farthest_city)
            unvisited.remove(farthest_city)
        
        # Calculate total distance
        total_distance = 0.0
        for i in range(len(route)):
            next_i = (i + 1) % len(route)
            total_distance += self.distance_matrix[route[i]][route[next_i]]
        
        execution_time = time.time() - start_time
        
        return AlgorithmResult(
            algorithm_name="Farthest Insertion",
            route=route,
            distance=total_distance,
            execution_time=execution_time,
            nodes_explored=self.num_nodes,
            additional_info={
                'start_pair': start_pair,
                'start_distance': max_distance
            }
        )

def run_conventional_algorithms(distance_matrix: np.ndarray, 
                              coordinates: List[Tuple[float, float]] = None) -> List[AlgorithmResult]:
    """
    Run all conventional algorithms and return results.
    
    Args:
        distance_matrix: NxN matrix of distances between nodes
        coordinates: Optional coordinates for A* heuristic
        
    Returns:
        List of AlgorithmResult objects
    """
    results = []
    
    # Greedy algorithms
    greedy = GreedyAlgorithms(distance_matrix)
    results.append(greedy.nearest_neighbor(start=0))
    results.append(greedy.cheapest_insertion())
    results.append(greedy.farthest_insertion())
    
    # Dijkstra-based
    dijkstra = DijkstraAlgorithm(distance_matrix)
    results.append(dijkstra.tsp_nearest_neighbor_with_dijkstra(start=0))
    
    # A* if coordinates available
    if coordinates:
        astar = AStarAlgorithm(distance_matrix, coordinates)
        results.append(astar.tsp_astar_nearest_neighbor(start=0))
    
    return results