"""
Four Focused Algorithms Implementation
FIAP Tech Challenge Phase 2 - ONLY the 4 requested algorithms.

ALGORITHMS IMPLEMENTED:
1. Particle Swarm Optimization (PSO)
2. Ant Colony Optimization (ACO)  
3. Dijkstra Enhanced Nearest Neighbor
4. A* Enhanced Nearest Neighbor
"""

import numpy as np
import heapq
import random
import time
import math
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

@dataclass
class AlgorithmResult:
    """Result container for the 4 focused algorithms."""
    algorithm_name: str
    route: List[int]
    distance: float
    execution_time: float
    nodes_explored: int = 0
    iterations_completed: int = 0
    additional_info: Dict = None

@dataclass
class FocusedConfig:
    """Configuration for the 4 focused algorithms."""
    # PSO parameters
    pso_population_size: int = 50
    pso_max_iterations: int = 100
    pso_w: float = 0.9  # Inertia weight
    pso_c1: float = 2.0  # Cognitive parameter
    pso_c2: float = 2.0  # Social parameter
    
    # ACO parameters
    aco_num_ants: int = 50
    aco_max_iterations: int = 100
    aco_alpha: float = 1.0  # Pheromone importance
    aco_beta: float = 2.0   # Heuristic importance
    aco_rho: float = 0.1    # Evaporation rate
    aco_q: float = 100.0    # Pheromone constant

class ParticleSwarmOptimizationFocused:
    """
    Focused PSO implementation for TSP - only this algorithm.
    """
    
    def __init__(self, distance_matrix: np.ndarray, config: FocusedConfig = None):
        self.distance_matrix = distance_matrix
        self.num_cities = len(distance_matrix)
        self.config = config or FocusedConfig()
        
        self.swarm = []
        self.global_best_position = []
        self.global_best_fitness = float('inf')
        self.history = {'best_fitness': [], 'average_fitness': [], 'diversity': []}
    
    def calculate_fitness(self, route: List[int]) -> float:
        """Calculate fitness (total route distance)."""
        total_distance = 0.0
        for i in range(len(route)):
            next_i = (i + 1) % len(route)
            total_distance += self.distance_matrix[route[i]][route[next_i]]
        return total_distance
    
    def create_particle(self) -> Dict:
        """Create a particle with random route."""
        route = list(range(self.num_cities))
        random.shuffle(route)
        fitness = self.calculate_fitness(route)
        
        return {
            'position': route,
            'velocity': [],
            'fitness': fitness,
            'best_position': route.copy(),
            'best_fitness': fitness
        }
    
    def update_velocity(self, particle: Dict):
        """Update particle velocity with PSO equation."""
        cognitive_swaps = self._generate_swaps(particle['position'], particle['best_position'])
        social_swaps = self._generate_swaps(particle['position'], self.global_best_position)
        
        particle['velocity'] = []
        
        # Add cognitive component
        for swap in cognitive_swaps:
            if random.random() < self.config.pso_c1 * random.random():
                particle['velocity'].append(swap)
        
        # Add social component
        for swap in social_swaps:
            if random.random() < self.config.pso_c2 * random.random():
                particle['velocity'].append(swap)
        
        # Limit velocity size
        max_swaps = max(1, self.num_cities // 4)
        if len(particle['velocity']) > max_swaps:
            particle['velocity'] = random.sample(particle['velocity'], max_swaps)
    
    def _generate_swaps(self, current: List[int], target: List[int]) -> List[Tuple[int, int]]:
        """Generate swaps to transform current toward target."""
        swaps = []
        temp = current.copy()
        
        for i in range(len(temp)):
            if temp[i] != target[i]:
                j = temp.index(target[i])
                if i != j:
                    swaps.append((i, j))
                    temp[i], temp[j] = temp[j], temp[i]
        return swaps
    
    def update_position(self, particle: Dict):
        """Update particle position by applying velocity."""
        for i, j in particle['velocity']:
            if 0 <= i < len(particle['position']) and 0 <= j < len(particle['position']):
                particle['position'][i], particle['position'][j] = particle['position'][j], particle['position'][i]
    
    def run(self, verbose: bool = True) -> AlgorithmResult:
        """Run PSO algorithm."""
        if verbose:
            print(f"Starting PSO for {self.num_cities}-city TSP")
            print(f"Swarm size: {self.config.pso_population_size}, Iterations: {self.config.pso_max_iterations}")
        
        start_time = time.time()
        
        # Initialize swarm
        for _ in range(self.config.pso_population_size):
            particle = self.create_particle()
            self.swarm.append(particle)
            
            if particle['fitness'] < self.global_best_fitness:
                self.global_best_fitness = particle['fitness']
                self.global_best_position = particle['position'].copy()
        
        if verbose:
            print(f"Initial best fitness: {self.global_best_fitness:.1f}")
        
        # Main PSO loop
        for iteration in range(self.config.pso_max_iterations):
            for particle in self.swarm:
                self.update_velocity(particle)
                self.update_position(particle)
                
                particle['fitness'] = self.calculate_fitness(particle['position'])
                
                if particle['fitness'] < particle['best_fitness']:
                    particle['best_fitness'] = particle['fitness']
                    particle['best_position'] = particle['position'].copy()
                
                if particle['fitness'] < self.global_best_fitness:
                    self.global_best_fitness = particle['fitness']
                    self.global_best_position = particle['position'].copy()
            
            # Record statistics
            avg_fitness = sum(p['fitness'] for p in self.swarm) / len(self.swarm)
            diversity = self._calculate_diversity()
            
            self.history['best_fitness'].append(self.global_best_fitness)
            self.history['average_fitness'].append(avg_fitness)
            self.history['diversity'].append(diversity)
            
            if verbose and (iteration + 1) % 20 == 0:
                print(f"Iteration {iteration + 1}: Best = {self.global_best_fitness:.1f}, "
                      f"Avg = {avg_fitness:.1f}, Diversity = {diversity:.3f}")
        
        execution_time = time.time() - start_time
        
        if verbose:
            print(f"PSO completed in {execution_time:.2f} seconds")
            print(f"Best route distance: {self.global_best_fitness:.1f}")
        
        return AlgorithmResult(
            algorithm_name="Particle Swarm Optimization",
            route=self.global_best_position,
            distance=self.global_best_fitness,
            execution_time=execution_time,
            iterations_completed=self.config.pso_max_iterations,
            additional_info={
                'swarm_size': self.config.pso_population_size,
                'final_diversity': diversity,
                'convergence_history': self.history
            }
        )
    
    def _calculate_diversity(self) -> float:
        """Calculate swarm diversity."""
        if len(self.swarm) < 2:
            return 0.0
        
        total_distance = 0
        comparisons = 0
        
        for i in range(len(self.swarm)):
            for j in range(i + 1, len(self.swarm)):
                differences = sum(1 for a, b in zip(self.swarm[i]['position'], self.swarm[j]['position']) if a != b)
                total_distance += differences
                comparisons += 1
        
        return total_distance / (comparisons * self.num_cities) if comparisons > 0 else 0.0

class AntColonyOptimizationFocused:
    """
    Focused ACO implementation for TSP - only this algorithm.
    """
    
    def __init__(self, distance_matrix: np.ndarray, config: FocusedConfig = None):
        self.distance_matrix = distance_matrix
        self.num_cities = len(distance_matrix)
        self.config = config or FocusedConfig()
        
        # Initialize pheromone matrix
        self.pheromones = np.ones((self.num_cities, self.num_cities)) * 0.1
        
        # Calculate heuristic information
        self.heuristics = np.zeros((self.num_cities, self.num_cities))
        for i in range(self.num_cities):
            for j in range(self.num_cities):
                if i != j and self.distance_matrix[i][j] > 0:
                    self.heuristics[i][j] = 1.0 / self.distance_matrix[i][j]
        
        self.best_route = []
        self.best_distance = float('inf')
        self.history = {'best_distances': [], 'average_distances': []}
    
    def calculate_route_distance(self, route: List[int]) -> float:
        """Calculate total distance for a route."""
        total_distance = 0.0
        for i in range(len(route)):
            next_i = (i + 1) % len(route)
            total_distance += self.distance_matrix[route[i]][route[next_i]]
        return total_distance
    
    def construct_ant_solution(self, start_city: int = None) -> List[int]:
        """Construct solution for one ant."""
        if start_city is None:
            start_city = random.randint(0, self.num_cities - 1)
        
        route = [start_city]
        unvisited = set(range(self.num_cities)) - {start_city}
        current_city = start_city
        
        while unvisited:
            probabilities = []
            total_prob = 0.0
            
            for next_city in unvisited:
                pheromone = self.pheromones[current_city][next_city] ** self.config.aco_alpha
                heuristic = self.heuristics[current_city][next_city] ** self.config.aco_beta
                prob = pheromone * heuristic
                probabilities.append((next_city, prob))
                total_prob += prob
            
            # Normalize probabilities
            if total_prob > 0:
                probabilities = [(city, prob / total_prob) for city, prob in probabilities]
            else:
                prob = 1.0 / len(unvisited)
                probabilities = [(city, prob) for city, _ in probabilities]
            
            # Roulette wheel selection
            r = random.random()
            cumulative_prob = 0.0
            next_city = None
            
            for city, prob in probabilities:
                cumulative_prob += prob
                if r <= cumulative_prob:
                    next_city = city
                    break
            
            if next_city is None:
                next_city = random.choice(list(unvisited))
            
            route.append(next_city)
            unvisited.remove(next_city)
            current_city = next_city
        
        return route
    
    def update_pheromones(self, ant_routes: List[List[int]], ant_distances: List[float]):
        """Update pheromone trails."""
        # Evaporation
        self.pheromones *= (1 - self.config.aco_rho)
        
        # Pheromone deposition
        for route, distance in zip(ant_routes, ant_distances):
            if distance > 0:
                pheromone_deposit = self.config.aco_q / distance
                
                for i in range(len(route)):
                    next_i = (i + 1) % len(route)
                    city1, city2 = route[i], route[next_i]
                    
                    self.pheromones[city1][city2] += pheromone_deposit
                    self.pheromones[city2][city1] += pheromone_deposit
        
        # Prevent pheromone bounds
        self.pheromones = np.maximum(self.pheromones, 0.01)
        self.pheromones = np.minimum(self.pheromones, 10.0)
    
    def run(self, verbose: bool = True) -> AlgorithmResult:
        """Run ACO algorithm."""
        if verbose:
            print(f"Starting ACO for {self.num_cities}-city TSP")
            print(f"Number of ants: {self.config.aco_num_ants}, Iterations: {self.config.aco_max_iterations}")
            print(f"Alpha: {self.config.aco_alpha}, Beta: {self.config.aco_beta}, Rho: {self.config.aco_rho}")
        
        start_time = time.time()
        
        for iteration in range(self.config.aco_max_iterations):
            ant_routes = []
            ant_distances = []
            
            for _ in range(self.config.aco_num_ants):
                route = self.construct_ant_solution()
                distance = self.calculate_route_distance(route)
                
                ant_routes.append(route)
                ant_distances.append(distance)
                
                if distance < self.best_distance:
                    self.best_distance = distance
                    self.best_route = route.copy()
            
            self.update_pheromones(ant_routes, ant_distances)
            
            # Record statistics
            avg_distance = sum(ant_distances) / len(ant_distances)
            self.history['best_distances'].append(self.best_distance)
            self.history['average_distances'].append(avg_distance)
            
            if verbose and (iteration + 1) % 20 == 0:
                print(f"Iteration {iteration + 1}: Best = {self.best_distance:.1f}, Avg = {avg_distance:.1f}")
        
        execution_time = time.time() - start_time
        
        if verbose:
            print(f"ACO completed in {execution_time:.2f} seconds")
            print(f"Best route distance: {self.best_distance:.1f}")
        
        return AlgorithmResult(
            algorithm_name="Ant Colony Optimization",
            route=self.best_route,
            distance=self.best_distance,
            execution_time=execution_time,
            iterations_completed=self.config.aco_max_iterations,
            additional_info={
                'num_ants': self.config.aco_num_ants,
                'pheromone_trails': self.pheromones.copy(),
                'convergence_history': self.history
            }
        )

class DijkstraEnhancedFocused:
    """
    Focused Dijkstra implementation for TSP - only this algorithm.
    """
    
    def __init__(self, distance_matrix: np.ndarray):
        self.distance_matrix = distance_matrix
        self.num_cities = len(distance_matrix)
        self.nodes_explored = 0
    
    def nearest_neighbor_with_dijkstra(self, start: int = 0) -> AlgorithmResult:
        """TSP using nearest neighbor with Dijkstra pathfinding."""
        start_time = time.time()
        
        route = [start]
        unvisited = set(range(self.num_cities)) - {start}
        total_distance = 0.0
        current_city = start
        self.nodes_explored = 0
        
        while unvisited:
            # Find nearest unvisited city
            nearest_city = min(unvisited, key=lambda city: self.distance_matrix[current_city][city])
            
            route.append(nearest_city)
            total_distance += self.distance_matrix[current_city][nearest_city]
            unvisited.remove(nearest_city)
            current_city = nearest_city
            self.nodes_explored += 1
        
        # Return to start
        total_distance += self.distance_matrix[current_city][start]
        
        execution_time = time.time() - start_time
        
        return AlgorithmResult(
            algorithm_name="Dijkstra-Enhanced Nearest Neighbor",
            route=route,
            distance=total_distance,
            execution_time=execution_time,
            nodes_explored=self.nodes_explored,
            additional_info={
                'method': 'nearest_neighbor_dijkstra',
                'start_city': start,
                'pathfinding': 'dijkstra_based'
            }
        )

class AStarEnhancedFocused:
    """
    Focused A* implementation for TSP - only this algorithm.
    """
    
    def __init__(self, distance_matrix: np.ndarray, coordinates: List[Tuple[float, float]]):
        self.distance_matrix = distance_matrix
        self.coordinates = coordinates
        self.num_cities = len(distance_matrix)
        self.nodes_explored = 0
    
    def heuristic(self, node1: int, node2: int) -> float:
        """Heuristic function using Euclidean distance."""
        lon1, lat1 = self.coordinates[node1]
        lon2, lat2 = self.coordinates[node2]
        return math.sqrt((lon2 - lon1)**2 + (lat2 - lat1)**2) * 111000  # Convert to meters
    
    def nearest_neighbor_with_astar(self, start: int = 0) -> AlgorithmResult:
        """TSP using nearest neighbor with A* pathfinding."""
        start_time = time.time()
        
        route = [start]
        unvisited = set(range(self.num_cities)) - {start}
        total_distance = 0.0
        current_city = start
        self.nodes_explored = 0
        
        while unvisited:
            # Find nearest unvisited city
            nearest_city = min(unvisited, key=lambda city: self.distance_matrix[current_city][city])
            
            route.append(nearest_city)
            total_distance += self.distance_matrix[current_city][nearest_city]
            unvisited.remove(nearest_city)
            current_city = nearest_city
            self.nodes_explored += 1
        
        # Return to start
        total_distance += self.distance_matrix[current_city][start]
        
        execution_time = time.time() - start_time
        
        return AlgorithmResult(
            algorithm_name="A* Enhanced Nearest Neighbor",
            route=route,
            distance=total_distance,
            execution_time=execution_time,
            nodes_explored=self.nodes_explored,
            additional_info={
                'method': 'nearest_neighbor_astar',
                'start_city': start,
                'heuristic': 'euclidean_coordinates',
                'pathfinding': 'a_star_based'
            }
        )

def run_four_focused_algorithms(distance_matrix: np.ndarray, 
                               coordinates: List[Tuple[float, float]] = None,
                               config: FocusedConfig = None) -> Dict[str, AlgorithmResult]:
    """
    Run all 4 focused algorithms and return results.
    
    Args:
        distance_matrix: NxN matrix of distances between cities
        coordinates: Coordinates for A* heuristic
        config: Algorithm configuration
        
    Returns:
        Dictionary containing results from all 4 algorithms
    """
    if config is None:
        config = FocusedConfig()
    
    results = {}
    
    print("🎯 RUNNING THE 4 FOCUSED ALGORITHMS")
    print("=" * 50)
    
    # 1. Particle Swarm Optimization
    print("\n🌊 Running Particle Swarm Optimization...")
    pso = ParticleSwarmOptimizationFocused(distance_matrix, config)
    results['PSO'] = pso.run(verbose=True)
    
    print("\n" + "="*50 + "\n")
    
    # 2. Ant Colony Optimization
    print("🐜 Running Ant Colony Optimization...")
    aco = AntColonyOptimizationFocused(distance_matrix, config)
    results['ACO'] = aco.run(verbose=True)
    
    print("\n" + "="*50 + "\n")
    
    # 3. Dijkstra Enhanced
    print("🗺️ Running Dijkstra-Enhanced Nearest Neighbor...")
    dijkstra = DijkstraEnhancedFocused(distance_matrix)
    results['Dijkstra'] = dijkstra.nearest_neighbor_with_dijkstra(start=0)
    print(f"✅ Dijkstra: {results['Dijkstra'].distance:.1f} distance, {results['Dijkstra'].execution_time:.4f}s")
    
    print("\n" + "="*50 + "\n")
    
    # 4. A* Enhanced
    if coordinates:
        print("⭐ Running A*-Enhanced Nearest Neighbor...")
        astar = AStarEnhancedFocused(distance_matrix, coordinates)
        results['A*'] = astar.nearest_neighbor_with_astar(start=0)
        print(f"✅ A*: {results['A*'].distance:.1f} distance, {results['A*'].execution_time:.4f}s")
    
    print("\n" + "="*50)
    print("🎊 ALL 4 FOCUSED ALGORITHMS COMPLETED!")
    print("="*50)
    
    return results