"""
Metaheuristic algorithms for TSP optimization.
Implements Particle Swarm Optimization (PSO) and Ant Colony Optimization (ACO).
FIAP Tech Challenge Phase 2 - Multi-Algorithm Route Optimization.
"""

import numpy as np
import random
from typing import List, Tuple, Dict, Optional
import time
from dataclasses import dataclass
import copy

@dataclass
class MetaheuristicConfig:
    """Configuration for metaheuristic algorithms."""
    # PSO parameters
    population_size: int = 50
    max_iterations: int = 100
    w: float = 0.9  # Inertia weight
    c1: float = 2.0  # Cognitive parameter
    c2: float = 2.0  # Social parameter
    
    # ACO parameters
    num_ants: int = 50
    alpha: float = 1.0  # Pheromone importance
    beta: float = 2.0   # Heuristic importance
    rho: float = 0.1    # Evaporation rate
    q: float = 100.0    # Pheromone constant

class Particle:
    """Particle for PSO algorithm representing a TSP route."""
    
    def __init__(self, num_cities: int):
        """
        Initialize particle with random route.
        
        Args:
            num_cities: Number of cities in TSP
        """
        self.num_cities = num_cities
        self.position = list(range(num_cities))
        random.shuffle(self.position)
        
        self.velocity = []
        self.fitness = float('inf')
        self.best_position = self.position.copy()
        self.best_fitness = float('inf')
    
    def update_velocity(self, global_best_position: List[int], w: float, c1: float, c2: float):
        """
        Update particle velocity based on PSO equation.
        For TSP, velocity is represented as a sequence of swap operations.
        """
        # Generate cognitive component (toward personal best)
        cognitive_swaps = self._generate_swaps(self.position, self.best_position)
        
        # Generate social component (toward global best)
        social_swaps = self._generate_swaps(self.position, global_best_position)
        
        # Combine components with PSO weights
        self.velocity = []
        
        # Add inertia component (keep some previous swaps)
        if hasattr(self, '_previous_swaps'):
            for swap in self._previous_swaps:
                if random.random() < w:
                    self.velocity.append(swap)
        
        # Add cognitive component
        for swap in cognitive_swaps:
            if random.random() < c1 * random.random():
                self.velocity.append(swap)
        
        # Add social component
        for swap in social_swaps:
            if random.random() < c2 * random.random():
                self.velocity.append(swap)
        
        # Limit velocity size to prevent excessive changes
        max_swaps = max(1, self.num_cities // 4)
        if len(self.velocity) > max_swaps:
            self.velocity = random.sample(self.velocity, max_swaps)
        
        self._previous_swaps = self.velocity.copy()
    
    def _generate_swaps(self, current: List[int], target: List[int]) -> List[Tuple[int, int]]:
        """Generate sequence of swaps to transform current position toward target."""
        swaps = []
        temp = current.copy()
        
        for i in range(len(temp)):
            if temp[i] != target[i]:
                # Find where target[i] is in temp
                j = temp.index(target[i])
                if i != j:
                    swaps.append((i, j))
                    # Perform the swap
                    temp[i], temp[j] = temp[j], temp[i]
        
        return swaps
    
    def update_position(self):
        """Update particle position by applying velocity (swaps)."""
        for i, j in self.velocity:
            if 0 <= i < len(self.position) and 0 <= j < len(self.position):
                self.position[i], self.position[j] = self.position[j], self.position[i]
    
    def update_best(self):
        """Update personal best if current position is better."""
        if self.fitness < self.best_fitness:
            self.best_fitness = self.fitness
            self.best_position = self.position.copy()

class ParticleSwarmOptimization:
    """
    Particle Swarm Optimization for TSP.
    Adapts continuous PSO concepts to discrete TSP problem.
    """
    
    def __init__(self, distance_matrix: np.ndarray, config: MetaheuristicConfig = None):
        """
        Initialize PSO algorithm.
        
        Args:
            distance_matrix: NxN matrix of distances between cities
            config: Algorithm configuration parameters
        """
        self.distance_matrix = distance_matrix
        self.num_cities = len(distance_matrix)
        self.config = config or MetaheuristicConfig()
        
        self.swarm: List[Particle] = []
        self.global_best_position: List[int] = []
        self.global_best_fitness = float('inf')
        
        self.history = {
            'best_fitness': [],
            'average_fitness': [],
            'diversity': []
        }
    
    def calculate_fitness(self, route: List[int]) -> float:
        """Calculate fitness (total route distance) for a given route."""
        total_distance = 0.0
        
        for i in range(len(route)):
            next_i = (i + 1) % len(route)
            total_distance += self.distance_matrix[route[i]][route[next_i]]
        
        return total_distance
    
    def initialize_swarm(self):
        """Initialize swarm with random particles."""
        self.swarm = []
        
        for _ in range(self.config.population_size):
            particle = Particle(self.num_cities)
            particle.fitness = self.calculate_fitness(particle.position)
            particle.update_best()
            
            # Update global best
            if particle.fitness < self.global_best_fitness:
                self.global_best_fitness = particle.fitness
                self.global_best_position = particle.position.copy()
            
            self.swarm.append(particle)
    
    def calculate_diversity(self) -> float:
        """Calculate swarm diversity."""
        if len(self.swarm) < 2:
            return 0.0
        
        total_distance = 0
        comparisons = 0
        
        for i in range(len(self.swarm)):
            for j in range(i + 1, len(self.swarm)):
                # Calculate Hamming distance between routes
                differences = sum(1 for a, b in zip(self.swarm[i].position, self.swarm[j].position) if a != b)
                total_distance += differences
                comparisons += 1
        
        return total_distance / (comparisons * self.num_cities) if comparisons > 0 else 0.0
    
    def run(self, verbose: bool = True) -> Tuple[List[int], float]:
        """
        Run PSO algorithm.
        
        Args:
            verbose: Whether to print progress
            
        Returns:
            Tuple of (best_route, best_distance)
        """
        if verbose:
            print(f"Starting PSO for {self.num_cities}-city TSP")
            print(f"Swarm size: {self.config.population_size}, Iterations: {self.config.max_iterations}")
        
        start_time = time.time()
        
        # Initialize swarm
        self.initialize_swarm()
        
        if verbose:
            print(f"Initial best fitness: {self.global_best_fitness:.1f}")
        
        # Main PSO loop
        for iteration in range(self.config.max_iterations):
            # Update each particle
            for particle in self.swarm:
                # Update velocity
                particle.update_velocity(
                    self.global_best_position,
                    self.config.w,
                    self.config.c1,
                    self.config.c2
                )
                
                # Update position
                particle.update_position()
                
                # Evaluate fitness
                particle.fitness = self.calculate_fitness(particle.position)
                
                # Update personal best
                particle.update_best()
                
                # Update global best
                if particle.fitness < self.global_best_fitness:
                    self.global_best_fitness = particle.fitness
                    self.global_best_position = particle.position.copy()
            
            # Record statistics
            avg_fitness = sum(p.fitness for p in self.swarm) / len(self.swarm)
            diversity = self.calculate_diversity()
            
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
        
        return self.global_best_position, self.global_best_fitness

class AntColonyOptimization:
    """
    Ant Colony Optimization for TSP.
    Implements classical ACO with pheromone trails and heuristic information.
    """
    
    def __init__(self, distance_matrix: np.ndarray, config: MetaheuristicConfig = None):
        """
        Initialize ACO algorithm.
        
        Args:
            distance_matrix: NxN matrix of distances between cities
            config: Algorithm configuration parameters
        """
        self.distance_matrix = distance_matrix
        self.num_cities = len(distance_matrix)
        self.config = config or MetaheuristicConfig()
        
        # Initialize pheromone matrix
        self.pheromones = np.ones((self.num_cities, self.num_cities)) * 0.1
        
        # Calculate heuristic information (inverse of distance)
        self.heuristics = np.zeros((self.num_cities, self.num_cities))
        for i in range(self.num_cities):
            for j in range(self.num_cities):
                if i != j and self.distance_matrix[i][j] > 0:
                    self.heuristics[i][j] = 1.0 / self.distance_matrix[i][j]
        
        self.best_route = []
        self.best_distance = float('inf')
        
        self.history = {
            'best_distances': [],
            'average_distances': []
        }
    
    def calculate_route_distance(self, route: List[int]) -> float:
        """Calculate total distance for a route."""
        total_distance = 0.0
        
        for i in range(len(route)):
            next_i = (i + 1) % len(route)
            total_distance += self.distance_matrix[route[i]][route[next_i]]
        
        return total_distance
    
    def construct_ant_solution(self, start_city: int = None) -> List[int]:
        """
        Construct a solution for one ant using probabilistic selection.
        
        Args:
            start_city: Starting city (random if None)
            
        Returns:
            Complete route for the ant
        """
        if start_city is None:
            start_city = random.randint(0, self.num_cities - 1)
        
        route = [start_city]
        unvisited = set(range(self.num_cities)) - {start_city}
        current_city = start_city
        
        while unvisited:
            # Calculate probabilities for next city selection
            probabilities = []
            total_prob = 0.0
            
            for next_city in unvisited:
                pheromone = self.pheromones[current_city][next_city] ** self.config.alpha
                heuristic = self.heuristics[current_city][next_city] ** self.config.beta
                prob = pheromone * heuristic
                probabilities.append((next_city, prob))
                total_prob += prob
            
            # Normalize probabilities
            if total_prob > 0:
                probabilities = [(city, prob / total_prob) for city, prob in probabilities]
            else:
                # Fallback to uniform distribution
                prob = 1.0 / len(unvisited)
                probabilities = [(city, prob) for city, _ in probabilities]
            
            # Select next city using roulette wheel selection
            r = random.random()
            cumulative_prob = 0.0
            next_city = None
            
            for city, prob in probabilities:
                cumulative_prob += prob
                if r <= cumulative_prob:
                    next_city = city
                    break
            
            # Fallback if selection failed
            if next_city is None:
                next_city = random.choice(list(unvisited))
            
            route.append(next_city)
            unvisited.remove(next_city)
            current_city = next_city
        
        return route
    
    def update_pheromones(self, ant_routes: List[List[int]], ant_distances: List[float]):
        """
        Update pheromone trails based on ant solutions.
        
        Args:
            ant_routes: List of routes constructed by ants
            ant_distances: Corresponding distances for each route
        """
        # Evaporation
        self.pheromones *= (1 - self.config.rho)
        
        # Pheromone deposition
        for route, distance in zip(ant_routes, ant_distances):
            if distance > 0:
                pheromone_deposit = self.config.q / distance
                
                for i in range(len(route)):
                    next_i = (i + 1) % len(route)
                    city1, city2 = route[i], route[next_i]
                    
                    # Add pheromone to both directions (symmetric TSP)
                    self.pheromones[city1][city2] += pheromone_deposit
                    self.pheromones[city2][city1] += pheromone_deposit
        
        # Prevent pheromone values from becoming too small
        self.pheromones = np.maximum(self.pheromones, 0.01)
        
        # Prevent pheromone values from becoming too large
        self.pheromones = np.minimum(self.pheromones, 10.0)
    
    def run(self, verbose: bool = True) -> Tuple[List[int], float]:
        """
        Run ACO algorithm.
        
        Args:
            verbose: Whether to print progress
            
        Returns:
            Tuple of (best_route, best_distance)
        """
        if verbose:
            print(f"Starting ACO for {self.num_cities}-city TSP")
            print(f"Number of ants: {self.config.num_ants}, Iterations: {self.config.max_iterations}")
            print(f"Alpha: {self.config.alpha}, Beta: {self.config.beta}, Rho: {self.config.rho}")
        
        start_time = time.time()
        
        for iteration in range(self.config.max_iterations):
            # Construct solutions for all ants
            ant_routes = []
            ant_distances = []
            
            for _ in range(self.config.num_ants):
                route = self.construct_ant_solution()
                distance = self.calculate_route_distance(route)
                
                ant_routes.append(route)
                ant_distances.append(distance)
                
                # Update best solution
                if distance < self.best_distance:
                    self.best_distance = distance
                    self.best_route = route.copy()
            
            # Update pheromones
            self.update_pheromones(ant_routes, ant_distances)
            
            # Record statistics
            avg_distance = sum(ant_distances) / len(ant_distances)
            self.history['best_distances'].append(self.best_distance)
            self.history['average_distances'].append(avg_distance)
            
            if verbose and (iteration + 1) % 20 == 0:
                print(f"Iteration {iteration + 1}: Best = {self.best_distance:.1f}, "
                      f"Avg = {avg_distance:.1f}")
        
        execution_time = time.time() - start_time
        
        if verbose:
            print(f"ACO completed in {execution_time:.2f} seconds")
            print(f"Best route distance: {self.best_distance:.1f}")
        
        return self.best_route, self.best_distance

def run_metaheuristic_algorithms(distance_matrix: np.ndarray, 
                                config: MetaheuristicConfig = None) -> Dict:
    """
    Run all metaheuristic algorithms and return results.
    
    Args:
        distance_matrix: NxN matrix of distances between cities
        config: Algorithm configuration parameters
        
    Returns:
        Dictionary containing results from PSO and ACO
    """
    results = {}
    
    if config is None:
        config = MetaheuristicConfig()
    
    # Run PSO
    print("Running Particle Swarm Optimization...")
    pso = ParticleSwarmOptimization(distance_matrix, config)
    pso_route, pso_distance = pso.run(verbose=True)
    
    results['PSO'] = {
        'algorithm': 'Particle Swarm Optimization',
        'route': pso_route,
        'distance': pso_distance,
        'history': pso.history
    }
    
    print("\n" + "="*50 + "\n")
    
    # Run ACO
    print("Running Ant Colony Optimization...")
    aco = AntColonyOptimization(distance_matrix, config)
    aco_route, aco_distance = aco.run(verbose=True)
    
    results['ACO'] = {
        'algorithm': 'Ant Colony Optimization',
        'route': aco_route,
        'distance': aco_distance,
        'history': aco.history
    }
    
    return results