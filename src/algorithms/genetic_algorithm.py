"""
Genetic Algorithm implementation for Traveling Salesman Problem (TSP).
FIAP Tech Challenge Phase 2 - Multi-Algorithm Route Optimization.
"""

import numpy as np
import random
from typing import List, Tuple, Dict, Optional, Callable
from dataclasses import dataclass
import copy
import time

@dataclass
class GeneticConfig:
    """Configuration parameters for the Genetic Algorithm."""
    population_size: int = 100
    elite_size: int = 20
    mutation_rate: float = 0.01
    generations: int = 500
    tournament_size: int = 5
    crossover_rate: float = 0.8
    selection_method: str = "tournament"  # "tournament", "roulette", "rank"
    crossover_method: str = "order"  # "order", "cycle", "pmx"
    mutation_method: str = "swap"  # "swap", "insert", "invert", "scramble"

class Individual:
    """
    Represents a single solution (route) in the genetic algorithm.
    Chromosome is represented as a permutation of city indices.
    """
    
    def __init__(self, route: List[int]):
        """
        Initialize individual with a route.
        
        Args:
            route: List of city indices representing the TSP route
        """
        self.route = route.copy()
        self.fitness: Optional[float] = None
        self.distance: Optional[float] = None
        
    def __len__(self) -> int:
        return len(self.route)
    
    def __str__(self) -> str:
        return f"Route: {self.route[:5]}...{self.route[-2:]} | Distance: {self.distance:.1f} | Fitness: {self.fitness:.6f}"
    
    def copy(self) -> 'Individual':
        """Create a deep copy of the individual."""
        new_individual = Individual(self.route)
        new_individual.fitness = self.fitness
        new_individual.distance = self.distance
        return new_individual

class GeneticAlgorithm:
    """
    Genetic Algorithm for solving the Traveling Salesman Problem.
    
    Implements various selection, crossover, and mutation operators
    with configurable parameters for optimization performance.
    """
    
    def __init__(self, distance_matrix: np.ndarray, config: GeneticConfig = None):
        """
        Initialize the Genetic Algorithm.
        
        Args:
            distance_matrix: NxN matrix of distances between cities
            config: Configuration parameters for the algorithm
        """
        self.distance_matrix = distance_matrix
        self.num_cities = len(distance_matrix)
        self.config = config or GeneticConfig()
        
        # Algorithm state
        self.population: List[Individual] = []
        self.best_individual: Optional[Individual] = None
        self.generation = 0
        self.history: Dict = {
            'best_distances': [],
            'average_distances': [],
            'diversity_scores': [],
            'generation_times': []
        }
        
        # Validate inputs
        self._validate_inputs()
        
    def _validate_inputs(self):
        """Validate algorithm inputs."""
        if self.distance_matrix.shape[0] != self.distance_matrix.shape[1]:
            raise ValueError("Distance matrix must be square")
        
        if not np.allclose(self.distance_matrix, self.distance_matrix.T):
            raise ValueError("Distance matrix must be symmetric")
        
        if np.any(self.distance_matrix < 0):
            raise ValueError("Distance matrix cannot contain negative values")
        
        if self.num_cities < 3:
            raise ValueError("TSP requires at least 3 cities")
    
    def calculate_distance(self, route: List[int]) -> float:
        """
        Calculate total distance for a given route.
        
        Args:
            route: List of city indices
            
        Returns:
            Total distance including return to start
        """
        total_distance = 0.0
        
        # Calculate distance between consecutive cities
        for i in range(len(route) - 1):
            total_distance += self.distance_matrix[route[i]][route[i + 1]]
        
        # Add return to start city (TSP closed loop)
        total_distance += self.distance_matrix[route[-1]][route[0]]
        
        return total_distance
    
    def calculate_fitness(self, individual: Individual) -> float:
        """
        Calculate fitness for an individual (inverse of distance).
        
        Args:
            individual: Individual to evaluate
            
        Returns:
            Fitness value (higher is better)
        """
        if individual.distance is None:
            individual.distance = self.calculate_distance(individual.route)
        
        # Fitness is inverse of distance (shorter routes have higher fitness)
        individual.fitness = 1.0 / (1.0 + individual.distance)
        return individual.fitness
    
    def create_random_individual(self) -> Individual:
        """Create a random individual with a valid TSP route."""
        route = list(range(self.num_cities))
        random.shuffle(route)
        return Individual(route)
    
    def initialize_population(self) -> List[Individual]:
        """
        Create initial population with diverse routes.
        Combines random initialization with some heuristic seeding.
        """
        population = []
        
        # Create random individuals
        for _ in range(self.config.population_size - 2):
            individual = self.create_random_individual()
            self.calculate_fitness(individual)
            population.append(individual)
        
        # Add nearest neighbor heuristic solution
        nn_route = self._nearest_neighbor_route()
        nn_individual = Individual(nn_route)
        self.calculate_fitness(nn_individual)
        population.append(nn_individual)
        
        # Add sorted route as baseline
        sorted_route = list(range(self.num_cities))
        sorted_individual = Individual(sorted_route)
        self.calculate_fitness(sorted_individual)
        population.append(sorted_individual)
        
        self.population = population
        return population
    
    def _nearest_neighbor_route(self, start_city: int = 0) -> List[int]:
        """
        Generate route using nearest neighbor heuristic.
        
        Args:
            start_city: Starting city index
            
        Returns:
            Route generated by nearest neighbor heuristic
        """
        route = [start_city]
        unvisited = set(range(self.num_cities)) - {start_city}
        current_city = start_city
        
        while unvisited:
            # Find nearest unvisited city
            nearest_city = min(unvisited, 
                             key=lambda city: self.distance_matrix[current_city][city])
            route.append(nearest_city)
            unvisited.remove(nearest_city)
            current_city = nearest_city
        
        return route
    
    # Selection Methods
    def tournament_selection(self, tournament_size: int = None) -> Individual:
        """Tournament selection."""
        size = tournament_size or self.config.tournament_size
        tournament = random.sample(self.population, size)
        return max(tournament, key=lambda ind: ind.fitness)
    
    def roulette_wheel_selection(self) -> Individual:
        """Roulette wheel selection based on fitness."""
        total_fitness = sum(ind.fitness for ind in self.population)
        selection_point = random.uniform(0, total_fitness)
        
        current_sum = 0
        for individual in self.population:
            current_sum += individual.fitness
            if current_sum >= selection_point:
                return individual
        
        return self.population[-1]  # Fallback
    
    def rank_selection(self) -> Individual:
        """Rank-based selection."""
        # Sort population by fitness
        sorted_pop = sorted(self.population, key=lambda ind: ind.fitness)
        
        # Assign ranks (1 to population_size)
        total_rank = sum(range(1, len(sorted_pop) + 1))
        selection_point = random.uniform(0, total_rank)
        
        current_sum = 0
        for i, individual in enumerate(sorted_pop):
            current_sum += i + 1  # Rank starts from 1
            if current_sum >= selection_point:
                return individual
        
        return sorted_pop[-1]  # Fallback
    
    def select_parent(self) -> Individual:
        """Select a parent based on configured selection method."""
        if self.config.selection_method == "tournament":
            return self.tournament_selection()
        elif self.config.selection_method == "roulette":
            return self.roulette_wheel_selection()
        elif self.config.selection_method == "rank":
            return self.rank_selection()
        else:
            raise ValueError(f"Unknown selection method: {self.config.selection_method}")
    
    # Crossover Methods
    def order_crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """
        Order Crossover (OX) - preserves order and adjacency.
        """
        size = len(parent1.route)
        
        # Choose two random crossover points
        start, end = sorted(random.sample(range(size), 2))
        
        # Create offspring
        offspring1_route = [-1] * size
        offspring2_route = [-1] * size
        
        # Copy segments from parents
        offspring1_route[start:end] = parent1.route[start:end]
        offspring2_route[start:end] = parent2.route[start:end]
        
        # Fill remaining positions maintaining order
        self._fill_order_crossover(offspring1_route, parent2.route, start, end)
        self._fill_order_crossover(offspring2_route, parent1.route, start, end)
        
        return Individual(offspring1_route), Individual(offspring2_route)
    
    def _fill_order_crossover(self, offspring: List[int], parent: List[int], start: int, end: int):
        """Helper method for order crossover."""
        size = len(offspring)
        remaining = [city for city in parent if city not in offspring[start:end]]
        
        pos = end
        for city in remaining:
            if pos >= size:
                pos = 0
            while offspring[pos] != -1:
                pos = (pos + 1) % size
            offspring[pos] = city
    
    def cycle_crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """
        Cycle Crossover (CX) - preserves absolute positions.
        """
        size = len(parent1.route)
        offspring1_route = [-1] * size
        offspring2_route = [-1] * size
        
        visited = [False] * size
        
        for start_idx in range(size):
            if visited[start_idx]:
                continue
                
            # Follow cycle
            idx = start_idx
            cycle_elements = []
            
            while not visited[idx]:
                visited[idx] = True
                cycle_elements.append(idx)
                
                # Find where parent1[idx] appears in parent2
                value = parent1.route[idx]
                idx = parent2.route.index(value)
            
            # Assign cycle elements alternately
            for i, pos in enumerate(cycle_elements):
                if len(cycle_elements) % 2 == 1:  # Odd cycle
                    offspring1_route[pos] = parent1.route[pos]
                    offspring2_route[pos] = parent2.route[pos]
                else:  # Even cycle
                    offspring1_route[pos] = parent2.route[pos]
                    offspring2_route[pos] = parent1.route[pos]
        
        return Individual(offspring1_route), Individual(offspring2_route)
    
    def pmx_crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """
        Partially Matched Crossover (PMX).
        """
        size = len(parent1.route)
        start, end = sorted(random.sample(range(size), 2))
        
        offspring1_route = parent1.route.copy()
        offspring2_route = parent2.route.copy()
        
        # Create mapping
        mapping1 = {}
        mapping2 = {}
        
        for i in range(start, end):
            mapping1[parent2.route[i]] = parent1.route[i]
            mapping2[parent1.route[i]] = parent2.route[i]
        
        # Apply crossover
        offspring1_route[start:end] = parent2.route[start:end]
        offspring2_route[start:end] = parent1.route[start:end]
        
        # Fix conflicts outside crossover region
        self._fix_pmx_conflicts(offspring1_route, mapping1, start, end)
        self._fix_pmx_conflicts(offspring2_route, mapping2, start, end)
        
        return Individual(offspring1_route), Individual(offspring2_route)
    
    def _fix_pmx_conflicts(self, offspring: List[int], mapping: Dict[int, int], start: int, end: int):
        """Helper method for PMX crossover."""
        for i in range(len(offspring)):
            if i < start or i >= end:
                while offspring[i] in mapping:
                    offspring[i] = mapping[offspring[i]]
    
    def crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """Perform crossover based on configured method."""
        if self.config.crossover_method == "order":
            return self.order_crossover(parent1, parent2)
        elif self.config.crossover_method == "cycle":
            return self.cycle_crossover(parent1, parent2)
        elif self.config.crossover_method == "pmx":
            return self.pmx_crossover(parent1, parent2)
        else:
            raise ValueError(f"Unknown crossover method: {self.config.crossover_method}")
    
    # Mutation Methods
    def swap_mutation(self, individual: Individual):
        """Swap two random cities in the route."""
        route = individual.route
        i, j = random.sample(range(len(route)), 2)
        route[i], route[j] = route[j], route[i]
    
    def insert_mutation(self, individual: Individual):
        """Remove a city and insert it at a random position."""
        route = individual.route
        i = random.randint(0, len(route) - 1)
        j = random.randint(0, len(route) - 1)
        
        city = route.pop(i)
        route.insert(j, city)
    
    def invert_mutation(self, individual: Individual):
        """Invert a random subsequence of the route."""
        route = individual.route
        i, j = sorted(random.sample(range(len(route)), 2))
        route[i:j+1] = reversed(route[i:j+1])
    
    def scramble_mutation(self, individual: Individual):
        """Scramble a random subsequence of the route."""
        route = individual.route
        i, j = sorted(random.sample(range(len(route)), 2))
        subsequence = route[i:j+1]
        random.shuffle(subsequence)
        route[i:j+1] = subsequence
    
    def mutate(self, individual: Individual):
        """Apply mutation based on configured method."""
        if random.random() < self.config.mutation_rate:
            if self.config.mutation_method == "swap":
                self.swap_mutation(individual)
            elif self.config.mutation_method == "insert":
                self.insert_mutation(individual)
            elif self.config.mutation_method == "invert":
                self.invert_mutation(individual)
            elif self.config.mutation_method == "scramble":
                self.scramble_mutation(individual)
            else:
                raise ValueError(f"Unknown mutation method: {self.config.mutation_method}")
            
            # Recalculate fitness after mutation
            individual.fitness = None
            individual.distance = None
    
    def evolve_generation(self):
        """Evolve the population for one generation."""
        generation_start = time.time()
        
        # Sort population by fitness
        self.population.sort(key=lambda ind: ind.fitness, reverse=True)
        
        # Update best individual
        if self.best_individual is None or self.population[0].fitness > self.best_individual.fitness:
            self.best_individual = self.population[0].copy()
        
        # Create new generation
        new_population = []
        
        # Elitism: keep best individuals
        for i in range(self.config.elite_size):
            new_population.append(self.population[i].copy())
        
        # Generate offspring
        while len(new_population) < self.config.population_size:
            parent1 = self.select_parent()
            parent2 = self.select_parent()
            
            if random.random() < self.config.crossover_rate:
                offspring1, offspring2 = self.crossover(parent1, parent2)
            else:
                offspring1, offspring2 = parent1.copy(), parent2.copy()
            
            # Apply mutation
            self.mutate(offspring1)
            self.mutate(offspring2)
            
            # Calculate fitness for new individuals
            self.calculate_fitness(offspring1)
            self.calculate_fitness(offspring2)
            
            new_population.extend([offspring1, offspring2])
        
        # Trim to exact population size
        self.population = new_population[:self.config.population_size]
        
        # Record statistics
        generation_time = time.time() - generation_start
        self._record_generation_stats(generation_time)
        self.generation += 1
    
    def _record_generation_stats(self, generation_time: float):
        """Record statistics for the current generation."""
        distances = [ind.distance for ind in self.population if ind.distance is not None]
        
        self.history['best_distances'].append(min(distances))
        self.history['average_distances'].append(sum(distances) / len(distances))
        self.history['generation_times'].append(generation_time)
        
        # Calculate diversity (average distance between routes)
        diversity = self._calculate_diversity()
        self.history['diversity_scores'].append(diversity)
    
    def _calculate_diversity(self) -> float:
        """Calculate population diversity."""
        if len(self.population) < 2:
            return 0.0
        
        total_differences = 0
        comparisons = 0
        
        for i in range(len(self.population)):
            for j in range(i + 1, len(self.population)):
                # Count different positions
                differences = sum(1 for a, b in zip(self.population[i].route, self.population[j].route) if a != b)
                total_differences += differences
                comparisons += 1
        
        return total_differences / (comparisons * self.num_cities) if comparisons > 0 else 0.0
    
    def run(self, verbose: bool = True, progress_callback: Callable = None) -> Individual:
        """
        Run the genetic algorithm.
        
        Args:
            verbose: Whether to print progress information
            progress_callback: Optional callback function for progress updates
            
        Returns:
            Best individual found
        """
        if verbose:
            print(f"Starting Genetic Algorithm for {self.num_cities}-city TSP")
            print(f"Population: {self.config.population_size}, Generations: {self.config.generations}")
            print(f"Elite: {self.config.elite_size}, Mutation Rate: {self.config.mutation_rate}")
            print(f"Selection: {self.config.selection_method}, Crossover: {self.config.crossover_method}")
        
        # Initialize population
        start_time = time.time()
        self.initialize_population()
        
        if verbose:
            best_distance = min(ind.distance for ind in self.population)
            print(f"Initial best distance: {best_distance:.1f}")
        
        # Evolution loop
        for generation in range(self.config.generations):
            self.evolve_generation()
            
            if verbose and (generation + 1) % 50 == 0:
                print(f"Generation {generation + 1}: Best = {self.best_individual.distance:.1f}, "
                      f"Avg = {self.history['average_distances'][-1]:.1f}, "
                      f"Diversity = {self.history['diversity_scores'][-1]:.3f}")
            
            if progress_callback:
                progress_callback(generation + 1, self.config.generations, self.best_individual)
        
        total_time = time.time() - start_time
        
        if verbose:
            print(f"\nAlgorithm completed in {total_time:.2f} seconds")
            print(f"Best route distance: {self.best_individual.distance:.1f}")
            print(f"Best route: {self.best_individual.route}")
        
        return self.best_individual
    
    def get_statistics(self) -> Dict:
        """Get comprehensive algorithm statistics."""
        return {
            'config': self.config,
            'best_distance': self.best_individual.distance if self.best_individual else None,
            'best_route': self.best_individual.route if self.best_individual else None,
            'generations_run': self.generation,
            'history': self.history,
            'final_diversity': self.history['diversity_scores'][-1] if self.history['diversity_scores'] else 0,
            'convergence_rate': self._calculate_convergence_rate()
        }
    
    def _calculate_convergence_rate(self) -> float:
        """Calculate the rate of convergence."""
        if len(self.history['best_distances']) < 2:
            return 0.0
        
        initial_distance = self.history['best_distances'][0]
        final_distance = self.history['best_distances'][-1]
        
        improvement = initial_distance - final_distance
        return improvement / initial_distance if initial_distance > 0 else 0.0