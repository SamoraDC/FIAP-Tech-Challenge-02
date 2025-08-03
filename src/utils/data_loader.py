"""
Data loading utilities for the FIAP Tech Challenge Phase 2.
Handles loading and preprocessing of Brazilian transportation network data.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import networkx as nx
from pathlib import Path

class TransportationDataLoader:
    """
    Loads and preprocesses the Brazilian transportation network data.
    
    Handles nodes.csv (1000 Brazilian locations) and edges.csv (~500k connections)
    for use in genetic algorithm and comparison optimization methods.
    """
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize the data loader.
        
        Args:
            data_dir: Directory containing nodes.csv and edges.csv files
        """
        self.data_dir = Path(data_dir)
        self.nodes_df: Optional[pd.DataFrame] = None
        self.edges_df: Optional[pd.DataFrame] = None
        self.graph: Optional[nx.Graph] = None
        
    def load_nodes(self) -> pd.DataFrame:
        """
        Load nodes.csv containing Brazilian location coordinates.
        
        Returns:
            DataFrame with columns: id, longitude, latitude
        """
        nodes_path = self.data_dir / "nodes.csv"
        
        if not nodes_path.exists():
            raise FileNotFoundError(f"Nodes file not found: {nodes_path}")
            
        print(f"Loading nodes from {nodes_path}")
        self.nodes_df = pd.read_csv(nodes_path)
        
        # Data validation
        expected_columns = ['id', 'longitude', 'latitude']
        if not all(col in self.nodes_df.columns for col in expected_columns):
            raise ValueError(f"Nodes file must contain columns: {expected_columns}")
            
        print(f"Loaded {len(self.nodes_df)} nodes")
        print(f"Coordinate bounds: Longitude [{self.nodes_df['longitude'].min():.3f}, {self.nodes_df['longitude'].max():.3f}]")
        print(f"                   Latitude [{self.nodes_df['latitude'].min():.3f}, {self.nodes_df['latitude'].max():.3f}]")
        
        return self.nodes_df
    
    def load_edges(self, sample_size: Optional[int] = None) -> pd.DataFrame:
        """
        Load edges.csv containing weighted connections between nodes.
        
        Args:
            sample_size: If provided, load only a random sample of edges for testing
            
        Returns:
            DataFrame with columns: id, destination_id, distance
        """
        edges_path = self.data_dir / "edges.csv"
        
        if not edges_path.exists():
            raise FileNotFoundError(f"Edges file not found: {edges_path}")
            
        print(f"Loading edges from {edges_path}")
        
        if sample_size:
            # Load sample for testing/development
            print(f"Loading sample of {sample_size} edges for testing")
            self.edges_df = pd.read_csv(edges_path, nrows=sample_size)
        else:
            # Load full dataset (may take time due to size)
            print("Loading full edge dataset (~500k edges)...")
            self.edges_df = pd.read_csv(edges_path)
        
        # Data validation
        expected_columns = ['id', 'destination_id', 'distance']
        if not all(col in self.edges_df.columns for col in expected_columns):
            raise ValueError(f"Edges file must contain columns: {expected_columns}")
            
        print(f"Loaded {len(self.edges_df)} edges")
        print(f"Distance range: [{self.edges_df['distance'].min():.1f}, {self.edges_df['distance'].max():.1f}] meters")
        print(f"Average distance: {self.edges_df['distance'].mean():.1f} meters")
        
        return self.edges_df
    
    def create_graph(self, use_sample: bool = False, sample_nodes: int = 50) -> nx.Graph:
        """
        Create NetworkX graph from loaded data.
        
        Args:
            use_sample: Whether to create a sample graph for testing
            sample_nodes: Number of nodes to include in sample
            
        Returns:
            NetworkX Graph object
        """
        if self.nodes_df is None:
            self.load_nodes()
        if self.edges_df is None:
            self.load_edges(sample_size=10000 if use_sample else None)
            
        print(f"Creating NetworkX graph...")
        
        # Create graph
        self.graph = nx.Graph()
        
        if use_sample:
            # Create sample for testing with connected nodes
            # First, get a connected subgraph by selecting nodes that have many connections
            node_counts = self.edges_df['id'].value_counts()
            top_connected_nodes = node_counts.head(sample_nodes).index.tolist()
            
            sample_node_ids = top_connected_nodes
            sample_nodes_df = self.nodes_df[self.nodes_df['id'].isin(sample_node_ids)]
            sample_edges_df = self.edges_df[
                (self.edges_df['id'].isin(sample_node_ids)) & 
                (self.edges_df['destination_id'].isin(sample_node_ids))
            ]
            
            # Add nodes with coordinates
            for _, node in sample_nodes_df.iterrows():
                self.graph.add_node(node['id'], 
                                  longitude=node['longitude'], 
                                  latitude=node['latitude'])
            
            # Add edges with distances
            for _, edge in sample_edges_df.iterrows():
                self.graph.add_edge(edge['id'], edge['destination_id'], 
                                  weight=edge['distance'])
                                  
            print(f"Created sample graph: {len(self.graph.nodes)} nodes, {len(self.graph.edges)} edges")
        else:
            # Create full graph
            # Add all nodes with coordinates
            for _, node in self.nodes_df.iterrows():
                self.graph.add_node(node['id'], 
                                  longitude=node['longitude'], 
                                  latitude=node['latitude'])
            
            # Add edges with distances (may take time)
            print("Adding edges to graph...")
            for _, edge in self.edges_df.iterrows():
                self.graph.add_edge(edge['id'], edge['destination_id'], 
                                  weight=edge['distance'])
                                  
            print(f"Created full graph: {len(self.graph.nodes)} nodes, {len(self.graph.edges)} edges")
        
        return self.graph
    
    def get_node_coordinates(self, node_id: int) -> Tuple[float, float]:
        """
        Get longitude, latitude coordinates for a given node.
        
        Args:
            node_id: Node identifier
            
        Returns:
            Tuple of (longitude, latitude)
        """
        if self.nodes_df is None:
            raise ValueError("Nodes not loaded. Call load_nodes() first.")
            
        node_row = self.nodes_df[self.nodes_df['id'] == node_id]
        if node_row.empty:
            raise ValueError(f"Node {node_id} not found")
            
        return node_row['longitude'].iloc[0], node_row['latitude'].iloc[0]
    
    def get_distance_matrix(self, node_subset: Optional[List[int]] = None) -> np.ndarray:
        """
        Create distance matrix for specified nodes.
        
        Args:
            node_subset: List of node IDs to include. If None, uses all nodes.
            
        Returns:
            Distance matrix as numpy array
        """
        if self.graph is None:
            raise ValueError("Graph not created. Call create_graph() first.")
            
        nodes = node_subset if node_subset else list(self.graph.nodes())
        n = len(nodes)
        
        # Create distance matrix
        distance_matrix = np.full((n, n), np.inf)
        
        for i, node1 in enumerate(nodes):
            distance_matrix[i, i] = 0  # Distance to self is 0
            for j, node2 in enumerate(nodes):
                if i != j and self.graph.has_edge(node1, node2):
                    distance_matrix[i, j] = self.graph[node1][node2]['weight']
        
        return distance_matrix, nodes
    
    def get_dataset_statistics(self) -> Dict:
        """
        Generate comprehensive statistics about the dataset.
        
        Returns:
            Dictionary containing dataset statistics
        """
        stats = {}
        
        if self.nodes_df is not None:
            stats['nodes'] = {
                'count': len(self.nodes_df),
                'longitude_range': [self.nodes_df['longitude'].min(), self.nodes_df['longitude'].max()],
                'latitude_range': [self.nodes_df['latitude'].min(), self.nodes_df['latitude'].max()],
                'longitude_center': self.nodes_df['longitude'].mean(),
                'latitude_center': self.nodes_df['latitude'].mean()
            }
        
        if self.edges_df is not None:
            stats['edges'] = {
                'count': len(self.edges_df),
                'distance_min': self.edges_df['distance'].min(),
                'distance_max': self.edges_df['distance'].max(),
                'distance_mean': self.edges_df['distance'].mean(),
                'distance_std': self.edges_df['distance'].std(),
                'unique_source_nodes': self.edges_df['id'].nunique(),
                'unique_destination_nodes': self.edges_df['destination_id'].nunique()
            }
        
        if self.graph is not None:
            stats['graph'] = {
                'nodes': len(self.graph.nodes),
                'edges': len(self.graph.edges),
                'density': nx.density(self.graph),
                'is_connected': nx.is_connected(self.graph)
            }
            
        return stats

# Convenience function for quick data loading
def load_transportation_data(sample_nodes: int = None) -> TransportationDataLoader:
    """
    Quick loader function for transportation data.
    
    Args:
        sample_nodes: If provided, creates sample graph with specified number of nodes
        
    Returns:
        Initialized TransportationDataLoader with loaded data
    """
    loader = TransportationDataLoader()
    loader.load_nodes()
    
    if sample_nodes:
        loader.load_edges(sample_size=10000)  # Load sample edges for testing
        loader.create_graph(use_sample=True, sample_nodes=sample_nodes)
    else:
        loader.load_edges()  # Load full dataset
        loader.create_graph(use_sample=False)
    
    return loader