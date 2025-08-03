"""
Data analysis utilities for exploring the transportation network dataset.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import logging
from .data_loader import DataLoader

logger = logging.getLogger(__name__)


class DataAnalyzer:
    """Provides analysis and visualization tools for the transportation network data."""
    
    def __init__(self, data_loader: DataLoader):
        """
        Initialize with a data loader instance.
        
        Args:
            data_loader: DataLoader instance with loaded data
        """
        self.data_loader = data_loader
        self.nodes_df = data_loader.nodes_df
        self.edges_df = data_loader.edges_df
    
    def analyze_nodes_distribution(self) -> Dict[str, any]:
        """
        Analyze the geographical distribution of nodes.
        
        Returns:
            Dictionary with analysis results
        """
        if self.nodes_df is None:
            raise ValueError("Nodes data not loaded")
        
        analysis = {
            'total_nodes': len(self.nodes_df),
            'longitude_stats': {
                'min': self.nodes_df['longitude'].min(),
                'max': self.nodes_df['longitude'].max(),
                'mean': self.nodes_df['longitude'].mean(),
                'std': self.nodes_df['longitude'].std()
            },
            'latitude_stats': {
                'min': self.nodes_df['latitude'].min(),
                'max': self.nodes_df['latitude'].max(),
                'mean': self.nodes_df['latitude'].mean(),
                'std': self.nodes_df['latitude'].std()
            }
        }
        
        # Calculate geographical bounds
        lon_range = analysis['longitude_stats']['max'] - analysis['longitude_stats']['min']
        lat_range = analysis['latitude_stats']['max'] - analysis['latitude_stats']['min']
        
        analysis['geographical_bounds'] = {
            'longitude_range': lon_range,
            'latitude_range': lat_range,
            'aspect_ratio': lon_range / lat_range if lat_range > 0 else 0
        }
        
        logger.info(f"Node distribution analysis completed:")
        logger.info(f"  Geographical area: {lon_range:.4f}° longitude × {lat_range:.4f}° latitude")
        logger.info(f"  Center point: ({analysis['longitude_stats']['mean']:.4f}, {analysis['latitude_stats']['mean']:.4f})")
        
        return analysis
    
    def analyze_edge_connectivity(self) -> Dict[str, any]:
        """
        Analyze the connectivity patterns in the network.
        
        Returns:
            Dictionary with connectivity analysis
        """
        if self.edges_df is None:
            raise ValueError("Edges data not loaded")
        
        # Calculate out-degree for each node
        out_degrees = self.edges_df['id'].value_counts().sort_index()
        
        # Calculate in-degree for each node
        in_degrees = self.edges_df['destination_id'].value_counts().sort_index()
        
        # Get all node IDs and fill missing degrees with 0
        all_node_ids = self.data_loader.get_all_node_ids()
        out_degrees = out_degrees.reindex(all_node_ids, fill_value=0)
        in_degrees = in_degrees.reindex(all_node_ids, fill_value=0)
        
        analysis = {
            'total_edges': len(self.edges_df),
            'out_degree_stats': {
                'min': out_degrees.min(),
                'max': out_degrees.max(),
                'mean': out_degrees.mean(),
                'std': out_degrees.std()
            },
            'in_degree_stats': {
                'min': in_degrees.min(),
                'max': in_degrees.max(),
                'mean': in_degrees.mean(),
                'std': in_degrees.std()
            },
            'connectivity_metrics': {
                'nodes_with_no_outgoing': (out_degrees == 0).sum(),
                'nodes_with_no_incoming': (in_degrees == 0).sum(),
                'most_connected_out': out_degrees.idxmax(),
                'most_connected_in': in_degrees.idxmax(),
                'average_degree': (out_degrees.mean() + in_degrees.mean()) / 2
            }
        }
        
        logger.info(f"Edge connectivity analysis completed:")
        logger.info(f"  Average out-degree: {analysis['out_degree_stats']['mean']:.2f}")
        logger.info(f"  Average in-degree: {analysis['in_degree_stats']['mean']:.2f}")
        logger.info(f"  Nodes with no outgoing edges: {analysis['connectivity_metrics']['nodes_with_no_outgoing']}")
        logger.info(f"  Nodes with no incoming edges: {analysis['connectivity_metrics']['nodes_with_no_incoming']}")
        
        return analysis
    
    def analyze_distance_distribution(self) -> Dict[str, any]:
        """
        Analyze the distribution of edge distances.
        
        Returns:
            Dictionary with distance analysis
        """
        if self.edges_df is None:
            raise ValueError("Edges data not loaded")
        
        distances = self.edges_df['distance']
        
        analysis = {
            'distance_stats': {
                'count': len(distances),
                'min': distances.min(),
                'max': distances.max(),
                'mean': distances.mean(),
                'median': distances.median(),
                'std': distances.std(),
                'q25': distances.quantile(0.25),
                'q75': distances.quantile(0.75)
            },
            'distance_ranges': {
                'short_edges': (distances <= 10000).sum(),
                'medium_edges': ((distances > 10000) & (distances <= 30000)).sum(),
                'long_edges': (distances > 30000).sum()
            }
        }
        
        # Calculate percentile-based insights
        analysis['percentile_insights'] = {
            'p90': distances.quantile(0.90),
            'p95': distances.quantile(0.95),
            'p99': distances.quantile(0.99)
        }
        
        logger.info(f"Distance distribution analysis completed:")
        logger.info(f"  Distance range: {analysis['distance_stats']['min']:.1f} to {analysis['distance_stats']['max']:.1f}")
        logger.info(f"  Average distance: {analysis['distance_stats']['mean']:.1f}")
        logger.info(f"  Median distance: {analysis['distance_stats']['median']:.1f}")
        
        return analysis
    
    def identify_potential_outliers(self) -> Dict[str, List]:
        """
        Identify potential outliers in the dataset.
        
        Returns:
            Dictionary with lists of potential outliers
        """
        outliers = {
            'coordinate_outliers': [],
            'distance_outliers': [],
            'connectivity_outliers': []
        }
        
        if self.nodes_df is not None:
            # Identify coordinate outliers using IQR method
            for coord in ['longitude', 'latitude']:
                Q1 = self.nodes_df[coord].quantile(0.25)
                Q3 = self.nodes_df[coord].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                coord_outliers = self.nodes_df[
                    (self.nodes_df[coord] < lower_bound) | 
                    (self.nodes_df[coord] > upper_bound)
                ]['id'].tolist()
                
                outliers['coordinate_outliers'].extend(coord_outliers)
        
        if self.edges_df is not None:
            # Identify distance outliers
            Q1 = self.edges_df['distance'].quantile(0.25)
            Q3 = self.edges_df['distance'].quantile(0.75)
            IQR = Q3 - Q1
            upper_bound = Q3 + 1.5 * IQR
            
            distance_outlier_edges = self.edges_df[
                self.edges_df['distance'] > upper_bound
            ][['id', 'destination_id', 'distance']].to_dict('records')
            
            outliers['distance_outliers'] = distance_outlier_edges
            
            # Identify connectivity outliers (nodes with unusually high degree)
            out_degrees = self.edges_df['id'].value_counts()
            degree_threshold = out_degrees.quantile(0.95)
            connectivity_outliers = out_degrees[out_degrees > degree_threshold].index.tolist()
            outliers['connectivity_outliers'] = connectivity_outliers
        
        logger.info(f"Outlier analysis completed:")
        logger.info(f"  Coordinate outliers: {len(set(outliers['coordinate_outliers']))}")
        logger.info(f"  Distance outliers: {len(outliers['distance_outliers'])}")
        logger.info(f"  Connectivity outliers: {len(outliers['connectivity_outliers'])}")
        
        return outliers
    
    def get_network_summary(self) -> Dict[str, any]:
        """
        Get a comprehensive summary of the network characteristics.
        
        Returns:
            Dictionary with network summary
        """
        summary = {
            'data_loaded': {
                'nodes': self.nodes_df is not None,
                'edges': self.edges_df is not None
            }
        }
        
        if self.nodes_df is not None and self.edges_df is not None:
            # Basic counts
            summary['basic_stats'] = {
                'total_nodes': len(self.nodes_df),
                'total_edges': len(self.edges_df),
                'edge_density': len(self.edges_df) / (len(self.nodes_df) * (len(self.nodes_df) - 1))
            }
            
            # Data consistency
            consistency = self.data_loader.validate_data_consistency()
            summary['data_consistency'] = consistency
            
            # Quick analyses
            try:
                node_analysis = self.analyze_nodes_distribution()
                edge_analysis = self.analyze_edge_connectivity()
                distance_analysis = self.analyze_distance_distribution()
                
                summary['geographical_extent'] = node_analysis['geographical_bounds']
                summary['connectivity'] = {
                    'avg_out_degree': edge_analysis['out_degree_stats']['mean'],
                    'avg_in_degree': edge_analysis['in_degree_stats']['mean']
                }
                summary['distances'] = {
                    'min': distance_analysis['distance_stats']['min'],
                    'max': distance_analysis['distance_stats']['max'],
                    'mean': distance_analysis['distance_stats']['mean']
                }
                
            except Exception as e:
                logger.warning(f"Error in summary analysis: {e}")
                summary['analysis_error'] = str(e)
        
        return summary
    
    def plot_basic_visualizations(self, save_path: Optional[str] = None):
        """
        Create basic visualizations of the dataset.
        
        Args:
            save_path: Optional path to save the plots
        """
        if self.nodes_df is None or self.edges_df is None:
            raise ValueError("Both nodes and edges data must be loaded")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Transportation Network Analysis', fontsize=16)
        
        # 1. Node geographical distribution
        axes[0, 0].scatter(self.nodes_df['longitude'], self.nodes_df['latitude'], 
                          alpha=0.6, s=1)
        axes[0, 0].set_title('Node Geographical Distribution')
        axes[0, 0].set_xlabel('Longitude')
        axes[0, 0].set_ylabel('Latitude')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Distance distribution histogram
        axes[0, 1].hist(self.edges_df['distance'], bins=50, alpha=0.7, edgecolor='black')
        axes[0, 1].set_title('Edge Distance Distribution')
        axes[0, 1].set_xlabel('Distance')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Node degree distribution
        out_degrees = self.edges_df['id'].value_counts()
        axes[1, 0].hist(out_degrees.values, bins=30, alpha=0.7, edgecolor='black')
        axes[1, 0].set_title('Node Out-Degree Distribution')
        axes[1, 0].set_xlabel('Out-Degree')
        axes[1, 0].set_ylabel('Number of Nodes')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Distance vs node degree scatter
        node_degrees = self.edges_df.groupby('id').agg({
            'distance': ['mean', 'count']
        }).round(2)
        node_degrees.columns = ['avg_distance', 'degree']
        
        axes[1, 1].scatter(node_degrees['degree'], node_degrees['avg_distance'], 
                          alpha=0.6)
        axes[1, 1].set_title('Average Distance vs Node Degree')
        axes[1, 1].set_xlabel('Node Degree')
        axes[1, 1].set_ylabel('Average Edge Distance')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Visualizations saved to {save_path}")
        
        plt.show()
        return fig