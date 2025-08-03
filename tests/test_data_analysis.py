"""
Test script for data loading and analysis.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.utils.data_loader import DataLoader
from src.utils.data_analysis import DataAnalyzer
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Main function to test data loading and analysis."""
    
    # Define data paths
    nodes_path = os.path.join('data', 'nodes.csv')
    edges_path = os.path.join('data', 'edges.csv')
    
    try:
        # Initialize data loader
        logger.info("Initializing data loader...")
        data_loader = DataLoader(nodes_path, edges_path)
        
        # Load data
        logger.info("Loading nodes data...")
        nodes_df = data_loader.load_nodes()
        
        logger.info("Loading edges data...")
        edges_df = data_loader.load_edges()
        
        # Validate data consistency
        logger.info("Validating data consistency...")
        consistency_results = data_loader.validate_data_consistency()
        print(f"Data consistency: {consistency_results}")
        
        # Initialize analyzer
        logger.info("Initializing data analyzer...")
        analyzer = DataAnalyzer(data_loader)
        
        # Get network summary
        logger.info("Generating network summary...")
        summary = analyzer.get_network_summary()
        print("\\n=== NETWORK SUMMARY ===")
        for key, value in summary.items():
            print(f"{key}: {value}")
        
        # Analyze nodes distribution
        logger.info("Analyzing nodes distribution...")
        node_analysis = analyzer.analyze_nodes_distribution()
        print("\\n=== NODE DISTRIBUTION ANALYSIS ===")
        print(f"Total nodes: {node_analysis['total_nodes']}")
        print(f"Longitude range: {node_analysis['longitude_stats']['min']:.6f} to {node_analysis['longitude_stats']['max']:.6f}")
        print(f"Latitude range: {node_analysis['latitude_stats']['min']:.6f} to {node_analysis['latitude_stats']['max']:.6f}")
        print(f"Geographical bounds: {node_analysis['geographical_bounds']}")
        
        # Analyze edge connectivity
        logger.info("Analyzing edge connectivity...")
        edge_analysis = analyzer.analyze_edge_connectivity()
        print("\\n=== EDGE CONNECTIVITY ANALYSIS ===")
        print(f"Total edges: {edge_analysis['total_edges']}")
        print(f"Average out-degree: {edge_analysis['out_degree_stats']['mean']:.2f}")
        print(f"Average in-degree: {edge_analysis['in_degree_stats']['mean']:.2f}")
        print(f"Connectivity metrics: {edge_analysis['connectivity_metrics']}")
        
        # Analyze distance distribution
        logger.info("Analyzing distance distribution...")
        distance_analysis = analyzer.analyze_distance_distribution()
        print("\\n=== DISTANCE DISTRIBUTION ANALYSIS ===")
        print(f"Distance stats: {distance_analysis['distance_stats']}")
        print(f"Distance ranges: {distance_analysis['distance_ranges']}")
        
        # Identify outliers
        logger.info("Identifying outliers...")
        outliers = analyzer.identify_potential_outliers()
        print("\\n=== OUTLIER ANALYSIS ===")
        print(f"Coordinate outliers: {len(set(outliers['coordinate_outliers']))}")
        print(f"Distance outliers: {len(outliers['distance_outliers'])}")
        print(f"Connectivity outliers: {len(outliers['connectivity_outliers'])}")
        
        # Test specific data access methods
        logger.info("Testing data access methods...")
        print("\\n=== DATA ACCESS TESTS ===")
        
        # Get coordinates for first few nodes
        node_ids = data_loader.get_all_node_ids()[:5]
        for node_id in node_ids:
            coords = data_loader.get_node_coordinates(node_id)
            print(f"Node {node_id}: {coords}")
        
        # Test edge distance lookup
        if len(edges_df) > 0:
            first_edge = edges_df.iloc[0]
            source_id = int(first_edge['id'])
            dest_id = int(first_edge['destination_id'])
            distance = data_loader.get_edge_distance(source_id, dest_id)
            print(f"Edge {source_id} -> {dest_id}: distance = {distance}")
        
        logger.info("Data analysis completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during data analysis: {e}")
        raise

if __name__ == "__main__":
    main()