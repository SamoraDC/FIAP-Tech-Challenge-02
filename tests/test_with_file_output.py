import pandas as pd
import os

# Write results to file
with open('test_results.txt', 'w') as f:
    f.write("=== DATA LOADING TEST RESULTS ===\n\n")
    
    try:
        # Test nodes loading
        f.write("Loading nodes.csv...\n")
        nodes_df = pd.read_csv('data/nodes.csv')
        f.write(f"SUCCESS: Loaded {len(nodes_df)} nodes\n")
        f.write(f"Columns: {list(nodes_df.columns)}\n")
        f.write(f"First 3 rows:\n{nodes_df.head(3).to_string()}\n\n")
        
        # Test edges loading
        f.write("Loading edges.csv...\n")
        edges_df = pd.read_csv('data/edges.csv')
        f.write(f"SUCCESS: Loaded {len(edges_df)} edges\n")
        f.write(f"Columns: {list(edges_df.columns)}\n")
        f.write(f"First 3 rows:\n{edges_df.head(3).to_string()}\n\n")
        
        # Basic statistics
        f.write("=== BASIC STATISTICS ===\n")
        f.write(f"Total nodes: {len(nodes_df)}\n")
        f.write(f"Total edges: {len(edges_df)}\n")
        f.write(f"Longitude range: {nodes_df['longitude'].min():.6f} to {nodes_df['longitude'].max():.6f}\n")
        f.write(f"Latitude range: {nodes_df['latitude'].min():.6f} to {nodes_df['latitude'].max():.6f}\n")
        f.write(f"Distance range: {edges_df['distance'].min():.1f} to {edges_df['distance'].max():.1f}\n")
        f.write(f"Average distance: {edges_df['distance'].mean():.1f}\n\n")
        
        # Data validation
        f.write("=== DATA VALIDATION ===\n")
        f.write(f"Nodes with null values: {nodes_df.isnull().sum().sum()}\n")
        f.write(f"Edges with null values: {edges_df.isnull().sum().sum()}\n")
        f.write(f"Duplicate node IDs: {nodes_df['id'].duplicated().sum()}\n")
        f.write(f"Negative distances: {(edges_df['distance'] < 0).sum()}\n")
        
        f.write("\n=== TEST COMPLETED SUCCESSFULLY ===\n")
        
    except Exception as e:
        f.write(f"ERROR: {str(e)}\n")

print("Test completed. Check test_results.txt for output.")