import pandas as pd
import sys
import os

print("Testing basic data access...")

# Test basic file access
try:
    nodes = pd.read_csv('data/nodes.csv')
    print(f"Nodes loaded: {len(nodes)} rows")
    print(f"Columns: {list(nodes.columns)}")
    print(f"First few nodes:\n{nodes.head()}")
    
    edges = pd.read_csv('data/edges.csv')
    print(f"\nEdges loaded: {len(edges)} rows")
    print(f"Columns: {list(edges.columns)}")
    print(f"First few edges:\n{edges.head()}")
    
    print("\n✓ Basic data loading successful!")
    
    # Now test our utilities
    print("\n=== Testing Custom Utilities ===")
    sys.path.append('src')
    
    from utils.data_loader import DataLoader
    
    loader = DataLoader('data/nodes.csv', 'data/edges.csv')
    nodes_df = loader.load_nodes()
    edges_df = loader.load_edges()
    
    print(f"✓ DataLoader: {len(nodes_df)} nodes, {len(edges_df)} edges")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()