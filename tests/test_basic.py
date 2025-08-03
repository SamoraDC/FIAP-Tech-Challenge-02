print("Starting test...")
import pandas as pd
print("Pandas imported successfully")

try:
    nodes_df = pd.read_csv('data/nodes.csv')
    print(f"SUCCESS: Loaded {len(nodes_df)} nodes")
    
    edges_df = pd.read_csv('data/edges.csv')
    print(f"SUCCESS: Loaded {len(edges_df)} edges")
    
except Exception as e:
    print(f"ERROR: {e}")

print("Test completed")