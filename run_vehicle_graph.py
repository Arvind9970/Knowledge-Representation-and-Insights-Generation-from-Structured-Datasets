"""
Optimized visualization script for Vehicle Knowledge Graph.
"""
import os
import sys
import logging
import argparse
import pandas as pd
from pathlib import Path

# Add parent directory to path to allow imports
sys.path.append(str(Path(__file__).parent))

from src.knowledge_graph_optimized import KnowledgeGraph

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Visualize Vehicle Knowledge Graph')
    parser.add_argument('--input', type=str, default='data/processed/processed_data.csv',
                      help='Path to processed data file')
    parser.add_argument('--output', type=str, default='vehicle_graph',
                      help='Output file path (without extension)')
    parser.add_argument('--max-nodes', type=int, default=100,
                      help='Maximum number of nodes to display')
    parser.add_argument('--max-edges', type=int, default=200,
                      help='Maximum number of edges to display')
    parser.add_argument('--layout', type=str, default='kamada_kawai',
                      choices=['spring', 'circular', 'kamada_kawai', 'random', 'fruchterman_reingold'],
                      help='Layout algorithm for the graph')
    parser.add_argument('--interactive', action='store_true',
                      help='Generate interactive HTML visualization')
    return parser.parse_args()

def load_data(file_path: str):
    """Load processed data from CSV file."""
    if not os.path.exists(file_path):
        logger.error(f"Processed data not found at {file_path}")
        return None
    
    try:
        logger.info(f"Loading data from {file_path}...")
        return pd.read_csv(file_path)
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return None

def create_knowledge_graph(data, kg: KnowledgeGraph) -> None:
    """Create knowledge graph from vehicle data."""
    logger.info("Creating knowledge graph from vehicle data...")
    
    # Add vehicles as entities
    for idx, row in data.iterrows():
        vehicle_id = f"vehicle_{int(row['id'])}" if pd.notna(row.get('id')) else f"vehicle_{idx}"
        
        # Prepare vehicle attributes
        attributes = {
            'price': row.get('price'),
            'year': int(row['year']) if pd.notna(row.get('year')) else None,
            'manufacturer': row.get('manufacturer'),
            'model': row.get('model'),
            'condition': row.get('condition'),
            'odometer': int(row['odometer']) if pd.notna(row.get('odometer')) else None,
            'fuel': row.get('fuel'),
            'transmission': row.get('transmission'),
            'type': row.get('type'),
            'state': row.get('state')
        }
        
        # Add vehicle entity
        kg.add_entity(vehicle_id, 'vehicle', attributes)
        
        # Add manufacturer as a separate entity and create relationship
        if pd.notna(row.get('manufacturer')) and row['manufacturer'] != 'Unknown':
            manufacturer_id = f"manufacturer_{row['manufacturer'].lower().replace(' ', '_')}"
            kg.add_entity(manufacturer_id, 'manufacturer', {'name': row['manufacturer']})
            kg.add_relationship(manufacturer_id, vehicle_id, 'manufactures')
        
        # Add state as a separate entity and create relationship
        if pd.notna(row.get('state')):
            state_id = f"state_{row['state'].lower()}"
            kg.add_entity(state_id, 'state', {'name': row['state']})
            kg.add_relationship(vehicle_id, state_id, 'located_in')
        
        # Add relationship for vehicle type if available
        if pd.notna(row.get('type')) and row['type'] != 'Unknown':
            type_id = f"type_{row['type'].lower().replace(' ', '_')}"
            kg.add_entity(type_id, 'vehicle_type', {'name': row['type']})
            kg.add_relationship(vehicle_id, type_id, 'is_type')
    
    logger.info(f"Knowledge graph created with {len(kg.graph.nodes)} nodes and {len(kg.graph.edges)} edges")

def main():
    """Main function to run the visualization."""
    args = parse_arguments()
    
    # Create output directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Load data
    data = load_data(args.input)
    if data is None:
        return
    
    # Initialize knowledge graph
    kg = KnowledgeGraph()
    
    # Create knowledge graph from data
    create_knowledge_graph(data, kg)
    
    # Generate visualization
    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)
    
    # Create output file path
    output_base = os.path.join(output_dir, os.path.basename(args.output))
    
    if args.interactive:
        output_file = f"{output_base}.html"
        logger.info(f"Generating interactive visualization to {output_file}...")
        kg.visualize_interactive(
            max_nodes=args.max_nodes,
            max_edges=args.max_edges,
            output_file=output_file
        )
    else:
        output_file = f"{output_base}.png"
        logger.info(f"Generating static visualization to {output_file}...")
        kg.visualize_graph(
            layout=args.layout,
            max_nodes=args.max_nodes,
            max_edges=args.max_edges,
            output_file=output_file
        )
    
    # Print graph statistics
    stats = kg.calculate_statistics()
    logger.info("\nGraph Statistics:")
    for key, value in stats.items():
        logger.info(f"- {key.replace('_', ' ').title()}: {value}")

if __name__ == "__main__":
    main()
