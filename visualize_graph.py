"""
Knowledge Graph Visualization Script

This script loads processed data and generates a visualization of the knowledge graph.
"""

import os
import sys
import logging
import pandas as pd
from pathlib import Path

# Add parent directory to path to allow imports
sys.path.append(str(Path(__file__).parent.parent))

from src.data_processor import DataProcessor
from src.insights_generator import InsightsGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def main():
    try:
        # Paths
        processed_data_path = "data/processed_data.csv"
        
        # Check if processed data exists
        if not os.path.exists(processed_data_path):
            logger.error(f"Processed data not found at {processed_data_path}. Please run the main application first.")
            return
        
        logger.info("Loading processed data...")
        data = pd.read_csv(processed_data_path)
        
        # Initialize components
        logger.info("Initializing components...")
        processor = DataProcessor()
        generator = InsightsGenerator()
        
        # Create knowledge graph
        logger.info("Creating knowledge graph...")
        generator.create_knowledge_graph(data)
        
        # Visualize the graph
        logger.info("Generating visualization (this may take a moment)...")
        generator.knowledge_graph.visualize_graph(
            layout='spring',
            max_nodes=100,  # Limit nodes for better visualization
            max_edges=200   # Limit edges for better visualization
        )
        
        logger.info("Visualization complete! Check the popup window.")
        
    except Exception as e:
        logger.error(f"Error generating visualization: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()
