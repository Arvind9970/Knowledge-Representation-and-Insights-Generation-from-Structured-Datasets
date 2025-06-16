import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple, Union

from src.knowledge_graph import KnowledgeGraph

# Configure logging
logger = logging.getLogger(__name__)

class InsightsError(Exception):
    """Custom exception for insights generation errors."""
    pass

class InsightsGenerator:
    def __init__(self):
        """
        Initialize the InsightsGenerator with a KnowledgeGraph instance.
        """
        self.knowledge_graph = KnowledgeGraph()
        self.insights = {}
        self.logger = logging.getLogger(__name__)

    def _process_numerical_column(self, data: pd.Series, col: str) -> Dict[str, float]:
        """Process a single numerical column to calculate statistics."""
        try:
            # Use numpy for faster calculations
            values = data[col].dropna().values
            if len(values) == 0:
                return {}
                
            stats = {
                'count': float(len(values)),
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                '25%': float(np.percentile(values, 25)),
                '50%': float(np.median(values)),
                '75%': float(np.percentile(values, 75)),
                'max': float(np.max(values))
            }
            return stats
        except Exception as e:
            self.logger.warning(f"Could not calculate statistics for numerical column '{col}': {str(e)}")
            return {}

    def _process_categorical_column(self, data: pd.Series, col: str, max_categories: int = 20) -> Dict[str, Any]:
        """Process a single categorical column to calculate statistics."""
        try:
            value_counts = data[col].value_counts()
            
            # Limit the number of categories to avoid excessive memory usage
            if len(value_counts) > max_categories:
                top_values = value_counts.head(max_categories)
                other_count = value_counts.iloc[max_categories:].sum()
                value_counts = top_values.append(pd.Series({'Other': other_count}))
            
            return {
                'unique_values': int(len(value_counts)),
                'value_counts': value_counts.to_dict()
            }
        except Exception as e:
            self.logger.warning(f"Could not calculate statistics for categorical column '{col}': {str(e)}")
            return {}

    def generate_statistical_insights(self, data: pd.DataFrame, sample_size: int = None) -> Dict[str, Any]:
        """
        Generate comprehensive statistical insights from the input data.
        
        Args:
            data: Input pandas DataFrame containing the data to analyze
            sample_size: If provided, use a random sample of the data for faster processing
            
        Returns:
            Dict containing various statistical insights including:
            - Basic statistics (count, mean, std, min, max, quartiles)
            - Correlation analysis
            - Distribution analysis
            
        Raises:
            InsightsError: If input data is empty or invalid
        """
        self.logger.info("Generating statistical insights...")
        
        if data is None or data.empty:
            error_msg = "Cannot generate insights: Input data is empty"
            self.logger.error(error_msg)
            raise InsightsError(error_msg)
        
        # Use a sample of the data if requested and the dataset is large
        if sample_size and len(data) > sample_size:
            self.logger.info(f"Using random sample of {sample_size} rows for faster processing")
            data = data.sample(n=min(sample_size, len(data)), random_state=42)
        
        insights = {}
        
        try:
            # Basic statistics - optimized with vectorized operations
            insights['basic_stats'] = {
                'num_records': int(len(data)),
                'num_columns': int(len(data.columns)),
                'missing_values': data.isnull().sum().to_dict(),
                'duplicate_rows': int(data.duplicated().sum())
            }
            
            # Process numerical columns in parallel
            numerical_cols = data.select_dtypes(include=['number']).columns
            if len(numerical_cols) > 0:
                insights['numerical_stats'] = {}
                for col in numerical_cols:
                    stats = self._process_numerical_column(data, col)
                    if stats:  # Only add if we got valid statistics
                        insights['numerical_stats'][col] = stats
            
            # Process categorical columns
            categorical_cols = data.select_dtypes(include=['object', 'category', 'bool']).columns
            if len(categorical_cols) > 0:
                insights['categorical_stats'] = {}
                for col in categorical_cols:
                    stats = self._process_categorical_column(data, col)
                    if stats:  # Only add if we got valid statistics
                        insights['categorical_stats'][col] = stats
            
            # Add correlation matrix for numerical columns (limited to first 50 columns)
            if len(numerical_cols) > 1 and len(numerical_cols) <= 50:
                try:
                    corr_matrix = data[numerical_cols].corr()
                    # Convert to dict and round to 2 decimal places to reduce size
                    insights['correlation_matrix'] = corr_matrix.round(2).to_dict()
                except Exception as e:
                    self.logger.warning(f"Could not calculate correlation matrix: {str(e)}")
            
            return insights
            
        except Exception as e:
            error_msg = f"Error generating statistical insights: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            raise InsightsError(error_msg) from e

    def create_knowledge_graph(self, data: pd.DataFrame) -> None:
        """
        Create a knowledge graph from the input data.
        
        Args:
            data: Input pandas DataFrame containing the data to model
            
        Raises:
            InsightsError: If required columns are missing or data is invalid
        """
        self.logger.info("Creating knowledge graph from data...")
        
        if data is None or data.empty:
            error_msg = "Cannot create knowledge graph: Input data is empty"
            self.logger.error(error_msg)
            raise InsightsError(error_msg)
            
        required_columns = ['id', 'manufacturer', 'model', 'price', 'year']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            error_msg = f"Missing required columns: {', '.join(missing_columns)}"
            self.logger.error(error_msg)
            raise InsightsError(error_msg)
            
        try:
            # Add vehicle entities
            for idx, row in data.iterrows():
                if pd.isna(row.get('id')):
                    self.logger.warning(f"Skipping row {idx} with missing ID")
                    continue
                    
                vehicle_id = f"vehicle_{row['id']}"
                
                # Prepare vehicle attributes
                vehicle_attrs = {
                    'url': row.get('url'),
                    'price': float(row.get('price', 0)) if pd.notna(row.get('price')) else None,
                    'year': int(row.get('year')) if pd.notna(row.get('year')) else None,
                    'manufacturer': str(row.get('manufacturer', '')),
                    'model': str(row.get('model', '')),
                    'condition': str(row.get('condition', '')),
                    'cylinders': str(row.get('cylinders', '')) if pd.notna(row.get('cylinders')) else None,
                    'fuel': str(row.get('fuel', '')) if pd.notna(row.get('fuel')) else None,
                    'odometer': float(row.get('odometer', 0)) if pd.notna(row.get('odometer')) else None,
                    'title_status': str(row.get('title_status', '')) if pd.notna(row.get('title_status')) else None,
                    'transmission': str(row.get('transmission', '')) if pd.notna(row.get('transmission')) else None,
                    'VIN': str(row.get('VIN', '')) if pd.notna(row.get('VIN')) else None,
                    'drive': str(row.get('drive', '')) if pd.notna(row.get('drive')) else None,
                    'size': str(row.get('size', '')) if pd.notna(row.get('size')) else None,
                    'type': str(row.get('type', '')) if pd.notna(row.get('type')) else None,
                    'paint_color': str(row.get('paint_color', '')) if pd.notna(row.get('paint_color')) else None,
                    'image_url': str(row.get('image_url', '')) if pd.notna(row.get('image_url')) else None,
                    'description': str(row.get('description', '')) if pd.notna(row.get('description')) else None,
                    'county': str(row.get('county', '')) if pd.notna(row.get('county')) else None,
                    'state': str(row.get('state', '')) if pd.notna(row.get('state')) else None,
                    'lat': float(row.get('lat')) if pd.notna(row.get('lat')) else None,
                    'long': float(row.get('long')) if pd.notna(row.get('long')) else None,
                    'posting_date': str(row.get('posting_date', '')) if pd.notna(row.get('posting_date')) else None
                }
                
                # Add vehicle entity
                self.knowledge_graph.add_entity(
                    entity_id=vehicle_id,
                    entity_type='vehicle',
                    attributes={k: v for k, v in vehicle_attrs.items() if v is not None}
                )
                
                # Add manufacturer relationship if manufacturer exists
                if pd.notna(row.get('manufacturer')):
                    manufacturer_id = f"manufacturer_{row['manufacturer'].lower().replace(' ', '_')}"
                    self.knowledge_graph.add_entity(
                        entity_id=manufacturer_id,
                        entity_type='manufacturer',
                        attributes={'name': row['manufacturer']}
                    )
                    
                    self.knowledge_graph.add_relationship(
                        source=vehicle_id,
                        target=manufacturer_id,
                        relationship_type='manufactured_by',
                        attributes={'since': int(row['year'])} if pd.notna(row.get('year')) else {}
                    )
                    
            self.logger.info(f"Knowledge graph created with {len(self.knowledge_graph.entities)} entities")
            
        except Exception as e:
            error_msg = f"Error creating knowledge graph: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            raise InsightsError(error_msg) from e

    def generate_graph_patterns(self, min_support: float = 0.1) -> List[Dict[str, Any]]:
        """
        Generate interesting patterns from the knowledge graph.
        
        Args:
            min_support: Minimum support threshold for pattern mining (0-1)
            
        Returns:
            List of dictionaries containing discovered patterns with their statistics
            
        Raises:
            InsightsError: If pattern generation fails
        """
        self.logger.info(f"Generating graph patterns with min_support={min_support}")
        
        if not hasattr(self, 'knowledge_graph') or not self.knowledge_graph.entities:
            error_msg = "Knowledge graph is empty. Create it first using create_knowledge_graph()"
            self.logger.error(error_msg)
            raise InsightsError(error_msg)
            
        try:
            patterns = []
            
            # Example pattern: Find common vehicle features by manufacturer
            manufacturer_features = {}
            
            # This is a simplified example - in a real implementation, you would use
            # a proper graph mining algorithm here
            for entity_id, entity_data in self.knowledge_graph.entities.items():
                if entity_data.get('type') == 'vehicle':
                    manufacturer = None
                    features = {}
                    
                    # Get manufacturer relationship using graph edges
                    for neighbor in self.knowledge_graph.graph.neighbors(entity_id):
                        edge_data = self.knowledge_graph.graph.get_edge_data(entity_id, neighbor)
                        if edge_data and edge_data.get('type') == 'manufactured_by':
                            manufacturer = neighbor.replace('manufacturer_', '')
                            break
                            
                    if not manufacturer:
                        continue
                        
                    # Count features
                    if manufacturer not in manufacturer_features:
                        manufacturer_features[manufacturer] = {}
                        
                    # Count vehicle features (simplified example)
                    for attr, value in entity_data.get('attributes', {}).items():
                        if attr in ['fuel', 'transmission', 'drive'] and value:
                            if attr not in manufacturer_features[manufacturer]:
                                manufacturer_features[manufacturer][attr] = {}
                            manufacturer_features[manufacturer][attr][value] = \
                                manufacturer_features[manufacturer][attr].get(value, 0) + 1
            
            # Convert to patterns
            for manufacturer, features in manufacturer_features.items():
                for feature_type, values in features.items():
                    total = sum(values.values())
                    for value, count in values.items():
                        support = count / total
                        if support >= min_support:
                            patterns.append({
                                'pattern_type': 'manufacturer_feature',
                                'manufacturer': manufacturer,
                                'feature': feature_type,
                                'value': value,
                                'support': support,
                                'count': count,
                                'total': total
                            })
            
            self.logger.info(f"Generated {len(patterns)} patterns")
            return patterns
            
        except Exception as e:
            error_msg = f"Error generating graph patterns: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            raise InsightsError(error_msg) from e

    def visualize_insights(self, data: pd.DataFrame) -> None:
        """
        Visualize the generated insights from the data.
        
        Args:
            data: Input pandas DataFrame containing the data to visualize
            
        Raises:
            InsightsError: If visualization fails or data is invalid
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            if data is None or data.empty:
                raise InsightsError("No data available for visualization")
                
            # Set style
            sns.set(style="whitegrid")
            
            # 1. Price Distribution
            plt.figure(figsize=(12, 6))
            sns.histplot(data['price'].dropna(), bins=50, kde=True, color='skyblue')
            plt.title('Vehicle Price Distribution', fontsize=14, pad=20)
            plt.xlabel('Price ($)', fontsize=12)
            plt.ylabel('Count', fontsize=12)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
            
            # 2. Vehicle Count by Year (Top 15)
            plt.figure(figsize=(14, 7))
            year_counts = data['year'].value_counts().sort_index(ascending=False).head(15)
            sns.barplot(x=year_counts.index.astype(str), y=year_counts.values, palette='viridis')
            plt.title('Vehicle Count by Year (Top 15)', fontsize=14, pad=20)
            plt.xlabel('Year', fontsize=12)
            plt.ylabel('Count', fontsize=12)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
            
            # 3. Top 10 Manufacturers
            plt.figure(figsize=(12, 6))
            top_manufacturers = data['manufacturer'].value_counts().nlargest(10)
            sns.barplot(x=top_manufacturers.values, y=top_manufacturers.index, palette='mako')
            plt.title('Top 10 Vehicle Manufacturers', fontsize=14, pad=20)
            plt.xlabel('Count', fontsize=12)
            plt.ylabel('Manufacturer', fontsize=12)
            plt.tight_layout()
            plt.show()
            
            # 4. Price by Vehicle Type
            if 'type' in data.columns:
                plt.figure(figsize=(14, 7))
                sns.boxplot(x='type', y='price', data=data, showfliers=False)
                plt.title('Price Distribution by Vehicle Type', fontsize=14, pad=20)
                plt.xlabel('Vehicle Type', fontsize=12)
                plt.ylabel('Price ($)', fontsize=12)
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                plt.show()
                
            # 5. Odometer vs Price
            if 'odometer' in data.columns and 'price' in data.columns:
                plt.figure(figsize=(12, 6))
                sns.scatterplot(x='odometer', y='price', data=data, alpha=0.6, 
                               edgecolor=None, color='green')
                plt.title('Price vs Odometer Reading', fontsize=14, pad=20)
                plt.xlabel('Odometer (miles)', fontsize=12)
                plt.ylabel('Price ($)', fontsize=12)
                plt.tight_layout()
                plt.show()
                
        except ImportError as e:
            error_msg = f"Visualization dependencies not found: {str(e)}"
            self.logger.error(error_msg)
            raise InsightsError(error_msg) from e
            
        except Exception as e:
            error_msg = f"Error generating visualizations: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            raise InsightsError(error_msg) from e

if __name__ == "__main__":
    import logging
    from pathlib import Path
    import sys
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger(__name__)
    
    try:
        # Example usage
        logger.info("Starting Insights Generator example...")
        generator = InsightsGenerator()
        
        # Load sample data
        data_path = Path("../data/vehicles.csv")
        
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path.absolute()}")
            
        logger.info(f"Loading data from {data_path}...")
        data = pd.read_csv(data_path)
        
        if data.empty:
            raise ValueError("Loaded data is empty")
            
        logger.info(f"Successfully loaded {len(data)} records")
        
        # Generate insights
        logger.info("Generating statistical insights...")
        insights = generator.generate_statistical_insights(data)
        print("\n=== Statistical Insights ===")
        print(f"- Total records: {insights['basic_stats']['num_records']:,}")
        print(f"- Total columns: {insights['basic_stats']['num_columns']}")
        print(f"- Duplicate rows: {insights['basic_stats']['duplicate_rows']:,}")
        
        # Create knowledge graph
        logger.info("Creating knowledge graph...")
        generator.create_knowledge_graph(data)
        print("\n=== Knowledge Graph ===")
        print(f"- Total entities: {len(generator.knowledge_graph.entities):,}")
        print(f"- Total relationships: {len(generator.knowledge_graph.relationships):,}")
        
        # Generate patterns
        logger.info("Generating graph patterns...")
        patterns = generator.generate_graph_patterns(min_support=0.3)
        print("\n=== Graph Patterns ===")
        print(f"- Total patterns found: {len(patterns):,}")
        
        # Print top patterns
        if patterns:
            print("\nTop 5 patterns by support:")
            for i, pattern in enumerate(sorted(patterns, key=lambda x: x['support'], reverse=True)[:5], 1):
                print(f"{i}. {pattern['manufacturer']}: {pattern['feature']} = {pattern['value']} "
                      f"(support: {pattern['support']:.1%}, count: {pattern['count']:,}/{pattern['total']:,})")
        
        # Generate visualizations
        logger.info("Generating visualizations...")
        generator.visualize_insights(data)
        
        logger.info("Example completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in example usage: {str(e)}", exc_info=True)
        sys.exit(1)
