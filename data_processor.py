import os
import sys
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, TypeVar

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('knowledge_insights.log')
    ]
)
logger = logging.getLogger(__name__)

# Type variable for DataFrame
df = TypeVar('pandas.core.frame.DataFrame')

class DataProcessingError(Exception):
    """Custom exception for data processing errors."""
    pass

class DataProcessor:
    def __init__(self):
        self.data = None
        self.processed_data = None
        self.metadata = {}

    def load_data(self, file_path: str, file_type: str = 'csv', chunksize: int = 10000) -> None:
        """
        Load data from various file formats with chunked processing for large files.

        Args:
            file_path (str): Path to the input file
            file_type (str, optional): Type of the file ('csv', 'excel', 'json'). 
                                     Defaults to 'csv'.
            chunksize (int, optional): Number of rows to process at a time. 
                                    If None, loads entire file at once.

        Raises:
            FileNotFoundError: If the input file doesn't exist
            ValueError: If the file type is not supported
            DataProcessingError: For other file reading errors
        """
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
                
            logger.info(f"Loading {file_type.upper()} file from {file_path}")
            
            chunks = []
            total_rows = 0
            
            # Get file size for progress estimation
            file_size = os.path.getsize(file_path)
            processed_size = 0
            
            if file_type == 'csv':
                # Use chunked reading for large files
                if chunksize and os.path.getsize(file_path) > 10 * 1024 * 1024:  # 10MB threshold
                    logger.info("Large file detected, using chunked processing...")
                    for chunk in pd.read_csv(file_path, chunksize=chunksize, 
                                          encoding='utf-8', 
                                          engine='python'):
                        chunks.append(chunk)
                        processed_size += chunk.memory_usage(deep=True).sum()
                        progress = min(100, int(processed_size / file_size * 100))
                        logger.info(f"Loading data: {progress}% complete")
                    self.data = pd.concat(chunks, ignore_index=True)
                else:
                    self.data = pd.read_csv(file_path, encoding='utf-8', 
                                         engine='python')
            
            elif file_type == 'excel':
                # For Excel, we can't chunk natively, but can read specific sheets
                self.data = pd.read_excel(file_path, engine='openpyxl')
                
            elif file_type == 'json':
                # For JSON, read in chunks if it's a JSON lines file
                if chunksize and os.path.getsize(file_path) > 10 * 1024 * 1024:  # 10MB threshold
                    logger.info("Large JSON file detected, using chunked processing...")
                    chunks = []
                    with open(file_path, 'r', encoding='utf-8') as f:
                        for i, line in enumerate(f):
                            chunks.append(pd.read_json(line, lines=True, typ='series').to_frame().T)
                            if i > 0 and i % chunksize == 0:
                                logger.info(f"Processed {i} rows...")
                    self.data = pd.concat(chunks, ignore_index=True)
                else:
                    self.data = pd.read_json(file_path, lines=True)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
                
            if self.data.empty:
                logger.warning("Loaded an empty dataset")
            else:
                logger.info(f"Successfully loaded {len(self.data)} records")
                logger.info(f"Memory usage: {self.data.memory_usage(deep=True).sum() / (1024*1024):.2f} MB")
            
        except Exception as e:
            logger.error(f"Error loading {file_type} file: {str(e)}", exc_info=True)
            raise DataProcessingError(f"Failed to load {file_type} file: {str(e)}")

    def clean_data(self) -> None:
        """
        Perform data cleaning operations including:
        1. Removing duplicate rows
        2. Handling missing values
        3. Converting data types
        4. Standardizing text data
        
        Raises:
            DataProcessingError: If data cleaning fails
        """
        try:
            if self.data is None or self.data.empty:
                raise DataProcessingError("No data available for cleaning")
                
            initial_count = len(self.data)
            
            # Remove duplicates
            self.data = self.data.drop_duplicates()
            dup_count = initial_count - len(self.data)
            if dup_count > 0:
                logger.info(f"Removed {dup_count} duplicate rows")
            
            # Handle missing values - only remove rows where all values are missing
            rows_before = len(self.data)
            self.data = self.data.dropna(how='all')
            rows_after = len(self.data)
            
            if rows_before > rows_after:
                logger.info(f"Removed {rows_before - rows_after} rows where all values were missing")
                
            # For columns with too many missing values, we'll fill them with appropriate defaults
            for col in self.data.columns:
                missing_count = self.data[col].isnull().sum()
                if missing_count > 0:
                    if self.data[col].dtype == 'object':
                        # For categorical/string columns, fill with 'Unknown'
                        self.data[col] = self.data[col].fillna('Unknown')
                    else:
                        # For numeric columns, fill with median
                        self.data[col] = self.data[col].fillna(self.data[col].median())
            
            # Convert data types
            self._convert_data_types()
            
            logger.info(f"Data cleaning complete. Final record count: {len(self.data)}")
            
        except Exception as e:
            logger.error(f"Error during data cleaning: {str(e)}")
            raise DataProcessingError(f"Data cleaning failed: {str(e)}")
            
    def _convert_data_types(self) -> None:
        """
        Convert column data types to appropriate types.
        """
        for col in self.data.columns:
            # Skip if no data
            if self.data[col].isnull().all():
                continue
                
            # Try to convert to numeric if possible
            if self.data[col].dtype == 'object':
                try:
                    self.data[col] = pd.to_numeric(self.data[col], errors='ignore')
                except Exception as e:
                    # If conversion to numeric fails, keep as string
                    pass
                    
            # Convert potential datetime columns
            if self.data[col].dtype == 'object':
                try:
                    self.data[col] = pd.to_datetime(self.data[col], errors='ignore')
                except:
                    pass

    def generate_metadata(self) -> Dict[str, Any]:
        """
        Generate comprehensive metadata about the dataset.
        
        Returns:
            Dict containing dataset metadata including:
            - Basic statistics (row count, column count)
            - Column data types
            - Missing value counts
            - Basic statistics for numerical columns
            
        Raises:
            DataProcessingError: If no data is available
        """
        if self.data is None or self.data.empty:
            raise DataProcessingError("No data available to generate metadata")
            
        try:
            metadata = {
                'dataset_info': {
                    'num_rows': len(self.data),
                    'num_columns': len(self.data.columns),
                    'total_missing_values': int(self.data.isnull().sum().sum()),
                    'duplicate_rows': int(self.data.duplicated().sum())
                },
                'column_info': {},
                'data_types': {}
            }
            
            # Generate column-wise metadata
            for col in self.data.columns:
                col_metadata = {
                    'data_type': str(self.data[col].dtype),
                    'missing_values': int(self.data[col].isnull().sum()),
                    'unique_values': int(self.data[col].nunique())
                }
                
                # Add statistics for numerical columns
                if pd.api.types.is_numeric_dtype(self.data[col]):
                    col_metadata.update({
                        'min': float(self.data[col].min()) if not self.data[col].isnull().all() else None,
                        'max': float(self.data[col].max()) if not self.data[col].isnull().all() else None,
                        'mean': float(self.data[col].mean()) if not self.data[col].isnull().all() else None,
                        'median': float(self.data[col].median()) if not self.data[col].isnull().all() else None,
                        'std': float(self.data[col].std()) if len(self.data[col]) > 1 else None
                    })
                
                metadata['column_info'][col] = col_metadata
                metadata['data_types'][col] = str(self.data[col].dtype)
            
            self.metadata = metadata
            return self.metadata
            
        except Exception as e:
            logger.error(f"Error generating metadata: {str(e)}")
            raise DataProcessingError(f"Failed to generate metadata: {str(e)}")

    def save_processed_data(self, output_path: str, file_type: str = 'csv') -> None:
        """
        Save the processed data to a file.
        
        Args:
            output_path (str): Path where to save the file
            file_type (str, optional): Output file type ('csv', 'parquet', 'json'). 
                                     Defaults to 'csv'.
                                      
        Raises:
            DataProcessingError: If saving fails
        """
        try:
            if self.data is None or self.data.empty:
                raise DataProcessingError("No data available to save")
                
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            
            logger.info(f"Saving processed data to {output_path}")
            
            if file_type == 'csv':
                self.data.to_csv(output_path, index=False)
            elif file_type == 'parquet':
                self.data.to_parquet(output_path, index=False)
            elif file_type == 'json':
                self.data.to_json(output_path, orient='records', lines=True)
            else:
                raise ValueError(f"Unsupported output file type: {file_type}")
                
            logger.info(f"Successfully saved {len(self.data)} records to {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving processed data: {str(e)}")
            raise DataProcessingError(f"Failed to save processed data: {str(e)}")

if __name__ == "__main__":
    # Example usage
    processor = DataProcessor()
    processor.load_data('data/input_data.csv')
    processor.clean_data()
    metadata = processor.generate_metadata()
    processor.save_processed_data('data/processed_data.csv')
