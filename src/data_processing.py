import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import os
from sklearn.preprocessing import LabelEncoder
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self, raw_data_dir: str, reference_data_dir: str):
        """
        Initialize the DataProcessor with directory paths.
        
        Args:
            raw_data_dir: Path to directory containing delay data files
            reference_data_dir: Path to directory containing reference files
        """
        self.raw_data_dir = raw_data_dir
        self.reference_data_dir = reference_data_dir
        self.delay_codes = None
        self.label_encoders = {}
        
    def load_delay_codes(self) -> pd.DataFrame:
        """Load and process delay codes reference data."""
        try:
            codes_file = os.path.join(self.reference_data_dir, 'ttc-subway-delay-codes.xlsx')
            # Skip first row if it's empty or contains header info
            self.delay_codes = pd.read_excel(codes_file, skiprows=1)
            # Log the columns to debug
            logger.info(f"Delay codes columns: {self.delay_codes.columns.tolist()}")
            return self.delay_codes
        except Exception as e:
            logger.error(f"Error loading delay codes: {str(e)}")
            raise

    def load_raw_data(self) -> pd.DataFrame:
        """Load and combine all raw delay data files."""
        try:
            all_files = [f for f in os.listdir(self.raw_data_dir) if f.endswith('.xlsx')]
            dfs = []
            
            for file in all_files:
                file_path = os.path.join(self.raw_data_dir, file)
                df = pd.read_excel(file_path)
                dfs.append(df)
                logger.info(f"Loaded data from {file}")
            
            combined_df = pd.concat(dfs, ignore_index=True)
            logger.info(f"Combined {len(all_files)} files, total rows: {len(combined_df)}")
            return combined_df
        except Exception as e:
            logger.error(f"Error loading raw data: {str(e)}")
            raise

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the raw data by handling missing values and outliers.
        
        Args:
            df: Raw DataFrame to clean
            
        Returns:
            Cleaned DataFrame
        """
        try:
            # Make a copy to avoid modifying the original
            df_clean = df.copy()
            
            # Convert Date to datetime if not already
            df_clean['Date'] = pd.to_datetime(df_clean['Date'])
            
            # Convert Time to proper time format and handle potential parsing errors
            df_clean['Time'] = pd.to_datetime(df_clean['Time'].astype(str).str.strip(), format='%H:%M').dt.time
            
            # Remove rows with extreme delays (outliers)
            delay_threshold = df_clean['Min Delay'].quantile(0.99)  # Remove top 1% of delays
            df_clean = df_clean[df_clean['Min Delay'] <= delay_threshold]
            
            # Handle missing values without using inplace
            df_clean = df_clean.assign(
                Bound=df_clean['Bound'].fillna('Unknown'),
                Line=df_clean['Line'].fillna('Unknown')
            )
            
            # Remove any remaining rows with missing values in critical columns
            critical_columns = ['Date', 'Time', 'Station', 'Code', 'Min Delay']
            df_clean = df_clean.dropna(subset=critical_columns)
            
            logger.info(f"Cleaned data: {len(df_clean)} rows remaining")
            return df_clean
        except Exception as e:
            logger.error(f"Error cleaning data: {str(e)}")
            raise

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create new features from existing data.
        
        Args:
            df: Cleaned DataFrame
            
        Returns:
            DataFrame with new features
        """
        try:
            df_featured = df.copy()
            
            # Extract hour directly from time object
            df_featured['Hour'] = df_featured['Time'].apply(lambda x: x.hour)
            
            df_featured['Time_Period'] = pd.cut(
                df_featured['Hour'],
                bins=[0, 6, 12, 18, 24],
                labels=['Night', 'Morning', 'Afternoon', 'Evening']
            )
            
            # Day of week (already present as 'Day' but ensure consistency)
            df_featured['Day_Of_Week'] = df_featured['Date'].dt.day_name()
            
            # Is weekend
            df_featured['Is_Weekend'] = df_featured['Day_Of_Week'].isin(['Saturday', 'Sunday']).astype(int)
            
            # Season
            df_featured['Month'] = df_featured['Date'].dt.month
            df_featured['Season'] = pd.cut(
                df_featured['Month'],
                bins=[0, 3, 6, 9, 12],
                labels=['Winter', 'Spring', 'Summer', 'Fall']
            )
            
            # Create station-specific features
            station_stats = df_featured.groupby('Station').agg({
                'Min Delay': ['mean', 'std']
            }).reset_index()
            station_stats.columns = ['Station', 'Station_Avg_Delay', 'Station_Std_Delay']
            
            # Merge station statistics back
            df_featured = df_featured.merge(station_stats, on='Station', how='left')
            
            # Categorize delays
            df_featured['Delay_Category'] = pd.cut(
                df_featured['Min Delay'],
                bins=[0, 5, 15, 30, float('inf')],
                labels=['Minor', 'Moderate', 'Major', 'Severe']
            )
            
            # If delay codes are loaded, add delay category mapping
            if self.delay_codes is not None:
                try:
                    # First, check what columns are available
                    available_columns = self.delay_codes.columns.tolist()
                    logger.info(f"Available columns in delay_codes: {available_columns}")
                    
                    # Try to find the correct column names by checking content
                    # Assuming the first column contains codes and second contains descriptions
                    delay_codes_mapped = pd.DataFrame({
                        'Code': self.delay_codes.iloc[:, 0],
                        'Category': self.delay_codes.iloc[:, 1]
                    })
                    
                    # Clean up the codes to match the format in the delay data
                    delay_codes_mapped['Code'] = delay_codes_mapped['Code'].astype(str).str.strip()
                    
                    # Merge with delay codes to get categories
                    df_featured = df_featured.merge(
                        delay_codes_mapped,
                        on='Code',
                        how='left'
                    )
                    df_featured['Code_Category'] = df_featured['Category'].fillna('Other')
                    
                except Exception as e:
                    logger.error(f"Error processing delay codes: {str(e)}")
                    df_featured['Code_Category'] = 'Unknown'
            
            logger.info("Added new features to the dataset")
            return df_featured
        except Exception as e:
            logger.error(f"Error engineering features: {str(e)}")
            raise

    def encode_categorical_features(self, df: pd.DataFrame, categorical_columns: Optional[List[str]] = None) -> Tuple[pd.DataFrame, Dict]:
        """
        Encode categorical features using LabelEncoder.
        
        Args:
            df: DataFrame with categorical features
            categorical_columns: List of categorical columns to encode
            
        Returns:
            Tuple of (encoded DataFrame, dictionary of encoders)
        """
        try:
            df_encoded = df.copy()
            
            if categorical_columns is None:
                categorical_columns = [
                    'Station', 'Line', 'Bound', 'Day_Of_Week',
                    'Time_Period', 'Season', 'Delay_Category'
                ]
                if 'Code_Category' in df_encoded.columns:
                    categorical_columns.append('Code_Category')
            
            for column in categorical_columns:
                if column in df_encoded.columns:
                    encoder = LabelEncoder()
                    df_encoded[f'{column}_Encoded'] = encoder.fit_transform(df_encoded[column].astype(str))
                    self.label_encoders[column] = encoder
            
            logger.info(f"Encoded {len(categorical_columns)} categorical features")
            return df_encoded, self.label_encoders
        except Exception as e:
            logger.error(f"Error encoding categorical features: {str(e)}")
            raise

    def process_data(self) -> Tuple[pd.DataFrame, Dict]:
        """
        Execute the complete data processing pipeline.
        
        Returns:
            Tuple of (processed DataFrame, dictionary of encoders)
        """
        try:
            # Load data
            self.load_delay_codes()
            raw_df = self.load_raw_data()
            
            # Process data
            cleaned_df = self.clean_data(raw_df)
            featured_df = self.engineer_features(cleaned_df)
            processed_df, encoders = self.encode_categorical_features(featured_df)
            
            # Create processed directory if it doesn't exist
            os.makedirs('./data/processed', exist_ok=True)
            
            # Save processed data
            processed_df.to_csv('./data/processed/processed_data.csv', index=False)
            logger.info("Completed data processing pipeline")
            
            return processed_df, encoders
        except Exception as e:
            logger.error(f"Error in data processing pipeline: {str(e)}")
            raise

def main():
    """Main function to run the data processing pipeline."""
    try:
        processor = DataProcessor(
            raw_data_dir='./data/raw/delay-data',
            reference_data_dir='./data/raw/reference-data'
        )
        processed_df, encoders = processor.process_data()
        logger.info("Data processing completed successfully")
    except Exception as e:
        logger.error(f"Failed to process data: {str(e)}")
        raise

if __name__ == "__main__":
    main() 