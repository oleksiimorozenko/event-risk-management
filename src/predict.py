import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pickle
import logging
from typing import Dict, List, Tuple, Optional, Any
import os
from model_training import DelayRiskModel

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DelayPredictor:
    def __init__(self, model_dir: str = './models'):
        """
        Initialize the delay predictor.
        
        Args:
            model_dir: Directory containing trained models and metadata
        """
        self.model_dir = model_dir
        self.regression_model = None
        self.classification_model = None
        self.scaler = None
        self.metadata = None
        self.load_models()
        
    def load_models(self) -> None:
        """Load trained models, scaler, and metadata."""
        try:
            # Load models
            with open(f"{self.model_dir}/regression_model.pkl", 'rb') as f:
                self.regression_model = pickle.load(f)
            
            with open(f"{self.model_dir}/classification_model.pkl", 'rb') as f:
                self.classification_model = pickle.load(f)
            
            with open(f"{self.model_dir}/scaler.pkl", 'rb') as f:
                self.scaler = pickle.load(f)
            
            with open(f"{self.model_dir}/model_metadata.pkl", 'rb') as f:
                self.metadata = pickle.load(f)
            
            logger.info("Successfully loaded models and metadata")
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            raise
    
    def prepare_prediction_data(
        self,
        date: str,
        time: str,
        station: str,
        line: str,
        bound: str,
        day_of_week: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Prepare input data for prediction.
        
        Args:
            date: Date string in 'YYYY-MM-DD' format
            time: Time string in 'HH:MM' format
            station: Station name
            line: Line code (e.g., 'YU', 'BD', 'SRT')
            bound: Direction ('N', 'S', 'E', 'W')
            day_of_week: Optional day of week (if not provided, will be calculated from date)
            
        Returns:
            DataFrame prepared for prediction
        """
        try:
            # Convert date and time to datetime
            dt = datetime.strptime(f"{date} {time}", "%Y-%m-%d %H:%M")
            
            # Calculate day of week if not provided
            if day_of_week is None:
                day_of_week = dt.strftime('%A')
            
            # Extract hour
            hour = dt.hour
            
            # Determine time period
            time_period = pd.cut(
                [hour],
                bins=[0, 6, 12, 18, 24],
                labels=['Night', 'Morning', 'Afternoon', 'Evening']
            )[0]
            
            # Determine season based on month
            month = dt.month
            season = pd.cut(
                [month],
                bins=[0, 3, 6, 9, 12],
                labels=['Winter', 'Spring', 'Summer', 'Fall']
            )[0]
            
            # Create base DataFrame
            df = pd.DataFrame({
                'Date': [date],
                'Time': [time],
                'Station': [station],
                'Line': [line],
                'Bound': [bound],
                'Day_Of_Week': [day_of_week],
                'Hour': [hour],
                'Time_Period': [time_period],
                'Season': [season],
                'Is_Weekend': [1 if day_of_week in ['Saturday', 'Sunday'] else 0]
            })
            
            # Add station statistics from metadata
            station_stats = self.metadata.get('station_stats', {})
            if station in station_stats:
                df['Station_Avg_Delay'] = station_stats[station]['mean']
                df['Station_Std_Delay'] = station_stats[station]['std']
            else:
                # Use default values if station not found
                df['Station_Avg_Delay'] = 0
                df['Station_Std_Delay'] = 0
            
            # Encode categorical features
            for col in ['Station', 'Line', 'Bound', 'Day_Of_Week', 'Time_Period', 'Season']:
                if col in self.metadata['label_encoders']:
                    encoder = self.metadata['label_encoders'][col]
                    df[f'{col}_Encoded'] = encoder.transform([df[col].iloc[0]])
            
            return df
            
        except Exception as e:
            logger.error(f"Error preparing prediction data: {str(e)}")
            raise
    
    def predict_delay(
        self,
        *,
        date: str,
        time: str,
        station: str,
        line: str,
        bound: str,
        day_of_week: Optional[str] = None,
        is_weekend: Optional[bool] = None,
        time_period: Optional[str] = None,
        season: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Predict delay for a specific scenario.
        
        Args:
            date: Date string in 'YYYY-MM-DD' format
            time: Time string in 'HH:MM' format
            station: Station name
            line: Line code (e.g., 'YU', 'BD', 'SRT')
            bound: Direction ('N', 'S', 'E', 'W')
            day_of_week: Optional day of week (if not provided, will be calculated from date)
            is_weekend: Optional boolean indicating if it's a weekend (if not provided, will be calculated from date)
            time_period: Optional time period ('Night', 'Morning', 'Afternoon', 'Evening')
            season: Optional season ('Winter', 'Spring', 'Summer', 'Fall')
            
        Returns:
            Dictionary containing predictions and risk assessment
        """
        try:
            # Convert date and time to datetime
            dt = datetime.strptime(f"{date} {time}", "%Y-%m-%d %H:%M")
            
            # Calculate day of week if not provided
            if day_of_week is None:
                day_of_week = dt.strftime('%A')
            
            # Calculate is_weekend if not provided
            if is_weekend is None:
                is_weekend = day_of_week in ['Saturday', 'Sunday']
            
            # Extract hour
            hour = dt.hour
            
            # Determine time period if not provided
            if time_period is None:
                time_period = pd.cut(
                    [hour],
                    bins=[0, 6, 12, 18, 24],
                    labels=['Night', 'Morning', 'Afternoon', 'Evening']
                )[0]
            
            # Determine season if not provided
            if season is None:
                month = dt.month
                season = pd.cut(
                    [month],
                    bins=[0, 3, 6, 9, 12],
                    labels=['Winter', 'Spring', 'Summer', 'Fall']
                )[0]
            
            # Create base DataFrame
            df = pd.DataFrame({
                'Date': [date],
                'Time': [time],
                'Station': [station],
                'Line': [line],
                'Bound': [bound],
                'Day_Of_Week': [day_of_week],
                'Hour': [hour],
                'Time_Period': [time_period],
                'Season': [season],
                'Is_Weekend': [1 if is_weekend else 0]
            })
            
            # Add station statistics from metadata
            station_stats = self.metadata.get('station_stats', {})
            if station in station_stats:
                df['Station_Avg_Delay'] = station_stats[station]['mean']
                df['Station_Std_Delay'] = station_stats[station]['std']
            else:
                # Use default values if station not found
                df['Station_Avg_Delay'] = 0
                df['Station_Std_Delay'] = 0
            
            # Encode categorical features
            for col in ['Station', 'Line', 'Bound', 'Day_Of_Week', 'Time_Period', 'Season']:
                if col in self.metadata['label_encoders']:
                    encoder = self.metadata['label_encoders'][col]
                    df[f'{col}_Encoded'] = encoder.transform([df[col].iloc[0]])
            
            # Get feature lists
            numerical_features, categorical_features = self.prepare_features()
            
            # Filter features to only include those that exist in the DataFrame
            available_features = [f for f in numerical_features + categorical_features if f in df.columns]
            
            # Prepare features for prediction
            X = df[available_features]
            
            # Scale numerical features that exist in the DataFrame
            numerical_features = [f for f in numerical_features if f in df.columns]
            if numerical_features:
                X[numerical_features] = self.scaler.transform(X[numerical_features])
            
            # Make predictions
            predicted_delay = self.regression_model.predict(X)[0]
            risk_category = self.classification_model.predict(X)[0]
            risk_probs = self.classification_model.predict_proba(X)[0]
            
            # Calculate risk score
            risk_score = self.calculate_risk_score(predicted_delay, risk_category, risk_probs)
            
            # Prepare result dictionary
            result = {
                'predicted_delay_minutes': round(predicted_delay, 2),
                'risk_category': int(risk_category),
                'risk_category_label': ['No/Minor', 'Moderate', 'Severe'][int(risk_category)],
                'risk_score': round(risk_score, 2),
                'risk_probabilities': {
                    'No/Minor': round(risk_probs[0] * 100, 2),
                    'Moderate': round(risk_probs[1] * 100, 2),
                    'Severe': round(risk_probs[2] * 100, 2)
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            raise
    
    def calculate_risk_score(
        self,
        predicted_delay: float,
        risk_category: int,
        risk_probs: np.ndarray
    ) -> float:
        """
        Calculate a combined risk score.
        
        Args:
            predicted_delay: Predicted delay in minutes
            risk_category: Predicted risk category (0, 1, or 2)
            risk_probs: Probability distribution over risk categories
            
        Returns:
            Risk score between 0 and 100
        """
        # Normalize delay to 0-50 scale (assuming max delay of 30 minutes)
        delay_score = min(predicted_delay / 30 * 50, 50)
        
        # Category score based on probabilities
        category_score = np.sum(risk_probs * np.array([0, 25, 50]))
        
        # Combine scores
        risk_score = delay_score + category_score
        
        return min(risk_score, 100)
    
    def predict_batch(
        self,
        scenarios: List[Dict[str, str]]
    ) -> List[Dict[str, Any]]:
        """
        Predict delays for multiple scenarios.
        
        Args:
            scenarios: List of dictionaries containing prediction scenarios
            
        Returns:
            List of prediction results
        """
        results = []
        for scenario in scenarios:
            try:
                result = self.predict_delay(**scenario)
                results.append(result)
            except Exception as e:
                logger.error(f"Error predicting for scenario {scenario}: {str(e)}")
                results.append({'error': str(e)})
        return results

    def prepare_features(self) -> Tuple[List[str], List[str]]:
        """Prepare feature lists for modeling."""
        try:
            # Numerical features
            numerical_features = [
                'Hour', 'Is_Weekend', 
                'Station_Avg_Delay', 'Station_Std_Delay'
            ]
            
            # Encoded categorical features
            categorical_features = [
                'Station_Encoded', 'Line_Encoded', 
                'Bound_Encoded', 'Day_Of_Week_Encoded',
                'Time_Period_Encoded', 'Season_Encoded'
            ]
            
            # Filter out any features that don't exist in the data
            available_features = set(self.metadata.get('categorical_features', []))
            categorical_features = [f for f in categorical_features if f in available_features]
            
            return numerical_features, categorical_features
        except Exception as e:
            logger.error(f"Error preparing features: {str(e)}")
            raise

def main():
    """Example usage of the DelayPredictor with command-line arguments."""
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description='Predict subway delays using trained models.')
    
    # Required arguments
    parser.add_argument('--date', type=str, required=True,
                      help='Date in YYYY-MM-DD format')
    parser.add_argument('--time', type=str, required=True,
                      help='Time in HH:MM format')
    parser.add_argument('--station', type=str, required=True,
                      help='Station name (e.g., FINCH STATION)')
    parser.add_argument('--line', type=str, required=True,
                      choices=['YU', 'BD', 'SRT'],
                      help='Line code (YU, BD, or SRT)')
    parser.add_argument('--bound', type=str, required=True,
                      choices=['N', 'S', 'E', 'W'],
                      help='Direction (N, S, E, or W)')
    
    # Optional arguments
    parser.add_argument('--day-of-week', type=str,
                      choices=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
                      help='Day of week (if not provided, will be calculated from date)')
    parser.add_argument('--is-weekend', type=bool,
                      help='Whether it is a weekend (if not provided, will be calculated from date)')
    parser.add_argument('--time-period', type=str,
                      choices=['Night', 'Morning', 'Afternoon', 'Evening'],
                      help='Time period (if not provided, will be calculated from time)')
    parser.add_argument('--season', type=str,
                      choices=['Winter', 'Spring', 'Summer', 'Fall'],
                      help='Season (if not provided, will be calculated from date)')
    parser.add_argument('--model-dir', type=str, default='./models',
                      help='Directory containing trained models (default: ./models)')
    
    # Show help if no arguments provided
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)
    
    args = parser.parse_args()
    
    try:
        # Initialize predictor
        predictor = DelayPredictor(model_dir=args.model_dir)
        
        # Prepare prediction arguments
        prediction_args = {
            'date': args.date,
            'time': args.time,
            'station': args.station,
            'line': args.line,
            'bound': args.bound
        }
        
        # Add optional arguments if provided
        if args.day_of_week:
            prediction_args['day_of_week'] = args.day_of_week
        if args.is_weekend is not None:
            prediction_args['is_weekend'] = args.is_weekend
        if args.time_period:
            prediction_args['time_period'] = args.time_period
        if args.season:
            prediction_args['season'] = args.season
        
        # Make prediction
        result = predictor.predict_delay(**prediction_args)
        
        # Print results
        logger.info("\nPrediction Result:")
        logger.info(f"Predicted Delay: {result['predicted_delay_minutes']} minutes")
        logger.info(f"Risk Category: {result['risk_category_label']}")
        logger.info(f"Risk Score: {result['risk_score']}")
        logger.info("\nRisk Probabilities:")
        for category, prob in result['risk_probabilities'].items():
            logger.info(f"  {category}: {prob}%")
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main() 