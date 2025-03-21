import pandas as pd
import numpy as np
from sklearn.model_selection import (
    RandomizedSearchCV, train_test_split,
    cross_val_score, KFold, cross_validate
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    classification_report, confusion_matrix,
    mean_squared_error, r2_score, mean_absolute_error,
    make_scorer
)
from sklearn.preprocessing import StandardScaler
import pickle
import logging
from typing import Tuple, Dict, Any, List
import os
from datetime import datetime
from scipy.stats import randint, uniform
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DelayRiskModel:
    def __init__(self, data_path: str = './data/processed/processed_data.csv'):
        """
        Initialize the delay risk model.
        
        Args:
            data_path: Path to the processed data CSV file
        """
        self.data_path = data_path
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.regression_model = None
        self.classification_model = None
        self.scaler = StandardScaler()
        
        # Best parameters from random search
        self.best_regression_params = None
        self.best_classification_params = None
        
        # Define parameter distributions for random search
        self.regression_param_dist = {
            'n_estimators': randint(50, 300),
            'max_depth': [None] + list(range(10, 50, 5)),
            'min_samples_split': randint(2, 20),
            'min_samples_leaf': randint(1, 10),
            'max_features': ['sqrt', 'log2'],
            'bootstrap': [True, False]
        }
        
        self.classification_param_dist = {
            'n_estimators': randint(50, 300),
            'max_depth': [None] + list(range(10, 50, 5)),
            'min_samples_split': randint(2, 20),
            'min_samples_leaf': randint(1, 10),
            'max_features': ['sqrt', 'log2'],
            'bootstrap': [True, False],
            'class_weight': ['balanced', 'balanced_subsample', None]
        }

        # Add CV configuration
        self.cv_folds = 5
        self.cv_results = {
            'regression': None,
            'classification': None
        }
        
        # Add feature importance storage
        self.feature_importance = {
            'regression': None,
            'classification': None
        }
        
        # Create directories for results
        os.makedirs('./reports/cv_results', exist_ok=True)
        os.makedirs('./reports/feature_importance', exist_ok=True)

    def load_data(self) -> pd.DataFrame:
        """Load and prepare the processed data."""
        try:
            self.data = pd.read_csv(self.data_path)
            logger.info(f"Loaded data with shape: {self.data.shape}")
            return self.data
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

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
            
            return numerical_features, categorical_features
        except Exception as e:
            logger.error(f"Error preparing features: {str(e)}")
            raise

    def create_target_variables(self) -> None:
        """Create both regression and classification targets."""
        try:
            # Regression target is the actual delay time
            self.data['regression_target'] = self.data['Min Delay']
            
            # Classification target based on delay severity
            # 0: No/Minor delay (0-5 minutes)
            # 1: Moderate delay (5-15 minutes)
            # 2: Severe delay (>15 minutes)
            self.data['classification_target'] = pd.cut(
                self.data['Min Delay'],
                bins=[-np.inf, 5, 15, np.inf],
                labels=[0, 1, 2]
            )
            
            logger.info("Created regression and classification targets")
        except Exception as e:
            logger.error(f"Error creating target variables: {str(e)}")
            raise

    def prepare_train_test_split(
        self, 
        features: List[str], 
        target: str,
        test_size: float = 0.2
    ) -> None:
        """Prepare train-test split for either regression or classification."""
        try:
            X = self.data[features]
            y = self.data[target]
            
            # Handle missing values
            # For numerical features, fill with median
            numerical_features, _ = self.prepare_features()
            X[numerical_features] = X[numerical_features].fillna(X[numerical_features].median())
            
            # For categorical features, fill with mode
            categorical_features = [f for f in features if f not in numerical_features]
            X[categorical_features] = X[categorical_features].fillna(X[categorical_features].mode().iloc[0])
            
            # Scale numerical features
            X[numerical_features] = self.scaler.fit_transform(X[numerical_features])
            
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            
            logger.info(f"Prepared train-test split for {target}")
            logger.info(f"Training set shape: {self.X_train.shape}")
            logger.info(f"Test set shape: {self.X_test.shape}")
        except Exception as e:
            logger.error(f"Error in train-test split: {str(e)}")
            raise

    def tune_regression_model(self) -> Dict[str, Any]:
        """
        Perform hyperparameter tuning for the regression model using RandomizedSearchCV.
        
        Returns:
            Dictionary containing best parameters
        """
        try:
            logger.info("Starting regression model hyperparameter tuning...")
            start_time = datetime.now()
            
            # Initialize the base model
            base_model = RandomForestRegressor(random_state=42)
            
            # Initialize RandomizedSearchCV with early stopping
            random_search = RandomizedSearchCV(
                estimator=base_model,
                param_distributions=self.regression_param_dist,
                n_iter=50,  # Number of parameter settings to try
                cv=5,
                scoring='neg_root_mean_squared_error',
                n_jobs=-1,
                verbose=1,
                random_state=42,
                return_train_score=True
            )
            
            # Fit RandomizedSearchCV
            random_search.fit(self.X_train, self.y_train)
            
            # Store best parameters
            self.best_regression_params = random_search.best_params_
            
            # Log results
            logger.info(f"Best regression parameters: {random_search.best_params_}")
            logger.info(f"Best regression score (neg_rmse): {random_search.best_score_:.4f}")
            logger.info(f"Tuning time: {datetime.now() - start_time}")
            
            # Initialize the model with best parameters
            self.regression_model = RandomForestRegressor(
                **random_search.best_params_,
                random_state=42
            )
            
            # Early stopping analysis
            cv_results = pd.DataFrame(random_search.cv_results_)
            cv_results['iteration'] = range(len(cv_results))
            cv_results['best_score_so_far'] = cv_results['mean_test_score'].cummax()
            
            # Check if we could have stopped earlier
            tolerance = 0.001  # Define improvement tolerance
            best_iter = cv_results[cv_results['mean_test_score'] == cv_results['mean_test_score'].max()].iloc[0]['iteration']
            logger.info(f"Best iteration found at: {best_iter + 1} out of {len(cv_results)}")
            
            return random_search.best_params_
            
        except Exception as e:
            logger.error(f"Error in regression model tuning: {str(e)}")
            raise

    def tune_classification_model(self) -> Dict[str, Any]:
        """
        Perform hyperparameter tuning for the classification model using RandomizedSearchCV.
        
        Returns:
            Dictionary containing best parameters
        """
        try:
            logger.info("Starting classification model hyperparameter tuning...")
            start_time = datetime.now()
            
            # Initialize the base model
            base_model = RandomForestClassifier(random_state=42)
            
            # Initialize RandomizedSearchCV with early stopping
            random_search = RandomizedSearchCV(
                estimator=base_model,
                param_distributions=self.classification_param_dist,
                n_iter=50,  # Number of parameter settings to try
                cv=5,
                scoring='f1_weighted',
                n_jobs=-1,
                verbose=1,
                random_state=42,
                return_train_score=True
            )
            
            # Fit RandomizedSearchCV
            random_search.fit(self.X_train, self.y_train)
            
            # Store best parameters
            self.best_classification_params = random_search.best_params_
            
            # Log results
            logger.info(f"Best classification parameters: {random_search.best_params_}")
            logger.info(f"Best classification score (f1_weighted): {random_search.best_score_:.4f}")
            logger.info(f"Tuning time: {datetime.now() - start_time}")
            
            # Initialize the model with best parameters
            self.classification_model = RandomForestClassifier(
                **random_search.best_params_,
                random_state=42
            )
            
            # Early stopping analysis
            cv_results = pd.DataFrame(random_search.cv_results_)
            cv_results['iteration'] = range(len(cv_results))
            cv_results['best_score_so_far'] = cv_results['mean_test_score'].cummax()
            
            # Check if we could have stopped earlier
            tolerance = 0.001  # Define improvement tolerance
            best_iter = cv_results[cv_results['mean_test_score'] == cv_results['mean_test_score'].max()].iloc[0]['iteration']
            logger.info(f"Best iteration found at: {best_iter + 1} out of {len(cv_results)}")
            
            return random_search.best_params_
            
        except Exception as e:
            logger.error(f"Error in classification model tuning: {str(e)}")
            raise

    def perform_cross_validation(self) -> Dict[str, Any]:
        """
        Perform comprehensive cross-validation for both models.
        
        Returns:
            Dictionary containing CV results for both models
        """
        logger.info("Starting comprehensive cross-validation analysis...")
        
        # Define scoring metrics for regression
        regression_scoring = {
            'rmse': make_scorer(lambda y, y_pred: -np.sqrt(mean_squared_error(y, y_pred))),
            'mae': make_scorer(mean_absolute_error),
            'r2': make_scorer(r2_score)
        }
        
        # Define scoring metrics for classification
        classification_scoring = {
            'accuracy': 'accuracy',
            'f1_weighted': 'f1_weighted',
            'precision_weighted': 'precision_weighted',
            'recall_weighted': 'recall_weighted'
        }
        
        try:
            # Regression CV
            logger.info("Performing regression cross-validation...")
            reg_cv = cross_validate(
                self.regression_model,
                self.X_train,
                self.y_train,
                cv=self.cv_folds,
                scoring=regression_scoring,
                return_train_score=True,
                n_jobs=-1
            )
            
            # Classification CV
            logger.info("Performing classification cross-validation...")
            clf_cv = cross_validate(
                self.classification_model,
                self.X_train,
                self.y_train,
                cv=self.cv_folds,
                scoring=classification_scoring,
                return_train_score=True,
                n_jobs=-1
            )
            
            # Store results
            self.cv_results['regression'] = reg_cv
            self.cv_results['classification'] = clf_cv
            
            # Log CV results
            self._log_cv_results()
            
            # Generate CV visualizations
            self._plot_cv_results()
            
            return {
                'regression': reg_cv,
                'classification': clf_cv
            }
            
        except Exception as e:
            logger.error(f"Error in cross-validation: {str(e)}")
            raise
            
    def _log_cv_results(self):
        """Log detailed cross-validation results"""
        logger.info("\nRegression CV Results:")
        for metric in ['rmse', 'mae', 'r2']:
            test_scores = self.cv_results['regression'][f'test_{metric}']
            logger.info(f"{metric.upper()}:")
            logger.info(f"  Mean: {np.mean(test_scores):.4f}")
            logger.info(f"  Std: {np.std(test_scores):.4f}")
            logger.info(f"  Min: {np.min(test_scores):.4f}")
            logger.info(f"  Max: {np.max(test_scores):.4f}")
        
        logger.info("\nClassification CV Results:")
        for metric in ['accuracy', 'f1_weighted', 'precision_weighted', 'recall_weighted']:
            test_scores = self.cv_results['classification'][f'test_{metric}']
            logger.info(f"{metric.upper()}:")
            logger.info(f"  Mean: {np.mean(test_scores):.4f}")
            logger.info(f"  Std: {np.std(test_scores):.4f}")
            logger.info(f"  Min: {np.min(test_scores):.4f}")
            logger.info(f"  Max: {np.max(test_scores):.4f}")
    
    def _plot_cv_results(self):
        """Generate visualizations for cross-validation results"""
        # Set style - using a default style instead of seaborn
        plt.style.use('default')
        
        # Create figure with white background
        fig, axes = plt.subplots(2, 1, figsize=(12, 10), facecolor='white')
        
        # Regression metrics boxplot
        reg_data = []
        reg_labels = []
        metrics = ['rmse', 'mae', 'r2']
        for metric in metrics:
            reg_data.append(self.cv_results['regression'][f'test_{metric}'])
            reg_labels.extend([metric.upper()] * self.cv_folds)
        
        axes[0].boxplot(reg_data, labels=[m.upper() for m in metrics])
        axes[0].set_title('Regression Metrics Across CV Folds')
        axes[0].set_ylabel('Score')
        axes[0].grid(True, linestyle='--', alpha=0.7)
        
        # Classification metrics boxplot
        clf_data = []
        clf_labels = []
        metrics = ['accuracy', 'f1_weighted', 'precision_weighted', 'recall_weighted']
        for metric in metrics:
            clf_data.append(self.cv_results['classification'][f'test_{metric}'])
            clf_labels.extend([metric.upper()] * self.cv_folds)
        
        axes[1].boxplot(clf_data, labels=[m.upper() for m in metrics])
        axes[1].set_title('Classification Metrics Across CV Folds')
        axes[1].set_ylabel('Score')
        axes[1].grid(True, linestyle='--', alpha=0.7)
        
        # Improve layout
        plt.tight_layout()
        
        # Save plot
        os.makedirs('./reports/cv_results', exist_ok=True)
        plt.savefig('./reports/cv_results/cv_metrics_boxplot.png', bbox_inches='tight', dpi=300)
        plt.close()
        
        logger.info("CV visualization saved to './reports/cv_results/cv_metrics_boxplot.png'")
    
    def analyze_feature_importance(self) -> Dict[str, pd.DataFrame]:
        """
        Analyze and visualize feature importance for both models.
        
        Returns:
            Dictionary containing feature importance DataFrames for both models
        """
        try:
            logger.info("Analyzing feature importance...")
            
            # Get feature names
            numerical_features, categorical_features = self.prepare_features()
            feature_names = numerical_features + categorical_features
            
            # Analyze regression model feature importance
            reg_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': self.regression_model.feature_importances_
            })
            reg_importance = reg_importance.sort_values('importance', ascending=False)
            
            # Analyze classification model feature importance
            clf_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': self.classification_model.feature_importances_
            })
            clf_importance = clf_importance.sort_values('importance', ascending=False)
            
            # Store results
            self.feature_importance['regression'] = reg_importance
            self.feature_importance['classification'] = clf_importance
            
            # Log top features
            self._log_feature_importance()
            
            # Create visualizations
            self._plot_feature_importance()
            
            # Save detailed results
            reg_importance.to_csv('./reports/feature_importance/regression_importance.csv', index=False)
            clf_importance.to_csv('./reports/feature_importance/classification_importance.csv', index=False)
            
            return {
                'regression': reg_importance,
                'classification': clf_importance
            }
            
        except Exception as e:
            logger.error(f"Error in feature importance analysis: {str(e)}")
            raise

    def _log_feature_importance(self):
        """Log detailed feature importance results"""
        logger.info("\nTop 5 Important Features for Regression Model:")
        reg_top5 = self.feature_importance['regression'].head()
        for _, row in reg_top5.iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")
        
        logger.info("\nTop 5 Important Features for Classification Model:")
        clf_top5 = self.feature_importance['classification'].head()
        for _, row in clf_top5.iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")

    def _plot_feature_importance(self):
        """Generate visualizations for feature importance"""
        plt.style.use('default')
        
        # Create figure with two subplots
        fig, axes = plt.subplots(2, 1, figsize=(12, 16), facecolor='white')
        
        # Plot regression feature importance
        reg_importance = self.feature_importance['regression']
        axes[0].barh(reg_importance['feature'], reg_importance['importance'])
        axes[0].set_title('Feature Importance - Regression Model')
        axes[0].set_xlabel('Importance Score')
        axes[0].grid(True, linestyle='--', alpha=0.7)
        
        # Plot classification feature importance
        clf_importance = self.feature_importance['classification']
        axes[1].barh(clf_importance['feature'], clf_importance['importance'])
        axes[1].set_title('Feature Importance - Classification Model')
        axes[1].set_xlabel('Importance Score')
        axes[1].grid(True, linestyle='--', alpha=0.7)
        
        # Improve layout
        plt.tight_layout()
        
        # Save plot
        plt.savefig('./reports/feature_importance/feature_importance.png', 
                    bbox_inches='tight', dpi=300)
        plt.close()
        
        logger.info("Feature importance visualization saved to './reports/feature_importance/feature_importance.png'")

    def train_models(self):
        """Train models and perform comprehensive analysis"""
        try:
            # Get features
            numerical_features, categorical_features = self.prepare_features()
            all_features = numerical_features + categorical_features
            
            # Prepare regression data
            logger.info("Preparing regression data...")
            self.prepare_train_test_split(all_features, 'regression_target')
            
            # Tune and train regression model
            logger.info("Training regression model...")
            self.tune_regression_model()
            self.regression_model.fit(self.X_train, self.y_train)
            
            # Prepare classification data
            logger.info("Preparing classification data...")
            self.prepare_train_test_split(all_features, 'classification_target')
            
            # Tune and train classification model
            logger.info("Training classification model...")
            self.tune_classification_model()
            self.classification_model.fit(self.X_train, self.y_train)
            
            # Perform cross-validation
            logger.info("Performing cross-validation analysis...")
            self.perform_cross_validation()
            
            # Analyze feature importance
            logger.info("Analyzing feature importance...")
            self.analyze_feature_importance()
            
            # Save models and results
            self.save_models()
            
            logger.info("Model training and analysis complete!")
            
        except Exception as e:
            logger.error(f"Error in model training: {str(e)}")
            raise

    def calculate_risk_score(self, features: pd.DataFrame) -> Tuple[float, int, float]:
        """
        Calculate a sophisticated risk score (0-100) based on multiple factors.
        
        Args:
            features: DataFrame with feature values
            
        Returns:
            Tuple of (risk_score, risk_category, predicted_delay)
        """
        try:
            # Scale numerical features
            numerical_features, _ = self.prepare_features()
            features_scaled = features.copy()
            features_scaled[numerical_features] = self.scaler.transform(features_scaled[numerical_features])
            
            # Get predictions from both models
            predicted_delay = self.regression_model.predict(features_scaled)[0]
            risk_category = self.classification_model.predict(features_scaled)[0]
            risk_probabilities = self.classification_model.predict_proba(features_scaled)[0]
            
            # 1. Delay-based score (0-40 points)
            # Normalize delay to a 0-1 scale (assuming max delay of 30 minutes)
            delay_score = min(predicted_delay / 30.0, 1.0) * 40
            
            # 2. Risk category confidence score (0-30 points)
            # Higher confidence in severe delays contributes more to risk
            category_weights = {0: 0.2, 1: 0.5, 2: 0.8}  # Weights for each risk category
            confidence_score = risk_probabilities[risk_category] * category_weights[risk_category] * 30
            
            # 3. Time-based risk factors (0-15 points)
            hour = features['Hour'].iloc[0]
            is_weekend = features['Is_Weekend'].iloc[0]
            
            # Peak hour risk (rush hours)
            if 7 <= hour <= 9 or 16 <= hour <= 18:  # Rush hours
                time_score = 10
            elif 9 < hour < 16:  # Regular hours
                time_score = 5
            else:  # Off-peak hours
                time_score = 2
            
            # Weekend adjustment
            if is_weekend:
                time_score *= 0.7  # Reduce weekend risk
            
            # 4. Station-specific risk (0-15 points)
            station = features['Station'].iloc[0]
            station_stats = self.metadata.get('station_stats', {}).get(station, {})
            
            if station_stats:
                # Use station's historical delay statistics
                station_avg_delay = station_stats.get('mean', 0)
                station_std_delay = station_stats.get('std', 0)
                
                # Calculate station risk based on historical performance
                station_risk = min((station_avg_delay + station_std_delay) / 20.0, 1.0) * 15
            else:
                station_risk = 7.5  # Default medium risk for unknown stations
            
            # Combine all risk components
            risk_score = delay_score + confidence_score + time_score + station_risk
            
            # Ensure score is between 0 and 100
            risk_score = max(0, min(100, risk_score))
            
            # Log risk score components for transparency
            logger.info(f"\nRisk Score Components:")
            logger.info(f"  Delay Score: {delay_score:.2f}")
            logger.info(f"  Confidence Score: {confidence_score:.2f}")
            logger.info(f"  Time Score: {time_score:.2f}")
            logger.info(f"  Station Risk: {station_risk:.2f}")
            logger.info(f"  Total Risk Score: {risk_score:.2f}")
            
            return risk_score, risk_category, predicted_delay
            
        except Exception as e:
            logger.error(f"Error calculating risk score: {str(e)}")
            raise

    def save_models(self, output_dir: str = './models') -> None:
        """Save trained models, scaler, and analysis results"""
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # Save models using pickle
            with open(f"{output_dir}/regression_model.pkl", 'wb') as f:
                pickle.dump(self.regression_model, f)
            
            with open(f"{output_dir}/classification_model.pkl", 'wb') as f:
                pickle.dump(self.classification_model, f)
            
            with open(f"{output_dir}/scaler.pkl", 'wb') as f:
                pickle.dump(self.scaler, f)
            
            # Save feature names and other metadata
            metadata = {
                'numerical_features': self.prepare_features()[0],
                'categorical_features': self.prepare_features()[1],
                'model_version': '1.0.0',
                'training_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                'best_regression_params': self.best_regression_params,
                'best_classification_params': self.best_classification_params,
                'feature_importance': self.feature_importance,
                'label_encoders': {
                    'Station': LabelEncoder().fit(self.data['Station'].unique()),
                    'Line': LabelEncoder().fit(self.data['Line'].unique()),
                    'Bound': LabelEncoder().fit(self.data['Bound'].unique()),
                    'Day_Of_Week': LabelEncoder().fit(self.data['Day_Of_Week'].unique()),
                    'Time_Period': LabelEncoder().fit(self.data['Time_Period'].unique()),
                    'Season': LabelEncoder().fit(self.data['Season'].unique())
                }
            }
            
            with open(f"{output_dir}/model_metadata.pkl", 'wb') as f:
                pickle.dump(metadata, f)
            
            logger.info(f"Saved models, metadata, and analysis results to {output_dir}")
        except Exception as e:
            logger.error(f"Error saving models: {str(e)}")
            raise

    def load_models(self, model_dir: str = './models') -> None:
        """Load trained models and scaler using pickle."""
        try:
            # Load models using pickle
            with open(f"{model_dir}/regression_model.pkl", 'rb') as f:
                self.regression_model = pickle.load(f)
            
            with open(f"{model_dir}/classification_model.pkl", 'rb') as f:
                self.classification_model = pickle.load(f)
            
            with open(f"{model_dir}/scaler.pkl", 'rb') as f:
                self.scaler = pickle.load(f)
            
            # Load metadata
            with open(f"{model_dir}/model_metadata.pkl", 'rb') as f:
                metadata = pickle.load(f)
                logger.info(f"Loaded model version: {metadata['model_version']}")
                logger.info(f"Model training date: {metadata['training_date']}")
            
            logger.info(f"Loaded models from {model_dir}")
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            raise

def main():
    """Main function to train and evaluate the models."""
    try:
        # Initialize and train models
        model = DelayRiskModel()
        model.load_data()
        model.create_target_variables()
        
        # Train both models (now includes hyperparameter tuning)
        model.train_models()
        
        logger.info("Model training completed successfully")
    except Exception as e:
        logger.error(f"Failed to train models: {str(e)}")
        raise

if __name__ == "__main__":
    main() 