# Subway delay risk management system

A comprehensive system for predicting and assessing subway delays in Toronto's TTC subway system.

## Overview

This system provides real-time predictions of subway delays and risk assessments using machine learning models. It processes historical delay data, trains predictive models, and offers multiple interfaces for making predictions.

## Features

- **Data Processing Pipeline**: Handles raw delay data, performs feature engineering, and prepares data for model training
- **Dual Model Approach**: 
  - Regression model for predicting delay duration
  - Classification model for risk category assessment
- **Risk Scoring System**: Combines multiple factors to provide a comprehensive risk score (0-100)
- **Multiple Interfaces**:
  - Command-line interface
  - Web interface
  - API endpoint

## Risk categories

The system categorizes delays into three risk levels:
1. **No/Minor** (0): 0-5 minutes delay
2. **Moderate** (1): 5-15 minutes delay
3. **Severe** (2): >15 minutes delay

## Risk score components

The risk score (0-100) is calculated based on:
- Delay duration (0-40 points)
- Risk category confidence (0-30 points)
- Time-based factors (0-15 points)
- Station-specific risk (0-15 points)

## Model training details

The system uses:
- RandomForestRegressor for delay prediction
- RandomForestClassifier for risk categorization
- RandomizedSearchCV for hyperparameter tuning
- 5-fold cross-validation for model evaluation

## Project structure

```
.
├── data/
│   ├── raw/
│   │   ├── delay-data/     # Raw delay data files
│   │   └── reference-data/ # Reference data files
│   └── processed/          # Processed data
|── docs/                   # Documentation
├── models/                 # Trained models and metadata
├── notebooks/              # Jupyter notebooks explaining the system
├── src/
│   ├── data_processing.py  # Data processing pipeline
│   ├── model_training.py   # Model training pipeline
│   ├── predict.py          # Prediction functionality
│   └── web_app.py          # Web interface
├── Dockerfile              # Container configuration
└── requirements.txt        # Python dependencies
```

## Setup and installation

1. Clone the repository:
```bash
git clone https://github.com/oleksiimorozenko/event-risk-management.git
cd event-risk-management
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Place your data files in the appropriate directories:
   - Run the 
   - Raw delay data in `data/raw/delay-data/`
   - Reference data in `data/raw/reference-data/`

4. Run the data processing pipeline:
```bash
python src/data_processing.py
```

5. Train the models:
```bash
python src/model_training.py
```

## Usage
The application is meant to be deployed to a public resource and used via web interface. However, it was architected to be platform- and OS-independent, thus using containers is the recommended and the most reproducible way of working with the application.

### Local development and deployment
For the local development, make sure you've completed the [local environment setup](https://github.com/UofT-DSI/onboarding/tree/main/environment_setup) section for your operating system described in the onboarding repository.

#### Process the data

Download and process the data:
```bash
python src/data_processing.py
```

#### Train the model

Train the models and save them to the model directory:
```bash
python src/model_training.py
```

#### Run from CLI

Make predictions using the command line:
```bash
python src/predict.py --date 2024-03-20 --time 08:30 --station "FINCH STATION" --line YU --bound N
```

#### Use via web interface

1. Start the web server:
```bash
python src/web_app.py
```

2. Access the web interface at `http://localhost:5000`

### Docker deployment
1. Train the models:
```bash
docker build -t event-risk-management-trainer:latest -f Dockerfile.train .
```

2. Save model files from the train container image:
```bash
docker cp event-risk-management-trainer:/app/models/. ./models
```

3. Build the main image:
```bash
docker build -t event-risk-management:latest .
```

4. Run the container:
```bash
docker run -p 5000:5000 event-risk-management:latest
```
### Cloud deployment
1. Create a registry account (e.g. Docker.com).
2. Generate authentication token.
3. Tag the local image:
```bash
docker tag event-risk-management:latest oleksiimorozenko/event-risk-management:latest
```
4. Tag the image:
```bash
docker push oleksiimorozenko/event-risk-management:latest
```
5. Deploy the image on the platform of your choice (e.g. Koyeb)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## TODOs

 -  Add field validations. After a station or a bound is selected, filter out incompatible selections.

 - Merge Dockerfiles and enable multi-stage build
