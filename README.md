# Privacy Modeling and Analysis for IoT-Enabled Smart Cities

A comprehensive full-stack software engineering project that combines machine learning, ontology modeling, and privacy analysis for IoT-enabled smart cities. The project implements a complete data pipeline, from data generation to analysis and visualization through an interactive dashboard.

## Project Overview

This project develops a privacy-preserving framework for IoT-enabled smart cities that:
- Generates synthetic IoT device data with privacy-sensitive attributes
- Analyzes privacy risks using machine learning models
- Visualizes insights through an interactive dashboard
- Integrates ontology-based knowledge representation
- Provides REST API endpoints for data access and model predictions

## Project Structure

### Initial Files (Required at Start)
```
.
├── api/                    # API implementation
│   ├── __init__.py
│   └── main.py            # FastAPI endpoints
├── ml_models/             # ML model implementations
│   ├── __init__.py
│   ├── data_preprocessor.py
│   └── privacy_risk_classifier.py
├── models/                # Domain models
│   ├── __init__.py
│   └── domain_models.py
├── ontology_handlers/     # Ontology processing
│   ├── __init__.py
│   └── privacy_ontology.py
├── ontologies/           # Ontology definitions
│   └── privacy_ontology.owl
├── tests/               # Test suite
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data_preprocessor.py
│   └── test_domain_models.py
├── .env                 # Environment variables
├── conftest.py         # Pytest configuration
├── docker-compose.yml  # Docker configuration
├── pytest.ini         # Test configuration
├── requirements.txt   # Project dependencies
├── setup.cfg         # Development tool configs
├── analyze_dataset.py    # Dataset analysis script
├── dashboard.py         # Interactive visualization
├── generate_dataset.py  # Data generation script
├── run_data_pipeline.py # Pipeline orchestration
└── train_classifier.py  # Model training script
```

### Generated Files and Directories
```
.
├── data/              # Generated during pipeline execution
│   ├── raw/          # Raw synthetic data
│   ├── processed/    # Processed datasets
│   └── analysis/     # Analysis outputs
│       ├── categorical/     # Categorical feature distributions
│       ├── numerical/      # Numerical feature distributions
│       └── correlations/   # Feature correlation analysis
├── models/           # Generated during training
│   ├── classifier/   # Trained ML models
│   └── metrics/      # Model performance metrics
├── htmlcov/         # Generated during testing
│   └── index.html   # Coverage reports
└── .coverage        # Test coverage data
```

## Execution Flow

1. **Environment Setup**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Data Generation and Processing**
   ```bash
   # This will create data/ directory and generate synthetic data
   python generate_dataset.py
   
   # Analyze the generated dataset
   python analyze_dataset.py
   
   # Or run the complete pipeline
   python run_data_pipeline.py
   ```

3. **Model Training**
   ```bash
   # This will create models/ directory and train the classifier
   python train_classifier.py
   ```

4. **Running the Dashboard**
   ```bash
   python dashboard.py
   ```

5. **Running Tests**
   ```bash
   # Run tests and generate coverage reports in htmlcov/
   pytest --cov=.
   ```

## Directory Purposes

- `api/`: FastAPI implementation for REST endpoints
- `ml_models/`: Machine learning model implementations
- `models/`: Domain model definitions
- `ontology_handlers/`: Ontology processing and management
- `ontologies/`: OWL ontology files
- `tests/`: Test suite and fixtures
- `data/`: (Generated) Data storage
- `models/`: (Generated) Trained model artifacts
- `htmlcov/`: (Generated) Test coverage reports

## Dependencies

The project requires Python 3.8+ and the following main packages:
- Data Processing: numpy, pandas
- Machine Learning: torch, scikit-learn, xgboost
- Visualization: dash, plotly
- API: fastapi, uvicorn
- Testing: pytest
- Ontology: owlready2, rdflib

For a complete list of dependencies, see `requirements.txt`.

## Development

- Code formatting: `black .`
- Import sorting: `isort .`
- Linting: `flake8`
- Type checking: `mypy .`

## License

MIT License

## Contributors

Lokik Ganeriwal



############################

## Data Analysis

#############################

The project includes comprehensive data analysis capabilities that generate visualizations and insights about the IoT privacy dataset.

### Features Analyzed

1. **Categorical Features**
   - Device Type Distribution
   - Data Type Distribution
   - Location Type Distribution
   - Access Pattern Analysis
   - Data Sharing Patterns
   - Compliance Status Distribution

2. **Numerical Features**
   - Access Frequency
   - Network Security Level
   - Data Sensitivity
   - Encryption Level
   - Retention Period
   - Data Volume
   - Last Audit Days
   - Storage Duration
   - Security Incidents

3. **Risk Analysis**
   - Risk Level Distribution
   - Feature Correlations with Risk
   - Security Metric Distributions

### Generated Visualizations

The analysis script (`analyze_dataset.py`) generates the following visualizations in the `data/analysis` directory:

1. **Distribution Plots**
   - Histograms for numerical features
   - Count plots for categorical features
   - Risk level distribution

2. **Correlation Analysis**
   - Correlation heatmap of numerical features
   - Feature importance plots
   - Risk factor relationships

### Running the Analysis

```bash
# Generate and analyze the dataset
python generate_dataset.py
python analyze_dataset.py

# Or run the complete pipeline
python run_data_pipeline.py
```

The analysis results will be saved in the following structure:
```
data/analysis/
├── categorical/
│   ├── device_type_distribution.png
│   ├── data_type_distribution.png
│   ├── location_type_distribution.png
│   ├── access_pattern_distribution.png
│   ├── data_sharing_distribution.png
│   └── compliance_status_distribution.png
├── numerical/
│   ├── access_frequency_distribution.png
│   ├── network_security_distribution.png
│   ├── data_sensitivity_distribution.png
│   └── ...
└── correlations/
    ├── correlation_heatmap.png
    └── risk_correlations.png
```

### Interpreting Results

1. **Feature Distributions**
   - Understand the distribution of different IoT devices and data types
   - Identify common access patterns and security levels
   - Analyze compliance status across the dataset

2. **Risk Analysis**
   - Identify high-risk device types and data categories
   - Understand correlations between security measures and risk levels
   - Analyze the effectiveness of privacy controls

3. **Security Metrics**
   - Evaluate encryption level distributions
   - Analyze network security patterns
   - Track security incident frequencies

to view the api


http://localhost:8000/docs#/


http://localhost:8000/redoc#operation/health_check_health_get
