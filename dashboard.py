import base64
import json
import logging
import random
import datetime
from pathlib import Path
from enum import Enum

import dash
from dash import Dash, dcc, html, Input, Output, State, callback_context, ALL
import dash_bootstrap_components as dbc
from owlready2 import *
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

from ml_models.privacy_risk_classifier import PrivacyRiskClassifier
from ontology_handlers.privacy_ontology import PrivacyOntologyHandler

# Define data types as an Enum
class DataType(Enum):
    PERSONAL = "Personal"
    SENSITIVE = "Sensitive"
    ANONYMIZED = "Anonymized"
    AGGREGATED = "Aggregated"

# Define device types as an Enum
class DeviceType(Enum):
    SMART_HOME = "Smart Home"
    HEALTHCARE = "Healthcare"
    TRANSPORTATION = "Transportation"
    INDUSTRIAL = "Industrial"
    WEARABLE = "Wearable"

# Define location types as an Enum
class LocationType(Enum):
    INDOOR = "Indoor"
    OUTDOOR = "Outdoor"
    MOBILE = "Mobile"
    FIXED = "Fixed"

# Define access patterns as an Enum
class AccessPattern(Enum):
    READ = "Read"
    WRITE = "Write"
    DELETE = "Delete"
    SHARE = "Share"

# Define compliance status as an Enum
class ComplianceStatus(Enum):
    COMPLIANT = "Compliant"
    NON_COMPLIANT = "Non-Compliant"
    PARTIALLY_COMPLIANT = "Partially Compliant"

# Define data sharing types as an Enum
class DataSharing(Enum):
    INTERNAL = "Internal"
    EXTERNAL = "External"
    THIRD_PARTY = "Third Party"

# Privacy Policy class
class PrivacyPolicy:
    def __init__(self, id, name, description, data_types, retention_period, 
                 access_control_rules, compliance_requirements, is_active=True):
        self.id = id
        self.name = name
        self.description = description
        self.data_types = data_types
        self.retention_period = retention_period
        self.access_control_rules = access_control_rules
        self.compliance_requirements = compliance_requirements
        self.is_active = is_active

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Custom styles
CUSTOM_STYLES = {
    "main-container": {
        "backgroundColor": "#f8f9fa",
        "minHeight": "100vh",
        "padding": "20px",
    },
    "navbar": {
        "boxShadow": "0 2px 4px rgba(0,0,0,0.1)",
        "marginBottom": "30px",
        "backgroundColor": "#2c3e50",
    },
    "navbar-brand": {
        "fontSize": "24px",
        "fontWeight": "bold",
        "color": "#1633c7",
    },
    "nav-link": {
        "color": "#ecf0f1",
        "fontSize": "16px",
        "marginLeft": "10px",
        "transition": "color 0.3s ease",
    },
    "nav-link-active": {
        "color": "#3498db",
        "borderBottom": "2px solid #3498db",
    },
    "tab-container": {
        "backgroundColor": "#ffffff",
        "borderRadius": "10px",
        "boxShadow": "0 4px 6px rgba(0,0,0,0.1)",
        "padding": "25px",
        "marginBottom": "30px",
    },
    "tab": {
        "padding": "15px 25px",
        "fontSize": "16px",
        "fontWeight": "500",
        "color": "#2c3e50",
        "backgroundColor": "transparent",
        "border": "none",
        "borderBottom": "2px solid transparent",
        "transition": "all 0.3s ease",
    },
    "tab-selected": {
        "color": "#3498db",
        "borderBottom": "2px solid #3498db",
        "backgroundColor": "#f8f9fa",
    },
    "content-container": {
        "backgroundColor": "#ffffff",
        "borderRadius": "10px",
        "boxShadow": "0 4px 6px rgba(0,0,0,0.1)",
        "padding": "25px",
        "marginTop": "20px",
    },
    "card": {
        "borderRadius": "8px",
        "boxShadow": "0 4px 6px rgba(0,0,0,0.1)",
        "transition": "transform 0.3s ease",
        "backgroundColor": "#ffffff",
        "&:hover": {
            "transform": "translateY(-5px)",
        },
    },
    "section-title": {
        "color": "#2c3e50",
        "fontSize": "24px",
        "fontWeight": "bold",
        "marginBottom": "20px",
        "paddingBottom": "10px",
        "borderBottom": "2px solid #3498db",
    },
    "diagram-card": {
        "boxShadow": "0 4px 8px 0 rgba(0,0,0,0.2)",
        "transition": "0.3s",
        "borderRadius": "5px",
        "padding": "15px",
        "marginBottom": "20px",
        "backgroundColor": "#ffffff",
    },
    "diagram-title": {"color": "#2c3e50", "marginBottom": "15px", "fontWeight": "bold"},
    "diagram-container": {
        "border": "1px solid #e0e0e0",
        "padding": "10px",
        "borderRadius": "5px",
        "backgroundColor": "#f8f9fa",
    },
    "ontology-card": {
        "boxShadow": "0 4px 8px 0 rgba(0,0,0,0.2)",
        "transition": "0.3s",
        "borderRadius": "5px",
        "padding": "15px",
        "marginBottom": "20px",
        "backgroundColor": "#ffffff",
    },
    "ontology-section": {"marginBottom": "30px"},
    "alert-card": {
        "boxShadow": "0 4px 8px 0 rgba(0,0,0,0.2)",
        "transition": "0.3s",
        "borderRadius": "5px",
        "padding": "15px",
        "marginBottom": "20px",
        "backgroundColor": "#fff3cd",
    },
    "analysis-card": {
        "boxShadow": "0 4px 8px 0 rgba(0,0,0,0.2)",
        "transition": "0.3s",
        "borderRadius": "5px",
        "padding": "15px",
        "marginBottom": "20px",
        "backgroundColor": "#ffffff",
    },
    "analysis-section": {
        "marginBottom": "30px",
        "padding": "20px",
        "backgroundColor": "#f8f9fa",
        "borderRadius": "10px",
    },
    "analysis-title": {
        "color": "#2c3e50",
        "marginBottom": "20px",
        "fontWeight": "bold",
        "borderBottom": "2px solid #3498db",
        "paddingBottom": "10px",
    },
    "analysis-subtitle": {
        "color": "#34495e",
        "marginBottom": "15px",
        "fontWeight": "600",
    },
    "analysis-description": {
        "color": "#7f8c8d",
        "marginBottom": "20px",
        "fontSize": "1.1em",
    },
}

# Initialize components
classifier = PrivacyRiskClassifier()
ontology_handler = PrivacyOntologyHandler()

# Sample data for risk assessment
policies_db = {
    "policy1": PrivacyPolicy(
        id="policy1",
        name="Data Minimization Policy",
        description="Ensures only necessary data is collected",
        data_types=[DataType.PERSONAL, DataType.SENSITIVE],
        retention_period=30,
        access_control_rules=[{"role": "admin", "permission": "read"}],
        compliance_requirements=["GDPR", "CCPA"],
        is_active=True
    ),
    "policy2": PrivacyPolicy(
        id="policy2",
        name="Encryption Policy",
        description="Requires encryption for all sensitive data",
        data_types=[DataType.SENSITIVE],
        retention_period=60,
        access_control_rules=[{"role": "admin", "permission": "read"}],
        compliance_requirements=["GDPR"],
        is_active=True
    )
}

# Sample risk history
risk_history = [
    {
        "id": "risk1",
        "device_id": "device1",
        "risk_level": 4,
        "timestamp": datetime.datetime.now(),
        "mitigation_suggestions": ["Implement stronger encryption", "Review access controls"],
        "resolved": False
    },
    {
        "id": "risk2",
        "device_id": "device2",
        "risk_level": 2,
        "timestamp": datetime.datetime.now() - datetime.timedelta(days=1),
        "mitigation_suggestions": ["Update privacy policy"],
        "resolved": True
    },
    {
        "id": "risk3",
        "device_id": "device3",
        "risk_level": 3,
        "timestamp": datetime.datetime.now() - datetime.timedelta(days=2),
        "mitigation_suggestions": ["Conduct security audit"],
        "resolved": False
    }
]

# Sample device categories
device_categories = {
    "Smart Home": [
        {"id": "device1", "name": "Smart Thermostat", "risk_level": 4},
        {"id": "device2", "name": "Smart Lock", "risk_level": 2},
        {"id": "device3", "name": "Smart Camera", "risk_level": 3}
    ],
    "Healthcare": [
        {"id": "device4", "name": "Health Monitor", "risk_level": 5},
        {"id": "device5", "name": "Medical Device", "risk_level": 4}
    ],
    "Transportation": [
        {"id": "device6", "name": "Smart Vehicle", "risk_level": 3},
        {"id": "device7", "name": "Traffic Sensor", "risk_level": 2}
    ]
}

# Create Data Analysis Tab Layout
def create_analysis_tab():
    """Create the layout for the Data Analysis tab with interactive features."""
    return html.Div([
        html.H3("Data Analysis Results", className="text-center mb-4", style=CUSTOM_STYLES["analysis-title"]),
        
        # View Mode Toggle
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Visualization Mode", className="card-title"),
                        dbc.RadioItems(
                            id="view-mode-selector",
                            options=[
                                {"label": "Static View", "value": "static"},
                                {"label": "Interactive View", "value": "interactive"}
                            ],
                            value="static",
                            inline=True,
                            className="mb-3"
                        )
                    ])
                ], style=CUSTOM_STYLES["analysis-card"])
            ], width={"size": 6, "offset": 3})
        ], className="mb-4"),
        
        # Categorical Features Section
        html.Div([
            html.H4("Categorical Features Analysis", style=CUSTOM_STYLES["analysis-subtitle"]),
            html.P(
                "Analysis of categorical variables including device types, data types, and compliance status.",
                style=CUSTOM_STYLES["analysis-description"]
            ),
            dbc.Spinner(html.Div(id="categorical-features-content"))
        ], style=CUSTOM_STYLES["analysis-section"], className="mb-4"),
        
        # Numerical Features Section
        html.Div([
            html.H4("Numerical Features Analysis", style=CUSTOM_STYLES["analysis-subtitle"]),
            html.P(
                "Analysis of numerical metrics including security levels, data volumes, and time-based features.",
                style=CUSTOM_STYLES["analysis-description"]
            ),
            dbc.Spinner(html.Div(id="numerical-features-content"))
        ], style=CUSTOM_STYLES["analysis-section"], className="mb-4"),
        
        # Correlation Analysis Section
        html.Div([
            html.H4("Correlation Analysis", style=CUSTOM_STYLES["analysis-subtitle"]),
            html.P(
                "Analysis of relationships between features and their impact on privacy risks.",
                style=CUSTOM_STYLES["analysis-description"]
            ),
            dbc.Spinner(html.Div(id="correlation-analysis-content"))
        ], style=CUSTOM_STYLES["analysis-section"]),
        
        # Store the visualizations data
        dcc.Store(id="visualizations-store", data=load_analysis_visualizations())
    ], className="p-4")

# Initialize the Dash app
app = dash.Dash(
    __name__, 
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True  # Add this line to suppress callback exceptions
)


# Load UML diagrams
def load_uml_diagrams():
    """Load UML diagrams from the uml directory."""
    uml_dir = Path("uml")
    diagrams = {}
    if not uml_dir.exists():
        logger.warning("UML directory not found")
        return diagrams
    
    try:
        for file in uml_dir.glob("*.png"):
            with open(file, "rb") as img_file:
                diagrams[file.stem] = base64.b64encode(img_file.read()).decode("utf-8")
        return diagrams
    except Exception as e:
        logger.error(f"Error loading UML diagrams: {str(e)}")
        return diagrams


# Load ontology diagram
def load_ontology_diagram():
    """Load the ontology diagram."""
    diagram_path = Path("ontologies/ontology_diagram.jpg")
    try:
        if diagram_path.exists():
            with open(diagram_path, "rb") as img_file:
                return base64.b64encode(img_file.read()).decode("utf-8")
        return None
    except Exception as e:
        logger.error(f"Error loading ontology diagram: {str(e)}")
        return None


# Load ontology
def load_ontology():
    """Load the privacy ontology."""
    try:
        onto_path = Path("ontologies/privacy_ontology.owl")
        if onto_path.exists():
            return get_ontology(str(onto_path)).load()
    except Exception as e:
        logger.error(f"Error loading ontology: {str(e)}")
    return None


# Generate sample risk data
def generate_risk_data():
    """Generate sample risk assessment data."""
    return {
        "high_risk": random.randint(1, 10),
        "medium_risk": random.randint(5, 15),
        "low_risk": random.randint(10, 20),
        "active_policies": random.randint(8, 15),
        "alerts": random.randint(0, 5),
    }


# Cache for visualizations
_visualization_cache = None

# Load analysis visualizations
def load_analysis_visualizations():
    """Load both static and interactive visualizations from the data/analysis directory."""
    global _visualization_cache
    
    # Return cached visualizations if available
    if _visualization_cache is not None:
        return _visualization_cache
    
    # Import necessary modules
    import base64
    from pathlib import Path
    import logging
    
    # Set up logger
    logger = logging.getLogger(__name__)
    
    analysis_dir = Path("data/analysis")
    visualizations = {
        "categorical": {"static": {}, "interactive": {}},
        "numerical": {"static": {}, "interactive": {}},
        "correlations": {"static": {}, "interactive": {}}
    }
    
    if not analysis_dir.exists():
        logger.warning(f"Analysis directory not found at {analysis_dir}")
        return visualizations
    
    # Define file mappings for each category
    file_mappings = {
        "categorical": [
            "device_type_distribution",
            "data_type_distribution",
            "location_type_distribution",
            "access_pattern_distribution",
            "data_sharing_distribution",
            "compliance_status_distribution"
        ],
        "numerical": [
            "access_frequency_distribution",
            "network_security_level_distribution",
            "data_sensitivity_distribution",
            "encryption_level_distribution",
            "retention_period_distribution",
            "data_volume_distribution",
            "last_audit_days_distribution",
            "storage_duration_distribution",
            "security_incidents_distribution"
        ],
        "correlations": [
            "correlation_heatmap",
            "risk_correlations",
        ]
    }
    
    # Load static and interactive visualizations
    missing_static_files = []
    missing_interactive_files = []
    loaded_files = set()  # Track loaded files to avoid duplicate logging
    
    for category, base_names in file_mappings.items():
        # Create paths
        category_dir = analysis_dir / category
        
        for base_name in base_names:
            # Load static visualizations (.png files)
            png_file = analysis_dir / category / f"{base_name}.png"
            if png_file.exists() and png_file not in loaded_files:
                try:
                    with open(png_file, "rb") as img_file:
                        visualizations[category]["static"][base_name] = base64.b64encode(img_file.read()).decode("utf-8")
                        logger.info(f"Loaded static visualization: {png_file.name}")
                        loaded_files.add(png_file)
                except Exception as e:
                    logger.error(f"Error loading static visualization {png_file.name}: {str(e)}")
            elif png_file not in loaded_files:
                missing_static_files.append(str(png_file))
                loaded_files.add(png_file)
            
            # Load interactive visualizations (.html files)
            html_file = analysis_dir / category / f"{base_name}.html"
            if html_file.exists() and html_file not in loaded_files:
                try:
                    with open(html_file, "r", encoding="utf-8") as f:
                        visualizations[category]["interactive"][base_name] = f.read()
                        logger.info(f"Loaded interactive visualization: {html_file.name}")
                        loaded_files.add(html_file)
                except Exception as e:
                    logger.error(f"Error loading interactive visualization {html_file.name}: {str(e)}")
            elif html_file not in loaded_files:
                missing_interactive_files.append(str(html_file))
                loaded_files.add(html_file)
    
    # Log summary of missing files once
    if missing_static_files:
        logger.warning(f"Missing static visualization files: {len(missing_static_files)} files")
        logger.debug(f"Missing static files: {', '.join(missing_static_files)}")
        
    if missing_interactive_files:
        logger.warning(f"Missing interactive visualization files: {len(missing_interactive_files)} files")
        logger.debug(f"Missing interactive files: {', '.join(missing_interactive_files)}")
    
    # Log summary of loaded visualizations
    for category in visualizations:
        static_count = len(visualizations[category]["static"])
        interactive_count = len(visualizations[category]["interactive"])
        logger.info(f"{category} visualizations loaded: {static_count} static, {interactive_count} interactive")
    
    # Cache the results
    _visualization_cache = visualizations
    return visualizations

# Home Page Layout
home_layout = dbc.Tabs(
    [
        # UML Diagrams Tab
        dbc.Tab(
            [
                html.Div([
                    html.H2("System Architecture Diagrams", className="text-center mb-4"),
                    
                    # Activity Diagram Section
                            dbc.Card([
                        dbc.CardHeader(html.H4("Activity Diagram - Privacy Risk Mitigation Process", className="text-center", style=CUSTOM_STYLES["diagram-title"])),
                                dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    html.Div([
                                                                html.Img(
                                            src=f"data:image/png;base64,{load_uml_diagrams()['Privacy Risk Mitigation Process']}",
                                            style={"width": "100%", "height": "auto", "objectFit": "contain"},
                                        )
                                    ], style=CUSTOM_STYLES["diagram-container"])
                        ], width=6),
                        dbc.Col([
                                    html.P("This activity diagram models the workflow of privacy risk mitigation in an IoT-enabled smart city environment. It visually outlines the decision-making and data-handling procedures from the moment data is collected to the point where privacy risks are assessed and addressed.", className="mb-3"),
                                    html.H6("Key Activities:", className="mb-2"),
                                    html.Ul([
                                        html.Li("Data Collection: The process begins with gathering data from IoT devices."),
                                        html.Li("Data Type Classification: Checks whether the data is personal or non-personal."),
                                        html.Li("User Consent Verification: If the data is personal, the system checks for user consent."),
                                        html.Li("Data Storage and Analysis: The data is stored securely and analyzed for privacy risks."),
                                        html.Li("Risk-Based Decision Making: Different actions based on risk levels (High, Medium, Low).")
                                    ]),
                                    html.P("This diagram emphasizes a privacy-by-design approach and shows how user consent, encryption, anonymization, and risk assessment are integral to protecting citizen data.", className="mt-3")
                                ], width=6)
                            ])
                        ])
                    ], style=CUSTOM_STYLES["diagram-card"], className="mb-4"),
                    
                    # Class Diagram Section
                            dbc.Card([
                        dbc.CardHeader(html.H4("Class Diagram - Privacy Analysis System", className="text-center", style=CUSTOM_STYLES["diagram-title"])),
                                dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    html.Div([
                                                                html.Img(
                                            src=f"data:image/png;base64,{load_uml_diagrams()['Privacy Analysis System']}",
                                            style={"width": "100%", "height": "auto", "objectFit": "contain"},
                                        )
                                    ], style=CUSTOM_STYLES["diagram-container"])
                        ], width=6),
                        dbc.Col([
                                    html.P("The class diagram models the architectural structure of your privacy analysis system. It showcases key components involved in ontology management and machine learning-based risk classification.", className="mb-3"),
                                    html.H6("Core Components:", className="mb-2"),
                                    html.Ul([
                                        html.Li("Ontology Handling: PrivacyOntologyHandler manages ontology files related to personal data and associated risks."),
                                        html.Li("Machine Learning: PrivacyRiskClassifier uses ML models to classify data based on associated privacy risks."),
                                        html.Li("Controller: PrivacyManager acts as the main controller, coordinating ontology queries and ML classification.")
                                    ]),
                                    html.P("This class diagram defines a modular and scalable architecture, combining AI/ML and semantic ontology-based analysis to enable dynamic, automated privacy risk detection in real time.", className="mt-3")
                                ], width=6)
                            ])
                        ])
                    ], style=CUSTOM_STYLES["diagram-card"], className="mb-4"),
                    
                    # Deployment Diagram Section
                            dbc.Card([
                        dbc.CardHeader(html.H4("Deployment Diagram - Privacy Analysis System Deployment", className="text-center", style=CUSTOM_STYLES["diagram-title"])),
                                dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    html.Div([
                                                                html.Img(
                                                                    src=f"data:image/png;base64,{load_uml_diagrams()['Privacy Analysis System Deployment']}",
                                            style={"width": "100%", "height": "auto", "objectFit": "contain"},
                                        )
                                    ], style=CUSTOM_STYLES["diagram-container"])
                        ], width=6),
                        dbc.Col([
                                    html.P("This deployment diagram shows the physical architecture and layered deployment of components in your privacy analysis system across the smart city infrastructure.", className="mb-3"),
                                    html.H6("Deployment Layers:", className="mb-2"),
                                    html.Ul([
                                        html.Li("Edge Layer: IoT Devices and Edge Gateway for local data processing"),
                                        html.Li("Processing Layer: Data Preprocessor and Privacy Risk Classifier"),
                                        html.Li("Knowledge Layer: Ontology Store and Risk Database"),
                                        html.Li("Application Layer: Privacy Dashboard and API Gateway"),
                                        html.Li("Cloud Services: Analytics Engine and Policy Manager")
                                    ]),
                                    html.P("This diagram reflects a distributed, layered architecture, supporting edge computing, centralized knowledge, and scalable cloud analytics, ensuring privacy control at every stage of data flow.", className="mt-3")
                                ], width=6)
                            ])
                        ])
                    ], style=CUSTOM_STYLES["diagram-card"], className="mb-4"),
                    
                    # Sequence Diagram Section
                            dbc.Card([
                        dbc.CardHeader(html.H4("Sequence Diagram - Privacy Risk Assessment Workflow", className="text-center", style=CUSTOM_STYLES["diagram-title"])),
                                dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    html.Div([
                                                                html.Img(
                                            src=f"data:image/png;base64,{load_uml_diagrams()['Privacy Risk Assessment Sequence']}",
                                            style={"width": "100%", "height": "auto", "objectFit": "contain"},
                                        )
                                    ], style=CUSTOM_STYLES["diagram-container"])
                        ], width=6),
                                dbc.Col([
                                    html.P("This sequence diagram shows a step-by-step interaction flow between actors and system components during a privacy risk assessment request from the user.", className="mb-3"),
                                    html.H6("Interaction Flow:", className="mb-2"),
                                    html.Ol([
                                        html.Li("User sends a risk assessment request via the Privacy Dashboard"),
                                        html.Li("Dashboard collects device data from the IoT Device"),
                                        html.Li("It queries the Privacy Ontology Handler for relevant rules"),
                                        html.Li("Sends the data to the Privacy Risk Classifier for risk evaluation"),
                                        html.Li("Results are stored in the Data Store"),
                                        html.Li("Dashboard receives and displays the assessment result to the user")
                                    ]),
                                    html.P("This diagram highlights the real-time privacy assessment workflow, showing how semantic rules and ML models work together to deliver quick and context-aware privacy feedback to users.", className="mt-3")
                                ], width=6)
                            ])
                        ])
                    ], style=CUSTOM_STYLES["diagram-card"])
                ], className="p-4")
            ],
            label="System Architecture",
            tab_id="tab-uml"
        ),
        # Ontology Tab
        dbc.Tab(
            [
                html.Div([
                    html.H3("Privacy Ontology", style=CUSTOM_STYLES["section-title"]),
                    dbc.Card([
                        dbc.CardHeader(html.H4("Privacy Knowledge Model", style=CUSTOM_STYLES["diagram-title"])),
                        dbc.CardBody([
                            html.Div([
                                                                html.Img(
                                    src=f"data:image/jpg;base64,{load_ontology_diagram()}",
                                    style={"width": "100%", "maxWidth": "800px", "height": "auto", "margin": "auto", "display": "block"}
                                )
                            ], style=CUSTOM_STYLES["diagram-container"])
                        ])
                    ], style=CUSTOM_STYLES["ontology-card"]),
                    
                    # Core Classes Section
                    html.Div([
                        html.H4("Core Classes", style=CUSTOM_STYLES["diagram-title"], className="mt-4"),
                        dbc.Row([
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardHeader(html.H5("Personal Data")),
                                    dbc.CardBody([
                                        html.P("Represents any personal data collected or processed"),
                                        html.H6("Subclasses:", className="mt-3"),
                                        html.Ul([
                                            html.Li("SensitivePersonalData"),
                                            html.Li("BiometricData"),
                                            html.Li("HealthData"),
                                            html.Li("LocationData"),
                                            html.Li("BehavioralData")
                                        ])
                                    ])
                                ], className="h-100")
                            ], width=4),
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardHeader(html.H5("Privacy Policy")),
                                    dbc.CardBody([
                                        html.P("Defines rules and regulations for data handling"),
                                        html.H6("Subclasses:", className="mt-3"),
                                        html.Ul([
                                            html.Li("LegalFramework"),
                                            html.Li("OrganizationalPolicy")
                                        ])
                                    ])
                                ], className="h-100")
                            ], width=4),
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardHeader(html.H5("Risk")),
                                    dbc.CardBody([
                                        html.P("Represents privacy risks associated with data processing"),
                                        html.H6("Subclasses:", className="mt-3"),
                                        html.Ul([
                                            html.Li("SecurityRisk"),
                                            html.Li("PrivacyRisk"),
                                            html.Li("ComplianceRisk")
                                        ])
                                    ])
                                ], className="h-100")
                            ], width=4)
                        ], className="mb-4"),
                        
                        # Data Processing Section
                        html.H4("Data Processing", style=CUSTOM_STYLES["diagram-title"], className="mt-4"),
                        dbc.Row([
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardHeader(html.H5("Data Processing Operations")),
                                    dbc.CardBody([
                                        html.P("Represents operations performed on personal data"),
                                        html.H6("Subclasses:", className="mt-3"),
                                        html.Ul([
                                            html.Li("Collection"),
                                            html.Li("Storage"),
                                            html.Li("Analysis"),
                                            html.Li("Sharing")
                                        ])
                                    ])
                                ], className="h-100")
                            ], width=6),
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardHeader(html.H5("Consent")),
                                    dbc.CardBody([
                                        html.P("Represents user consent for data processing"),
                                        html.H6("Subclasses:", className="mt-3"),
                                        html.Ul([
                                            html.Li("ExplicitConsent"),
                                            html.Li("ImpliedConsent")
                                        ])
                                    ])
                                ], className="h-100")
                            ], width=6)
                        ], className="mb-4"),
                        
                        # Properties Section
                        html.H4("Properties", style=CUSTOM_STYLES["diagram-title"], className="mt-4"),
                        dbc.Row([
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardHeader(html.H5("Object Properties")),
                                    dbc.CardBody([
                                        html.Ul([
                                            html.Li("ownsData: Links users to their personal data"),
                                            html.Li("regulatedBy: Links data to privacy policies"),
                                            html.Li("introducesRisk: Links processing to risks"),
                                            html.Li("hasConsent: Links users to consent decisions"),
                                            html.Li("implementsMeasure: Links controllers to security measures"),
                                            html.Li("transfersTo: Indicates data transfers"),
                                            html.Li("processesData: Links processors to data"),
                                            html.Li("mitigatesRisk: Links measures to risks"),
                                            html.Li("requiresConsent: Links processing to consent")
                                        ])
                                    ])
                                ], className="h-100")
                            ], width=6),
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardHeader(html.H5("Data Properties")),
                                    dbc.CardBody([
                                        html.Ul([
                                            html.Li("dataType: Type of personal data"),
                                            html.Li("riskLevel: Numeric level of risk (1-5)"),
                                            html.Li("consentStatus: Whether consent is given"),
                                            html.Li("processingPurpose: Purpose of processing"),
                                            html.Li("retentionPeriod: Data retention duration"),
                                            html.Li("encryptionLevel: Level of encryption"),
                                            html.Li("transferLocation: Location of transfer"),
                                            html.Li("dataSensitivity: Sensitivity level"),
                                            html.Li("processingFrequency: Processing frequency")
                                        ])
                                    ])
                                ], className="h-100")
                            ], width=6)
                        ], className="mb-4"),
                        
                        # Key Concepts Section
                        html.H4("Key Concepts", style=CUSTOM_STYLES["diagram-title"], className="mt-4"),
                        dbc.Row([
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardHeader(html.H5("Data Controller")),
                                    dbc.CardBody(html.P("Entity that determines purposes and means of processing personal data"))
                                ], className="h-100")
                            ], width=4),
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardHeader(html.H5("Data Processor")),
                                    dbc.CardBody(html.P("Entity that processes data on behalf of the controller"))
                                ], className="h-100")
                            ], width=4),
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardHeader(html.H5("Security Measures")),
                                    dbc.CardBody([
                                        html.P("Technical and organizational security measures"),
                                        html.H6("Subclasses:", className="mt-3"),
                                        html.Ul([
                                            html.Li("TechnicalMeasure"),
                                            html.Li("OrganizationalMeasure")
                                        ])
                                    ])
                                ], className="h-100")
                            ], width=4)
                        ], className="mb-4")
                    ], style=CUSTOM_STYLES["ontology-section"])
                ], className="p-4")
            ],
            label="Ontology",
            tab_id="tab-ontology"
        ),
        # Risk Assessment Tab
        dbc.Tab(
            [
                html.Div([
                    html.H3("Risk Assessment Dashboard", style=CUSTOM_STYLES["section-title"]),
                    
                        # Risk Overview Cards
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardBody([
                                    html.H4("High Risk Items", className="card-title text-danger"),
                                    html.H2(id="high-risk-count", className="display-4 text-center"),
                                    html.P("Devices requiring immediate attention", className="card-text text-muted")
                                ])
                            ])
                        ], width=4),
                        dbc.Col([
                            dbc.Card([
                                dbc.CardBody([
                                    html.H4("Medium Risk Items", className="card-title text-warning"),
                                    html.H2(id="medium-risk-count", className="display-4 text-center"),
                                    html.P("Devices requiring monitoring", className="card-text text-muted")
                                ])
                            ])
                        ], width=4),
                        dbc.Col([
                            dbc.Card([
                                dbc.CardBody([
                                    html.H4("Low Risk Items", className="card-title text-success"),
                                    html.H2(id="low-risk-count", className="display-4 text-center"),
                                    html.P("Devices with minimal risk", className="card-text text-muted")
                                ])
                            ])
                        ], width=4),
                    ], className="mb-4"),
                    
                    # Active Policies and Alerts
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader(html.H4("Active Privacy Policies")),
                                dbc.CardBody([
                                    html.H2(id="active-policies-count", className="display-4 text-center text-primary"),
                                    html.P("Currently enforced policies", className="text-muted text-center"),
                                    html.Div(id="policies-list")
                                ])
                            ])
                        ], width=6),
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader(html.H4("Recent Privacy Alerts")),
                                dbc.CardBody([
                                    html.H2(id="alerts-count", className="display-4 text-center text-danger"),
                                    html.P("Alerts requiring attention", className="text-muted text-center"),
                                    html.Div(id="alerts-list")
                                ])
                            ])
                        ], width=6),
                    ], className="mb-4"),
                    
                    # Risk trend and recommendations
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader(html.H4("Privacy Risk Trend (Last 30 Days)")),
                                dbc.CardBody([
                                    dcc.Graph(id="risk-trend-graph")
                                ])
                            ])
                        ], width=12)
                    ])
                ], className="p-4")
            ],
            label="Risk Assessment",
            tab_id="tab-risk"
        ),
        # Data Analysis Tab
        dbc.Tab(
            [
                html.Div([
                    html.H3("Data Analysis Results", className="text-center mb-4", style=CUSTOM_STYLES["analysis-title"]),
                    
                    # View Mode Toggle
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardBody([
                                    html.H5("Visualization Mode", className="card-title"),
                                    dbc.RadioItems(
                                        id="view-mode-selector",
                                        options=[
                                            {"label": "Static View", "value": "static"},
                                            {"label": "Interactive View", "value": "interactive"}
                                        ],
                                        value="static",
                                        inline=True,
                                        className="mb-3"
                                    )
                                ])
                            ], style=CUSTOM_STYLES["analysis-card"])
                        ], width={"size": 6, "offset": 3})
                    ], className="mb-4"),
                    
                    # Categorical Features Section
                    html.Div([
                        html.H4("Categorical Features Analysis", style=CUSTOM_STYLES["analysis-subtitle"]),
                        html.P(
                            "Analysis of categorical variables including device types, data types, and compliance status.",
                            style=CUSTOM_STYLES["analysis-description"]
                        ),
                        dbc.Spinner(html.Div(id="categorical-features-content"))
                    ], style=CUSTOM_STYLES["analysis-section"], className="mb-4"),
                    
                    # Numerical Features Section
                    html.Div([
                        html.H4("Numerical Features Analysis", style=CUSTOM_STYLES["analysis-subtitle"]),
                                                                html.P(
                            "Analysis of numerical metrics including security levels, data volumes, and time-based features.",
                            style=CUSTOM_STYLES["analysis-description"]
                        ),
                        dbc.Spinner(html.Div(id="numerical-features-content"))
                    ], style=CUSTOM_STYLES["analysis-section"], className="mb-4"),
                    
                    # Correlation Analysis Section
                    html.Div([
                        html.H4("Correlation Analysis", style=CUSTOM_STYLES["analysis-subtitle"]),
                                                                html.P(
                            "Analysis of relationships between features and their impact on privacy risks.",
                            style=CUSTOM_STYLES["analysis-description"]
                        ),
                        dbc.Spinner(html.Div(id="correlation-analysis-content"))
                    ], style=CUSTOM_STYLES["analysis-section"]),
                    
                    # Store the visualizations data
                    dcc.Store(id="visualizations-store", data=load_analysis_visualizations())
                ], className="p-4")
            ],
            label="Data Analysis",
            tab_id="tab-analysis"
        ),
    ],
    id="tabs",
    active_tab="tab-home"
)

# Documentation Page Layout
docs_layout = html.Div(
    [
        html.H2("Documentation", className="text-center mb-4"),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardHeader(
                                    html.H4(
                                        "API Documentation",
                                        className="text-center",
                                        style=CUSTOM_STYLES["diagram-title"],
                                    )
                                ),
                                dbc.CardBody(
                                    [
                                        html.H5("Endpoints"),
                                        html.Ul(
                                            [
                                                html.Li(
                                                    [
                                                        html.Strong(
                                                            "POST /api/assess_risk"
                                                        ),
                                                        html.P(
                                                            "Assess privacy risk for a device"
                                                        ),
                                                    ]
                                                ),
                                                html.Li(
                                                    [
                                                        html.Strong(
                                                            "GET /api/risk_history"
                                                        ),
                                                        html.P(
                                                            "Get risk assessment history"
                                                        ),
                                                    ]
                                                ),
                                                html.Li(
                                                    [
                                                        html.Strong(
                                                            "POST /api/register_device"
                                                        ),
                                                        html.P(
                                                            "Register a new IoT device"
                                                        ),
                                                    ]
                                                ),
                                                html.Li(
                                                    [
                                                        html.Strong(
                                                            "GET /api/policies"
                                                        ),
                                                        html.P(
                                                            "Get active privacy policies"
                                                        ),
                                                    ]
                                                ),
                                            ]
                                        ),
                                    ]
                                ),
                            ],
                            style=CUSTOM_STYLES["diagram-card"],
                        )
                    ],
                    width=6,
                ),
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardHeader(
                                    html.H4(
                                        "System Architecture",
                                        className="text-center",
                                        style=CUSTOM_STYLES["diagram-title"],
                                    )
                                ),
                                dbc.CardBody(
                                    [
                                        html.H5("Components"),
                                        html.Ul(
                                            [
                                                html.Li("Privacy Risk Classifier"),
                                                html.Li("Ontology Handler"),
                                                html.Li("Data Preprocessor"),
                                                html.Li("API Server"),
                                                html.Li("Dashboard"),
                                            ]
                                        ),
                                        html.H5("Data Flow"),
                                        html.P(
                                            "The system processes device data through the following steps:"
                                        ),
                                        html.Ol(
                                            [
                                                html.Li(
                                                    "Data preprocessing and normalization"
                                                ),
                                                html.Li(
                                                    "Risk assessment using ML models"
                                                ),
                                                html.Li(
                                                    "Ontology-based policy validation"
                                                ),
                                                html.Li("Risk level determination"),
                                                html.Li("Policy recommendations"),
                                            ]
                                        ),
                                    ]
                                ),
                            ],
                            style=CUSTOM_STYLES["diagram-card"],
                        )
                    ],
                    width=6,
                ),
            ]
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardHeader(
                                    html.H4(
                                        "Privacy Policies",
                                        className="text-center",
                                        style=CUSTOM_STYLES["diagram-title"],
                                    )
                                ),
                                dbc.CardBody(
                                    [
                                        html.H5("Supported Regulations"),
                                        html.Ul(
                                            [
                                                html.Li(
                                                    "GDPR (General Data Protection Regulation)"
                                                ),
                                                html.Li(
                                                    "CCPA (California Consumer Privacy Act)"
                                                ),
                                                html.Li(
                                                    "PIPEDA (Personal Information Protection and Electronic Documents Act)"
                                                ),
                                            ]
                                        ),
                                        html.H5("Policy Components"),
                                        html.P("Each policy includes:"),
                                        html.Ul(
                                            [
                                                html.Li("Data collection requirements"),
                                                html.Li("Storage and processing rules"),
                                                html.Li("User consent mechanisms"),
                                                html.Li("Data retention periods"),
                                                html.Li("Security measures"),
                                            ]
                                        ),
                                    ]
                                ),
                            ],
                            style=CUSTOM_STYLES["diagram-card"],
                        )
                    ],
                    width=12,
                )
            ]
        ),
    ],
    className="p-4",
)

# About Page Layout
about_layout = html.Div(
    [
        html.H2("About Privacy Risk Assessment System", className="text-center mb-4"),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardHeader(
                                    html.H4(
                                        "System Overview",
                                        className="text-center",
                                        style=CUSTOM_STYLES["diagram-title"],
                                    )
                                ),
                                dbc.CardBody(
                                    [
                                        html.P(
                                            "The Privacy Risk Assessment System is a comprehensive solution for evaluating and managing privacy risks in IoT environments. It combines machine learning, semantic web technologies, and privacy regulations to provide real-time risk assessment and policy recommendations."
                                        ),
                                        html.H5("Key Features"),
                                        html.Ul(
                                            [
                                                html.Li(
                                                    "Real-time privacy risk assessment"
                                                ),
                                                html.Li(
                                                    "Multi-regulation compliance checking"
                                                ),
                                                html.Li(
                                                    "Ontology-based policy management"
                                                ),
                                                html.Li(
                                                    "Interactive dashboard for monitoring"
                                                ),
                                                html.Li(
                                                    "Automated policy recommendations"
                                                ),
                                            ]
                                        ),
                                    ]
                                ),
                            ],
                            style=CUSTOM_STYLES["diagram-card"],
                        )
                    ],
                    width=8,
                    className="mx-auto",
                )
            ]
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardHeader(
                                    html.H4(
                                        "Technology Stack",
                                        className="text-center",
                                        style=CUSTOM_STYLES["diagram-title"],
                                    )
                                ),
                                dbc.CardBody(
                                    [
                                        html.H5("Core Technologies"),
                                        html.Ul(
                                            [
                                                html.Li("Python 3.9+"),
                                                html.Li("FastAPI for REST API"),
                                                html.Li("Dash for Dashboard"),
                                                html.Li("Owlready2 for Ontology"),
                                                html.Li("Scikit-learn for ML models"),
                                            ]
                                        ),
                                        html.H5("Development Tools"),
                                        html.Ul(
                                            [
                                                html.Li("Git for version control"),
                                                html.Li("Pytest for testing"),
                                                html.Li("Black for code formatting"),
                                                html.Li("MkDocs for documentation"),
                                            ]
                                        ),
                                    ]
                                ),
                            ],
                            style=CUSTOM_STYLES["diagram-card"],
                        )
                    ],
                    width=8,
                    className="mx-auto",
                )
            ]
        ),
    ],
    className="p-4",
)

# Create the layout
app.layout = dbc.Container(
    [
        # Navigation Bar
        dbc.Navbar(
            [
                dbc.NavbarBrand(
                    [
                        html.I(className="fas fa-shield-alt me-2"),
                        "Privacy Modelling and Analysis on IoT-enabled Smart Cities -- Dashboard"
                    ],
                    className="ms-2",
                    style=CUSTOM_STYLES["navbar-brand"]
                ),
            ],
            color="dark",
            dark=True,
            style=CUSTOM_STYLES["navbar"],
            className="mb-4",
        ),
        
        # Main Content
        dbc.Card(
            [
                dbc.CardBody(
                    [
                        dbc.Tabs(
                            [
                                dbc.Tab(
                                    [    
                                        # Sub-tabs for different sections
                                        dbc.Tabs([
                                            dbc.Tab(
                                                [
        html.Div([
                                                        html.H2("System Architecture Diagrams", className="text-center mb-4"),
                                                        
                                                        # Activity Diagram Section
                                                                dbc.Card([
                                                            dbc.CardHeader(html.H4("Activity Diagram - Privacy Risk Mitigation Process", className="text-center", style=CUSTOM_STYLES["diagram-title"])),
                                                                    dbc.CardBody([
                                                                dbc.Row([
                                                                    dbc.Col([
        html.Div([
                                                                            html.Img(
                                                                                src=f"data:image/png;base64,{load_uml_diagrams()['Privacy Risk Mitigation Process']}",
                                                                                style={"width": "100%", "height": "auto", "objectFit": "contain"},
                                                                            )
                                                                        ], style=CUSTOM_STYLES["diagram-container"])
                                                            ], width=6),
                                                            dbc.Col([
                                                                        html.P("This activity diagram models the workflow of privacy risk mitigation in an IoT-enabled smart city environment. It visually outlines the decision-making and data-handling procedures from the moment data is collected to the point where privacy risks are assessed and addressed.", className="mb-3"),
                                                                        html.H6("Key Activities:", className="mb-2"),
                                                                        html.Ul([
                                                                            html.Li("Data Collection: The process begins with gathering data from IoT devices."),
                                                                            html.Li("Data Type Classification: Checks whether the data is personal or non-personal."),
                                                                            html.Li("User Consent Verification: If the data is personal, the system checks for user consent."),
                                                                            html.Li("Data Storage and Analysis: The data is stored securely and analyzed for privacy risks."),
                                                                            html.Li("Risk-Based Decision Making: Different actions based on risk levels (High, Medium, Low).")
                                                                        ]),
                                                                        html.P("This diagram emphasizes a privacy-by-design approach and shows how user consent, encryption, anonymization, and risk assessment are integral to protecting citizen data.", className="mt-3")
                                                                    ], width=6)
                                                                ])
                                                            ])
                                                        ], style=CUSTOM_STYLES["diagram-card"], className="mb-4"),
                                                        
                                                        # Class Diagram Section
                                                                dbc.Card([
                                                            dbc.CardHeader(html.H4("Class Diagram - Privacy Analysis System", className="text-center", style=CUSTOM_STYLES["diagram-title"])),
                                                                    dbc.CardBody([
                                                                dbc.Row([
                                                                    dbc.Col([
                                                                        html.Div([
                                                                            html.Img(
                                                                                src=f"data:image/png;base64,{load_uml_diagrams()['Privacy Analysis System']}",
                                                                                style={"width": "100%", "height": "auto", "objectFit": "contain"},
                                                                            )
                                                                        ], style=CUSTOM_STYLES["diagram-container"])
                                                            ], width=6),
                                                            dbc.Col([
                                                                        html.P("The class diagram models the architectural structure of your privacy analysis system. It showcases key components involved in ontology management and machine learning-based risk classification.", className="mb-3"),
                                                                        html.H6("Core Components:", className="mb-2"),
                                                                        html.Ul([
                                                                            html.Li("Ontology Handling: PrivacyOntologyHandler manages ontology files related to personal data and associated risks."),
                                                                            html.Li("Machine Learning: PrivacyRiskClassifier uses ML models to classify data based on associated privacy risks."),
                                                                            html.Li("Controller: PrivacyManager acts as the main controller, coordinating ontology queries and ML classification.")
                                                                        ]),
                                                                        html.P("This class diagram defines a modular and scalable architecture, combining AI/ML and semantic ontology-based analysis to enable dynamic, automated privacy risk detection in real time.", className="mt-3")
                                                                    ], width=6)
                                                                ])
                                                            ])
                                                        ], style=CUSTOM_STYLES["diagram-card"], className="mb-4"),
                                                        
                                                        # Deployment Diagram Section
                                                                dbc.Card([
                                                            dbc.CardHeader(html.H4("Deployment Diagram - Privacy Analysis System Deployment", className="text-center", style=CUSTOM_STYLES["diagram-title"])),
                                                                    dbc.CardBody([
                                                                dbc.Row([
                                                                    dbc.Col([
                                                                        html.Div([
                                                                            html.Img(
                                                                                src=f"data:image/png;base64,{load_uml_diagrams()['Privacy Analysis System Deployment']}",
                                                                                style={"width": "100%", "height": "auto", "objectFit": "contain"},
                                                                            )
                                                                        ], style=CUSTOM_STYLES["diagram-container"])
                                                            ], width=6),
                                                            dbc.Col([
                                                                        html.P("This deployment diagram shows the physical architecture and layered deployment of components in your privacy analysis system across the smart city infrastructure.", className="mb-3"),
                                                                        html.H6("Deployment Layers:", className="mb-2"),
                                                                        html.Ul([
                                                                            html.Li("Edge Layer: IoT Devices and Edge Gateway for local data processing"),
                                                                            html.Li("Processing Layer: Data Preprocessor and Privacy Risk Classifier"),
                                                                            html.Li("Knowledge Layer: Ontology Store and Risk Database"),
                                                                            html.Li("Application Layer: Privacy Dashboard and API Gateway"),
                                                                            html.Li("Cloud Services: Analytics Engine and Policy Manager")
                                                                        ]),
                                                                        html.P("This diagram reflects a distributed, layered architecture, supporting edge computing, centralized knowledge, and scalable cloud analytics, ensuring privacy control at every stage of data flow.", className="mt-3")
                                                                    ], width=6)
                                                                ])
                                                            ])
                                                        ], style=CUSTOM_STYLES["diagram-card"], className="mb-4"),
                                                        
                                                        # Sequence Diagram Section
                                                                dbc.Card([
                                                            dbc.CardHeader(html.H4("Sequence Diagram - Privacy Risk Assessment Workflow", className="text-center", style=CUSTOM_STYLES["diagram-title"])),
                                                                    dbc.CardBody([
                                                                dbc.Row([
                                                                    dbc.Col([
                                                                        html.Div([
                                                                            html.Img(
                                                                                src=f"data:image/png;base64,{load_uml_diagrams()['Privacy Risk Assessment Sequence']}",
                                                                                style={"width": "100%", "height": "auto", "objectFit": "contain"},
                                                                            )
                                                                        ], style=CUSTOM_STYLES["diagram-container"])
                                                            ], width=6),
                                                                    dbc.Col([
                                                                        html.P("This sequence diagram shows a step-by-step interaction flow between actors and system components during a privacy risk assessment request from the user.", className="mb-3"),
                                                                        html.H6("Interaction Flow:", className="mb-2"),
                                                                        html.Ol([
                                                                            html.Li("User sends a risk assessment request via the Privacy Dashboard"),
                                                                            html.Li("Dashboard collects device data from the IoT Device"),
                                                                            html.Li("It queries the Privacy Ontology Handler for relevant rules"),
                                                                            html.Li("Sends the data to the Privacy Risk Classifier for risk evaluation"),
                                                                            html.Li("Results are stored in the Data Store"),
                                                                            html.Li("Dashboard receives and displays the assessment result to the user")
                                                                        ]),
                                                                        html.P("This diagram highlights the real-time privacy assessment workflow, showing how semantic rules and ML models work together to deliver quick and context-aware privacy feedback to users.", className="mt-3")
                                                                    ], width=6)
                                                                ])
                                                            ])
                                                        ], style=CUSTOM_STYLES["diagram-card"])
                                                    ], className="p-4")
                                                ],
                                                label="UML Diagrams",
                                                tab_id="tab-uml"
                                            ),
                                            dbc.Tab(
                                                [
                                                    html.Div([
                                                        html.H3("Privacy Ontology", style=CUSTOM_STYLES["section-title"]),
                                                        dbc.Card([
                                                            dbc.CardHeader(html.H4("Privacy Knowledge Model", style=CUSTOM_STYLES["diagram-title"])),
                                                            dbc.CardBody([
                                                                html.Div([
                                                                    html.Img(
                                                                        src=f"data:image/jpg;base64,{load_ontology_diagram()}",
                                                                        style={"width": "100%", "maxWidth": "800px", "height": "auto", "margin": "auto", "display": "block"}
                                                                    )
                                                                ], style=CUSTOM_STYLES["diagram-container"])
                                                            ])
                                                        ], style=CUSTOM_STYLES["ontology-card"]),
                                                        
                                                        # Core Classes Section
        html.Div([
                                                            html.H4("Core Classes", style=CUSTOM_STYLES["diagram-title"], className="mt-4"),
                                                            dbc.Row([
                                                                dbc.Col([
                                                                    dbc.Card([
                                                                        dbc.CardHeader(html.H5("Personal Data")),
                                                                        dbc.CardBody([
                                                                            html.P("Represents any personal data collected or processed"),
                                                                            html.H6("Subclasses:", className="mt-3"),
                                                                            html.Ul([
                                                                                html.Li("SensitivePersonalData"),
                                                                                html.Li("BiometricData"),
                                                                                html.Li("HealthData"),
                                                                                html.Li("LocationData"),
                                                                                html.Li("BehavioralData")
                                                                            ])
                                                                        ])
                                                                    ], className="h-100")
                                                                ], width=4),
                                                                dbc.Col([
                                                                    dbc.Card([
                                                                        dbc.CardHeader(html.H5("Privacy Policy")),
                                                                        dbc.CardBody([
                                                                            html.P("Defines rules and regulations for data handling"),
                                                                            html.H6("Subclasses:", className="mt-3"),
                                                                            html.Ul([
                                                                                html.Li("LegalFramework"),
                                                                                html.Li("OrganizationalPolicy")
                                                                            ])
                                                                        ])
                                                                    ], className="h-100")
                                                                ], width=4),
                                                                dbc.Col([
                                                                    dbc.Card([
                                                                        dbc.CardHeader(html.H5("Risk")),
                                                                        dbc.CardBody([
                                                                            html.P("Represents privacy risks associated with data processing"),
                                                                            html.H6("Subclasses:", className="mt-3"),
                                                                            html.Ul([
                                                                                html.Li("SecurityRisk"),
                                                                                html.Li("PrivacyRisk"),
                                                                                html.Li("ComplianceRisk")
                                                                            ])
                                                                        ])
                                                                    ], className="h-100")
                                                                ], width=4)
                                                            ], className="mb-4"),
                                                            
                                                            # Data Processing Section
                                                            html.H4("Data Processing", style=CUSTOM_STYLES["diagram-title"], className="mt-4"),
                                                            dbc.Row([
                                                                dbc.Col([
                                                                    dbc.Card([
                                                                        dbc.CardHeader(html.H5("Data Processing Operations")),
                                                                        dbc.CardBody([
                                                                            html.P("Represents operations performed on personal data"),
                                                                            html.H6("Subclasses:", className="mt-3"),
                                                                            html.Ul([
                                                                                html.Li("Collection"),
                                                                                html.Li("Storage"),
                                                                                html.Li("Analysis"),
                                                                                html.Li("Sharing")
                                                                            ])
                                                                        ])
                                                                    ], className="h-100")
                                                                ], width=6),
                                                                dbc.Col([
                                                                    dbc.Card([
                                                                        dbc.CardHeader(html.H5("Consent")),
                                                                        dbc.CardBody([
                                                                            html.P("Represents user consent for data processing"),
                                                                            html.H6("Subclasses:", className="mt-3"),
                                                                            html.Ul([
                                                                                html.Li("ExplicitConsent"),
                                                                                html.Li("ImpliedConsent")
                                                                            ])
                                                                        ])
                                                                    ], className="h-100")
                                                                ], width=6)
                                                            ], className="mb-4"),
                                                            
                                                            # Properties Section
                                                            html.H4("Properties", style=CUSTOM_STYLES["diagram-title"], className="mt-4"),
                                                            dbc.Row([
                                                                dbc.Col([
                                                                    dbc.Card([
                                                                        dbc.CardHeader(html.H5("Object Properties")),
                                                                        dbc.CardBody([
                                                                            html.Ul([
                                                                                html.Li("ownsData: Links users to their personal data"),
                                                                                html.Li("regulatedBy: Links data to privacy policies"),
                                                                                html.Li("introducesRisk: Links processing to risks"),
                                                                                html.Li("hasConsent: Links users to consent decisions"),
                                                                                html.Li("implementsMeasure: Links controllers to security measures"),
                                                                                html.Li("transfersTo: Indicates data transfers"),
                                                                                html.Li("processesData: Links processors to data"),
                                                                                html.Li("mitigatesRisk: Links measures to risks"),
                                                                                html.Li("requiresConsent: Links processing to consent")
                                                                            ])
                                                                        ])
                                                                    ], className="h-100")
                                                                ], width=6),
                                                                dbc.Col([
                                                                    dbc.Card([
                                                                        dbc.CardHeader(html.H5("Data Properties")),
                                                                        dbc.CardBody([
                                                                            html.Ul([
                                                                                html.Li("dataType: Type of personal data"),
                                                                                html.Li("riskLevel: Numeric level of risk (1-5)"),
                                                                                html.Li("consentStatus: Whether consent is given"),
                                                                                html.Li("processingPurpose: Purpose of processing"),
                                                                                html.Li("retentionPeriod: Data retention duration"),
                                                                                html.Li("encryptionLevel: Level of encryption"),
                                                                                html.Li("transferLocation: Location of transfer"),
                                                                                html.Li("dataSensitivity: Sensitivity level"),
                                                                                html.Li("processingFrequency: Processing frequency")
                                                                            ])
                                                                        ])
                                                                    ], className="h-100")
                                                                ], width=6)
                                                            ], className="mb-4"),
                                                            
                                                            # Key Concepts Section
                                                            html.H4("Key Concepts", style=CUSTOM_STYLES["diagram-title"], className="mt-4"),
                                                            dbc.Row([
                                                                dbc.Col([
                                                                    dbc.Card([
                                                                        dbc.CardHeader(html.H5("Data Controller")),
                                                                        dbc.CardBody(html.P("Entity that determines purposes and means of processing personal data"))
                                                                    ], className="h-100")
                                                                ], width=4),
                                                                dbc.Col([
                                                                    dbc.Card([
                                                                        dbc.CardHeader(html.H5("Data Processor")),
                                                                        dbc.CardBody(html.P("Entity that processes data on behalf of the controller"))
                                                                    ], className="h-100")
                                                                ], width=4),
                                                                dbc.Col([
                                                                    dbc.Card([
                                                                        dbc.CardHeader(html.H5("Security Measures")),
                                                                        dbc.CardBody([
                                                                            html.P("Technical and organizational security measures"),
                                                                            html.H6("Subclasses:", className="mt-3"),
                                                                            html.Ul([
                                                                                html.Li("TechnicalMeasure"),
                                                                                html.Li("OrganizationalMeasure")
                                                                            ])
                                                                        ])
                                                                    ], className="h-100")
                                                                ], width=4)
                                                            ], className="mb-4")
                                                        ], style=CUSTOM_STYLES["ontology-section"])
                                                    ], className="p-4")
                                                ],
                                                label="Ontology",
                                                tab_id="tab-ontology"
                                            ),
                                            dbc.Tab(
                                                [
                                                    html.Div([
                                                        html.H3("Risk Assessment Dashboard", style=CUSTOM_STYLES["section-title"]),
                                                        
                                                        # Risk Overview Cards
                                                        dbc.Row([
                                                            dbc.Col([
                                                                dbc.Card([
                                                                    dbc.CardBody([
                                                                        html.H4("High Risk Items", className="card-title text-danger"),
                                                                        html.H2(id="high-risk-count", className="display-4 text-center"),
                                                                        html.P("Devices requiring immediate attention", className="card-text text-muted")
                                                                    ])
                                                                ])
                                                            ], width=4),
                                                            dbc.Col([
                                                                dbc.Card([
                                                                    dbc.CardBody([
                                                                        html.H4("Medium Risk Items", className="card-title text-warning"),
                                                                        html.H2(id="medium-risk-count", className="display-4 text-center"),
                                                                        html.P("Devices requiring monitoring", className="card-text text-muted")
                                                                    ])
                                                                ])
                                                            ], width=4),
                                                            dbc.Col([
                                                                dbc.Card([
                                                                    dbc.CardBody([
                                                                        html.H4("Low Risk Items", className="card-title text-success"),
                                                                        html.H2(id="low-risk-count", className="display-4 text-center"),
                                                                        html.P("Devices with minimal risk", className="card-text text-muted")
                                                                    ])
                                                                ])
                                                            ], width=4),
                                                        ], className="mb-4"),
                                                        
                                                        # Active Policies and Alerts
                                                        dbc.Row([
                                                            dbc.Col([
                                                                dbc.Card([
                                                                    dbc.CardHeader(html.H4("Active Privacy Policies")),
                                                                    dbc.CardBody([
                                                                        html.H2(id="active-policies-count", className="display-4 text-center text-primary"),
                                                                        html.P("Currently enforced policies", className="text-muted text-center"),
                                                                        html.Div(id="policies-list")
                                                                    ])
                                                                ])
                                                            ], width=6),
                                                            dbc.Col([
                                                                dbc.Card([
                                                                    dbc.CardHeader(html.H4("Recent Privacy Alerts")),
                                                                    dbc.CardBody([
                                                                        html.H2(id="alerts-count", className="display-4 text-center text-danger"),
                                                                        html.P("Alerts requiring attention", className="text-muted text-center"),
                                                                        html.Div(id="alerts-list")
                                                                    ])
                                                                ])
                                                            ], width=6),
                                                        ], className="mb-4"),
                                                        
                                                        # Risk trend and recommendations
                                                        dbc.Row([
                                                            dbc.Col([
                                                                dbc.Card([
                                                                    dbc.CardHeader(html.H4("Privacy Risk Trend (Last 30 Days)")),
                                                                    dbc.CardBody([
                                                                        dcc.Graph(id="risk-trend-graph")
                                                                    ])
                                                                ])
                                                            ], width=12)
                                                        ])
                                                    ], className="p-4")
                                                ],
                                                label="Risk Assessment",
                                                tab_id="tab-risk"
                                            ),
                                            dbc.Tab(create_analysis_tab(),
                                                    label="Data Analysis",
                                                    tab_id="tab-analysis")
                                        ], id="home-tabs", active_tab="tab-uml", className="mt-4")
                                    ],
                                    label="Home",
                                    tab_id="tab-home",
                                    label_style=CUSTOM_STYLES["tab"],
                                    active_label_style=CUSTOM_STYLES["tab-selected"],
                                ),
                                dbc.Tab(
                                    docs_layout,
                                    label="Documentation",
                                    tab_id="tab-docs",
                                    label_style=CUSTOM_STYLES["tab"],
                                    active_label_style=CUSTOM_STYLES["tab-selected"],
                                ),
                                dbc.Tab(
                                    about_layout,
                                    label="About",
                                    tab_id="tab-about",
                                    label_style=CUSTOM_STYLES["tab"],
                                    active_label_style=CUSTOM_STYLES["tab-selected"],
                                ),
                            ],
                            id="tabs",
                            active_tab="tab-home",
                            className="mb-3",
                        ),
                    ]
                )
            ],
            style=CUSTOM_STYLES["tab-container"],
        ),
        
        # Store components
        dcc.Store(id="risk-data-store"),
        dcc.Interval(id="interval-component", interval=5000, n_intervals=0),
        
        # Font Awesome CSS
        html.Link(
            rel="stylesheet",
            href="https://use.fontawesome.com/releases/v5.15.4/css/all.css",
            integrity="sha384-DyZ88mC6Up2uqS4h/KRgHuoeGwBcD4Ng9SiP4dIRy0EXTlnuz47vAwmeGwVChigm",
            crossOrigin="anonymous",
        ),
    ],
    fluid=True,
    style=CUSTOM_STYLES["main-container"],
)

# Callback for updating risk data
@app.callback(
    Output("risk-data-store", "data"),
    Input("interval-component", "n_intervals")
)
def update_risk_data(n):
    return generate_risk_data()

# Callback for updating analysis content
@app.callback(
    [
        Output("categorical-features-content", "children"),
        Output("numerical-features-content", "children"),
        Output("correlation-analysis-content", "children")
    ],
    [
        Input("view-mode-selector", "value"),
        Input("visualizations-store", "data")
    ]
)
def update_analysis_content(view_mode, visualizations):
    if not visualizations:
        return [
            dbc.Alert("No visualization data available", color="warning"),
            dbc.Alert("No visualization data available", color="warning"),
            dbc.Alert("No visualization data available", color="warning")
        ]
    
    def create_visualization_cards(category_data, view_mode):
        if not category_data.get(view_mode, {}):
            if view_mode == "static" and category_data.get("interactive", {}):
                return dbc.Alert(
                    [
                        html.H4("Static Visualizations Not Available", className="alert-heading"),
                        html.P("Static visualizations are not available. Please switch to Interactive View to see the visualizations."),
                        dbc.Button("Switch to Interactive View", id={"type": "switch-view-btn", "index": 0}, color="primary")
                    ],
                    color="info",
                    className="mb-4"
                )
            return dbc.Alert(f"No {view_mode} visualizations available", color="warning")
        
        cards = []
        data = category_data[view_mode]
        
        for title, content in data.items():
            card = dbc.Card([
                        dbc.CardHeader(html.H5(title.replace("_", " ").title())),
                        dbc.CardBody([
                            html.Img(
                        src=f"data:image/png;base64,{content}",
                                style={"width": "100%", "height": "auto"}
                    ) if view_mode == "static" else
                            html.Iframe(
                        srcDoc=content,
                                style={"width": "100%", "height": "600px", "border": "none"}
                            )
                        ])
            ], style=CUSTOM_STYLES["analysis-card"], className="mb-4")
    
            cards.append(dbc.Col(card, width=12 if "correlation" in title.lower() else 6))
        
        return dbc.Row(cards) if cards else dbc.Alert("No visualizations available", color="warning")
    
    try:
        categorical_content = create_visualization_cards(visualizations["categorical"], view_mode)
        numerical_content = create_visualization_cards(visualizations["numerical"], view_mode)
        correlation_content = create_visualization_cards(visualizations["correlations"], view_mode)
        return categorical_content, numerical_content, correlation_content
    
    except Exception as e:
        logger.error(f"Error updating analysis content: {str(e)}")
        return [
            dbc.Alert("Error loading visualizations", color="danger"),
            dbc.Alert("Error loading visualizations", color="danger"),
            dbc.Alert("Error loading visualizations", color="danger")
        ]

# Add callback for switching view mode
@app.callback(
    Output("view-mode-selector", "value"),
    Input({"type": "switch-view-btn", "index": ALL}, "n_clicks"),
    State("view-mode-selector", "value"),
    prevent_initial_call=True
)
def switch_view_mode(n_clicks, current_value):
    if not any(n_clicks):
        raise dash.exceptions.PreventUpdate
    return "interactive" if current_value == "static" else "static"

# Add custom CSS
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>Privacy Modelling and Analysis Dashboard</title>
        {%favicon%}
        {%css%}
        <style>
            /* Global Styles */
            body {
                background-color: #f8f9fa;
                font-family: 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            }

            /* Navigation Styles */
            .navbar {
                backdrop-filter: blur(10px);
                -webkit-backdrop-filter: blur(10px);
                background-color: rgba(255, 255, 255, 0.95) !important;
                box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
            }

            .nav-link-hover {
                color: #2c3e50 !important;
                transition: all 0.3s ease;
                position: relative;
                font-weight: 500;
                padding: 0.5rem 1rem;
                margin: 0 0.25rem;
            }
            
            .nav-link-hover:hover {
                color: #3498db !important;
                transform: translateY(-1px);
            }
            
            .nav-link-hover::after {
                content: '';
                position: absolute;
                width: 0;
                height: 2px;
                bottom: -2px;
                left: 0;
                background-color: #3498db;
                transition: width 0.3s ease;
            }
            
            .nav-link-hover:hover::after {
                width: 100%;
            }

            /* Tab Styles */
            .nav-tabs {
                border-bottom: none;
                margin-bottom: 1rem;
            }
            
            .nav-tabs .nav-link {
                border: none;
                color: #2c3e50;
                font-weight: 500;
                padding: 1rem 1.5rem;
                transition: all 0.3s ease;
                border-radius: 8px;
                margin-right: 0.5rem;
            }
            
            .nav-tabs .nav-link.active {
                color: #ffffff;
                background-color: #3498db;
                box-shadow: 0 4px 6px rgba(52, 152, 219, 0.2);
            }
            
            .nav-tabs .nav-link:hover:not(.active) {
                background-color: rgba(52, 152, 219, 0.1);
                transform: translateY(-1px);
            }

            /* Card Styles */
            .card {
                border: none;
                border-radius: 12px;
                transition: all 0.3s ease;
                background: white;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
            }
            
            .card:hover {
                transform: translateY(-5px);
                box-shadow: 0 8px 15px rgba(0, 0, 0, 0.1);
            }

            .card-header {
                background-color: transparent;
                border-bottom: 1px solid rgba(0, 0, 0, 0.05);
                padding: 1.25rem;
            }

            .card-body {
                padding: 1.5rem;
            }

            /* Content Area Styles */
            .tab-content {
                padding: 1.5rem;
                background-color: transparent;
            }
            
            .tab-pane {
                animation: fadeIn 0.5s ease;
            }

            /* Alert Styles */
            .alert {
                border: none;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
            }

            /* Button Styles */
            .btn {
                border-radius: 8px;
                padding: 0.5rem 1.25rem;
                font-weight: 500;
                transition: all 0.3s ease;
            }

            .btn:hover {
                transform: translateY(-1px);
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }

            /* Animations */
            @keyframes fadeIn {
                from { 
                    opacity: 0; 
                    transform: translateY(20px); 
                }
                to { 
                    opacity: 1; 
                    transform: translateY(0); 
                }
            }

            /* Responsive Adjustments */
            @media (max-width: 768px) {
                .nav-tabs .nav-link {
                    padding: 0.75rem 1rem;
                }

                .card {
                    margin-bottom: 1rem;
                }
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

@app.callback(
    [
        Output("high-risk-count", "children"),
        Output("medium-risk-count", "children"),
        Output("low-risk-count", "children"),
        Output("active-policies-count", "children"),
        Output("alerts-count", "children"),
        Output("policies-list", "children"),
        Output("alerts-list", "children"),
        Output("risk-trend-graph", "figure")
    ],
    [Input("interval-component", "n_intervals")]
)
def update_risk_assessment(n):
    """Update the risk assessment dashboard."""
    try:
        # Get current risks from device categories
        all_risks = []
        for category, devices in device_categories.items():
            if devices:  # Check if devices list is not empty
                all_risks.extend([device.get("risk_level", 0) for device in devices])
        
        # Count risks by level (with safe defaults)
        high_risk = sum(1 for r in all_risks if r >= 4)
        medium_risk = sum(1 for r in all_risks if r == 3)
        low_risk = sum(1 for r in all_risks if r <= 2)
        
        # Get active policies (safely)
        policies = [p for p in policies_db.values() if hasattr(p, 'is_active') and p.is_active]
        active_policies_count = len(policies)
        
        # Get recent alerts
        recent_alerts = [r for r in risk_history if isinstance(r, dict) and not r.get("resolved", False)]
        alerts_count = len(recent_alerts)
        
        # Create policies list with device categories
        policies_list = dbc.ListGroup([
            dbc.ListGroupItem([
                html.H6(getattr(policy, 'name', 'Unnamed Policy'), className="mb-1"),
                html.P(getattr(policy, 'description', 'No description'), className="mb-1 text-muted"),
                html.Small(f"Data Types: {', '.join(str(dt) for dt in getattr(policy, 'data_types', []))}", className="text-muted"),
                html.Div([
                    html.H6("Affected Device Categories:", className="mt-2"),
                    html.Ul([
                        html.Li(category) for category, devices in device_categories.items()
                        if devices and any(device.get("risk_level", 0) >= 3 for device in devices)
                    ] or [html.Li("None")])
                ])
            ]) for policy in policies
        ]) if policies else dbc.Alert("No active policies", color="info")
        
        # Create alerts list with device categories
        alerts_list = dbc.ListGroup([
            dbc.ListGroupItem([
                html.H6(f"Risk Level: {alert.get('risk_level', 'Unknown')}", className="mb-1"),
                html.P(alert.get('mitigation_suggestions', ['No suggestion'])[0] 
                      if alert.get('mitigation_suggestions') else "No suggestions", 
                      className="mb-1 text-muted"),
                html.Small(f"Device ID: {alert.get('device_id', 'Unknown')}", className="text-muted"),
                html.Div([
                    html.H6("Device Category:", className="mt-2"),
                    html.P(next((
                        category for category, devices in device_categories.items()
                        if devices and any(device.get("id") == alert.get("device_id") for device in devices)
                    ), "Unknown"))
                ])
            ]) for alert in recent_alerts
        ]) if recent_alerts else dbc.Alert("No active alerts", color="success")
        
        # Create risk trend graph with device categories
        fig = go.Figure()
        
        # Create dataframe for risk history if data exists
        if risk_history and isinstance(risk_history, list) and len(risk_history) > 0:
            try:
                # Convert risk history to DataFrame
                risk_history_df = pd.DataFrame(risk_history)
                
                # Ensure timestamp column exists and is datetime
                if 'timestamp' in risk_history_df.columns:
                    risk_history_df['date'] = pd.to_datetime(risk_history_df['timestamp'], errors='coerce')
                    risk_history_df = risk_history_df.dropna(subset=['date'])
                    
                    if not risk_history_df.empty:
                        # Set date as index
                        risk_history_df = risk_history_df.set_index('date')
                        
                        # Convert risk_level to numeric, replacing non-numeric values with NaN
                        if 'risk_level' in risk_history_df.columns:
                            risk_history_df['risk_level'] = pd.to_numeric(risk_history_df['risk_level'], errors='coerce')
                        
                        # Resample daily and calculate mean for numeric columns only
                        numeric_cols = risk_history_df.select_dtypes(include=['int64', 'float64']).columns
                        risk_history_df = risk_history_df[numeric_cols].resample('D').mean().fillna(0)
                        
                        # Add overall risk level trace
                        if 'risk_level' in risk_history_df.columns:
                            fig.add_trace(go.Scatter(
                                x=risk_history_df.index,
                                y=risk_history_df['risk_level'],
                                mode='lines+markers',
                                name='Overall Risk Level',
                                line=dict(color='#3498db', width=2),
                                marker=dict(size=8, symbol='circle')
                            ))
                        
                        # Add traces for each device category
                        category_colors = {
                            'Smart Home': '#e74c3c',
                            'Healthcare': '#2ecc71',
                            'Transportation': '#f1c40f',
                            'Others': '#1abc9c'
                        }
                        
                        # Initialize empty lists for each category
                        category_data = {cat: [] for cat in category_colors.keys()}
                        
                        # Process data for each timestamp
                        for timestamp in risk_history_df.index:
                            # Process each category
                            for category in category_colors.keys():
                                devices = device_categories.get(category, [])
                                if devices and isinstance(devices, list):
                                    # Get devices in this category
                                    category_devices = [device for device in devices if device.get("id")]
                                    
                                    if category_devices:
                                        # Get risk levels for devices in this category
                                        device_risks = []
                                        for device in category_devices:
                                            device_id = device.get("id")
                                            risk_col = f'device_{device_id}_risk'
                                            if risk_col in risk_history_df.columns:
                                                device_risks.append(risk_history_df.at[timestamp, risk_col])
                                            else:
                                                device_risks.append(device.get("risk_level", 0))
                                        
                                        # Calculate average, handling NaN values
                                        avg_risk = np.mean([r for r in device_risks if pd.notnull(r)]) if device_risks else 0
                                    else:
                                        avg_risk = 0
                                else:
                                    avg_risk = 0
                                
                                category_data[category].append(avg_risk)
                        
                        # Add traces for all categories
                        for category, risks in category_data.items():
                            color = category_colors[category]
                            fig.add_trace(go.Scatter(
                                x=risk_history_df.index,
                                y=risks,
                                mode='lines',
                                name=f'{category} Average',
                                line=dict(dash='dash', color=color, width=2),
                                opacity=0.8 if any(risks) else 0.3  # Lower opacity if no data
                            ))
                        
                        # Update layout
                        fig.update_layout(
                            title="Risk Level Trend by Device Category",
                            xaxis_title="Date",
                            yaxis_title="Risk Level (0-5)",
                            template="plotly_white",
                            margin=dict(l=40, r=150, t=40, b=40),  # Increased right margin for legend
                            legend=dict(
                                orientation="v",  # Vertical orientation
                                yanchor="top",
                                y=1,
                                xanchor="left",
                                x=1.02,  # Position to the right of the graph
                                font=dict(size=12),
                                bgcolor='rgba(255, 255, 255, 0.8)',  # Semi-transparent white background
                                bordercolor='rgba(0, 0, 0, 0.2)',  # Light border
                                borderwidth=1
                            ),
                            xaxis=dict(
                                showgrid=True,
                                gridwidth=1,
                                gridcolor='#f0f0f0',
                                showline=True,
                                linewidth=1,
                                linecolor='#d3d3d3'
                            ),
                            yaxis=dict(
                                showgrid=True,
                                gridwidth=1,
                                gridcolor='#f0f0f0',
                                range=[0, 5],  # Set fixed range for risk levels
                                showline=True,
                                linewidth=1,
                                linecolor='#d3d3d3'
                            ),
                            plot_bgcolor='white',
                            paper_bgcolor='white',
                            showlegend=True
                        )
                        
                        # Update trace properties for better visibility
                        for trace in fig.data:
                            if trace.name == 'Overall Risk Level':
                                trace.update(
                                    line=dict(width=5),  # Thicker line for overall risk
                                    marker=dict(size=10)  # Larger markers
                                )
                            else:
                                trace.update(
                                    line=dict(width=4),
                                    opacity=0.8  # Ensure all category lines are visible
                                )
                    else:
                        fig.update_layout(
                            title="No Valid Risk Data Available",
                            template="plotly_white",
                            margin=dict(l=40, r=40, t=40, b=40)
                        )
                else:
                    fig.update_layout(
                        title="Missing Timestamp Data",
                        template="plotly_white",
                        margin=dict(l=40, r=40, t=40, b=40)
                    )
            except Exception as e:
                logger.error(f"Error creating risk trend graph: {str(e)}")
                fig.update_layout(
                    title="Error Creating Risk Trend Graph",
                    template="plotly_white",
                    margin=dict(l=40, r=40, t=40, b=40)
                )
        else:
            fig.update_layout(
                title="No Risk History Data Available",
                template="plotly_white",
                margin=dict(l=40, r=40, t=40, b=40)
            )
        
        return (
            high_risk,
            medium_risk,
            low_risk,
            active_policies_count,
            alerts_count,
            policies_list,
            alerts_list,
            fig
        )
        
    except Exception as e:
        logger.error(f"Error updating risk assessment: {str(e)}")
        # Return safe default values
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="Error Loading Risk Data",
            template="plotly_white",
            margin=dict(l=40, r=40, t=40, b=40)
        )
        
        return (
            0, 0, 0, 0, 0,
            dbc.Alert("Error loading policies: " + str(e), color="danger"),
            dbc.Alert("Error loading alerts: " + str(e), color="danger"),
            empty_fig
        )

if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=8080, debug=False)



