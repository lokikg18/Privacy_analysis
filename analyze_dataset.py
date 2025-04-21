import logging
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define feature categories
CATEGORICAL_FEATURES = [
    "device_type",
    "data_type",
    "location_type",
    "access_pattern",
    "data_sharing",
    "compliance_status",
]

NUMERICAL_FEATURES = [
    "access_frequency",
    "network_security_level",
    "data_sensitivity",
    "encryption_level",
    "retention_period",
    "data_volume",
    "last_audit_days",
    "storage_duration",
    "security_incidents",
]

def setup_analysis_directories():
    """Create analysis directories if they don't exist."""
    try:
        project_root = Path.cwd()
        analysis_dir = project_root / "data" / "analysis"
        
        # Create main subdirectories
        subdirs = ["categorical", "numerical", "correlations"]
        for subdir in subdirs:
            dir_path = analysis_dir / subdir
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {dir_path}")
            
        return analysis_dir
    except Exception as e:
        logger.error(f"Error creating analysis directories: {e}")
        raise

def analyze_categorical_features(df: pd.DataFrame, analysis_dir: Path) -> None:
    """Generate interactive visualizations for categorical features."""
    categorical_dir = analysis_dir / "categorical"
    
    for feature in CATEGORICAL_FEATURES:
        try:
            # Create DataFrame for plotting
            value_counts = df[feature].value_counts()
            plot_df = pd.DataFrame({
                'Category': value_counts.index,
                'Count': value_counts.values
            })
            
            # Create distribution plot using plotly
            fig = px.bar(
                plot_df,
                x='Category',
                y='Count',
                title=f'Distribution of {feature.replace("_", " ").title()}',
                template='plotly_white'
            )
            
            # Add styling
            fig.update_layout(
                showlegend=False,
                title_x=0.5,
                title_font_size=20,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                hoverlabel=dict(bgcolor="white", font_size=14),
                margin=dict(t=50, l=50, r=30, b=50),
                xaxis_title=feature.replace("_", " ").title(),
                yaxis_title="Count",
                xaxis=dict(tickangle=45)
            )
            
            # Add value labels on top of bars
            fig.update_traces(
                texttemplate='%{y}',
                textposition='outside'
            )
            
            # Save as HTML for interactivity
            output_html = categorical_dir / f"{feature}_distribution.html"
            fig.write_html(str(output_html))
            logger.info(f"Saved {feature} interactive plot to {output_html}")
            
            # Save as PNG for static view
            output_png = categorical_dir / f"{feature}_distribution.png"
            fig.write_image(str(output_png))
            logger.info(f"Saved {feature} static plot to {output_png}")
            
        except Exception as e:
            logger.error(f"Error creating plot for {feature}: {str(e)}")
            continue

def analyze_numerical_features(df: pd.DataFrame, analysis_dir: Path) -> None:
    """Generate interactive visualizations for numerical features."""
    numerical_dir = analysis_dir / "numerical"
    
    for feature in NUMERICAL_FEATURES:
        try:
            # Create distribution plot using plotly
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=(
                    f'Distribution of {feature.replace("_", " ").title()}',
                    f'Box Plot of {feature.replace("_", " ").title()}'
                ),
                vertical_spacing=0.2
            )
            
            # Add histogram
            fig.add_trace(
                go.Histogram(
                    x=df[feature],
                    name='Distribution',
                    nbinsx=30,
                    showlegend=False
                ),
                row=1, col=1
            )
            
            # Add box plot
            fig.add_trace(
                go.Box(
                    y=df[feature],
                    name='Box Plot',
                    showlegend=False,
                    boxpoints='outliers'
                ),
                row=2, col=1
            )
            
            # Update layout
            fig.update_layout(
                height=800,
                title_x=0.5,
                title_font_size=20,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                hoverlabel=dict(bgcolor="white", font_size=14),
                margin=dict(t=50, l=50, r=30, b=50)
            )
            
            # Update axes labels
            fig.update_xaxes(title_text="Value", row=1, col=1)
            fig.update_yaxes(title_text="Count", row=1, col=1)
            fig.update_yaxes(title_text="Value", row=2, col=1)
            
            # Save as HTML for interactivity
            output_html = numerical_dir / f"{feature}_distribution.html"
            fig.write_html(str(output_html))
            logger.info(f"Saved {feature} interactive plot to {output_html}")
            
            # Save as PNG for static view
            output_png = numerical_dir / f"{feature}_distribution.png"
            fig.write_image(str(output_png))
            logger.info(f"Saved {feature} static plot to {output_png}")
            
        except Exception as e:
            logger.error(f"Error creating plot for {feature}: {str(e)}")
            continue

def analyze_correlations(df: pd.DataFrame, analysis_dir: Path) -> None:
    """Generate correlation analysis visualizations."""
    correlations_dir = analysis_dir / "correlations"
    
    try:
        # Calculate correlations
        numerical_df = df[NUMERICAL_FEATURES]
        corr_matrix = numerical_df.corr()
        
        # Create heatmap using plotly
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmin=-1,
            zmax=1,
            text=np.round(corr_matrix, 2),
            texttemplate='%{text}',
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        # Update layout
        fig.update_layout(
            title='Feature Correlation Heatmap',
            title_x=0.5,
            width=1000,
            height=800,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            hoverlabel=dict(bgcolor="white", font_size=14),
            xaxis=dict(tickangle=45)
        )
        
        # Save as HTML for interactivity
        output_html = correlations_dir / "correlation_heatmap.html"
        fig.write_html(str(output_html))
        logger.info(f"Saved correlation heatmap interactive plot to {output_html}")
        
        # Save as PNG for static view
        output_png = correlations_dir / "correlation_heatmap.png"
        fig.write_image(str(output_png))
        logger.info(f"Saved correlation heatmap static plot to {output_png}")
        
        # Create feature importance plot for risk level
        if 'risk_level' in df.columns:
            correlations_with_risk = df[NUMERICAL_FEATURES + ['risk_level']].corr()['risk_level'].sort_values()
            
            fig = go.Figure(data=go.Bar(
                x=correlations_with_risk.index,
                y=correlations_with_risk.values,
                marker_color=np.where(correlations_with_risk > 0, 'darkred', 'darkblue')
            ))
            
            fig.update_layout(
                title='Feature Correlations with Risk Level',
                title_x=0.5,
                xaxis_title='Features',
                yaxis_title='Correlation Coefficient',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                hoverlabel=dict(bgcolor="white", font_size=14),
                xaxis=dict(tickangle=45),
                height=600
            )
            
            # Add value labels on bars
            fig.update_traces(
                texttemplate='%{y:.2f}',
                textposition='outside'
            )
            
            # Save as HTML for interactivity
            output_html = correlations_dir / "risk_correlations.html"
            fig.write_html(str(output_html))
            logger.info(f"Saved risk correlations interactive plot to {output_html}")
            
            # Save as PNG for static view
            output_png = correlations_dir / "risk_correlations.png"
            fig.write_image(str(output_png))
            logger.info(f"Saved risk correlations static plot to {output_png}")
            
    except Exception as e:
        logger.error(f"Error in correlation analysis: {str(e)}")
        raise

def main():
    try:
        # Setup directories
        analysis_dir = setup_analysis_directories()
        
        # Load the dataset
        project_root = Path.cwd()
        processed_dir = project_root / "data" / "processed"
        train_file = processed_dir / "train_dataset.csv"
        
        if not train_file.exists():
            raise FileNotFoundError(f"Dataset file not found at {train_file}")
        
        logger.info(f"Loading dataset from {train_file}")
        df = pd.read_csv(train_file)
        logger.info(f"Loaded {len(df)} samples")
        
        # Generate analyses
        analyze_categorical_features(df, analysis_dir)
        analyze_numerical_features(df, analysis_dir)
        analyze_correlations(df, analysis_dir)
        
        logger.info("Dataset analysis completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        raise

if __name__ == "__main__":
    main()
