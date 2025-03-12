"""
Generate Visualizations for Hospital Readmission Prediction Tutorial

This script creates a comprehensive set of visualizations based on the MIMIC
readmission prediction analysis results.

Usage:
    python generate_visualizations.py

Output:
    Creates various visualizations in the 'visualizations' directory
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from readmission_visualizations import (
    create_class_distribution_plot,
    create_demographic_visualizations,
    create_comorbidity_visualization,
    create_feature_importance_visualization,
    create_model_performance_visualization,
    create_roc_pr_curves,
    create_executive_summary_dashboard,
    create_risk_stratification_visualization
)

def main():
    """Generate all visualizations for the hospital readmission prediction tutorial"""
    
    # Create the visualizations directory if it doesn't exist
    os.makedirs('visualizations', exist_ok=True)
    
    print("Generating visualizations for hospital readmission prediction tutorial...")
    
    # Generate all visualizations
    create_class_distribution_plot()
    print("✓ Created class distribution visualizations")
    
    create_demographic_visualizations()
    print("✓ Created demographic visualizations")
    
    create_comorbidity_visualization()
    print("✓ Created comorbidity visualizations")
    
    create_feature_importance_visualization()
    print("✓ Created feature importance visualizations")
    
    create_model_performance_visualization()
    print("✓ Created model performance visualizations")
    
    create_roc_pr_curves()
    print("✓ Created ROC and Precision-Recall curves")
    
    create_executive_summary_dashboard()
    print("✓ Created executive summary dashboard")
    
    create_risk_stratification_visualization()
    print("✓ Created risk stratification visualizations")
    
    print("\nAll visualizations have been saved to the 'visualizations' directory.")
    print("These visualizations can be directly incorporated into your presentation slides.")

if __name__ == "__main__":
    main()