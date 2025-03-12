import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.impute import SimpleImputer
import joblib
import os
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure visualization settings
plt.style.use('ggplot')
sns.set(font_scale=1.2)

# Create output directory for plots
os.makedirs('plots', exist_ok=True)

def load_data():
    """
    Load and perform initial exploration of MIMIC data files
    """
    print("Loading MIMIC data files...")
    
    # Load the key tables with error handling
    try:
        admissions = pd.read_csv('ADMISSIONS.csv', low_memory=False)
        print(f"Admissions shape: {admissions.shape}")
        
        patients = pd.read_csv('PATIENTS.csv', low_memory=False)
        print(f"Patients shape: {patients.shape}")
        
        diagnoses = pd.read_csv('DIAGNOSES_ICD.csv', low_memory=False)
        print(f"Diagnoses shape: {diagnoses.shape}")
        
        procedures = pd.read_csv('PROCEDURES_ICD.csv', low_memory=False)
        print(f"Procedures shape: {procedures.shape}")
        
        prescriptions = pd.read_csv('PRESCRIPTIONS.csv', low_memory=False)
        print(f"Prescriptions shape: {prescriptions.shape}")
        
        try:
            labevents = pd.read_csv('LABEVENTS.csv', low_memory=False)
            print(f"Labevents shape: {labevents.shape}")
        except Exception as e:
            print(f"Warning: Could not load LABEVENTS.csv: {e}")
            print("Continuing without lab events data.")
            labevents = pd.DataFrame()  # Empty DataFrame
        
        return admissions, patients, diagnoses, procedures, prescriptions, labevents
        
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

def create_readmission_labels(admissions_df, window_days=30):
    """
    Create labels for readmissions within specified window
    
    Parameters:
    -----------
    admissions_df : pandas DataFrame
        The admissions table with admission and discharge times
    window_days : int
        Number of days within which a readmission is counted
        
    Returns:
    --------
    DataFrame with readmission labels
    """
    print(f"Generating {window_days}-day readmission labels...")
    
    # Create a copy to avoid modifying the original
    df = admissions_df.copy()
    
    # Convert admission and discharge times to datetime with error handling
    df['ADMITTIME'] = pd.to_datetime(df['ADMITTIME'], errors='coerce')
    df['DISCHTIME'] = pd.to_datetime(df['DISCHTIME'], errors='coerce')
    
    # Drop rows with missing admit or discharge times
    valid_dates = (~df['ADMITTIME'].isna()) & (~df['DISCHTIME'].isna())
    if not valid_dates.all():
        print(f"Warning: Dropping {(~valid_dates).sum()} rows with invalid dates")
        df = df[valid_dates].copy()
    
    # Initialize readmission column
    df['READMISSION_30D'] = 0
    
    # Sort admissions by patient and admission time
    df = df.sort_values(['SUBJECT_ID', 'ADMITTIME'])
    
    # Group by patient
    patient_count = len(df['SUBJECT_ID'].unique())
    print(f"Processing readmissions for {patient_count} patients...")
    
    readmission_count = 0
    
    for patient_id, patient_df in df.groupby('SUBJECT_ID'):
        # Skip if patient has only one admission
        if len(patient_df) < 2:
            continue
            
        # Convert to list for easier handling
        admit_times = patient_df['ADMITTIME'].tolist()
        discharge_times = patient_df['DISCHTIME'].tolist()
        hadm_ids = patient_df['HADM_ID'].tolist()
        
        # Check each admission (except the last one)
        for i in range(len(admit_times) - 1):
            # Calculate time difference between current discharge and next admission
            time_diff = (admit_times[i+1] - discharge_times[i]).total_seconds() / (60*60*24)
            
            # Mark as readmission if within window
            if time_diff <= window_days:
                # Find the index of this admission in the dataframe
                idx = df[df['HADM_ID'] == hadm_ids[i]].index
                df.loc[idx, 'READMISSION_30D'] = 1
                readmission_count += 1
    
    # Calculate readmission rate
    readmission_rate = df['READMISSION_30D'].mean() * 100
    print(f"{window_days}-day Readmission Rate: {readmission_rate:.2f}%")
    print(f"Found {readmission_count} readmissions out of {len(df)} admissions")
    
    return df

def create_features(admissions_df, patients_df, diagnoses_df, procedures_df, lab_df):
    """
    Create features for readmission prediction with robust date handling
    
    Returns:
    --------
    DataFrame with engineered features
    """
    print("Creating feature dataset...")
    
    # Make a copy to avoid modifying original dataframes
    admissions_df = admissions_df.copy()
    patients_df = patients_df.copy()
    
    # Ensure date columns are datetime objects
    admissions_df['ADMITTIME'] = pd.to_datetime(admissions_df['ADMITTIME'], errors='coerce')
    admissions_df['DISCHTIME'] = pd.to_datetime(admissions_df['DISCHTIME'], errors='coerce')
    patients_df['DOB'] = pd.to_datetime(patients_df['DOB'], errors='coerce')
    
    # Merge patient demographics
    features_df = admissions_df.merge(patients_df, on='SUBJECT_ID', how='left')
    print(f"After patient merge: {features_df.shape[0]} rows")
    
    # Calculate patient age at admission - with error handling
    features_df['AGE'] = np.nan  # Initialize with NaN
    
    # Filter out rows with valid dates for age calculation
    valid_dates_mask = (~pd.isna(features_df['ADMITTIME'])) & (~pd.isna(features_df['DOB']))
    
    print("Calculating age using year difference method")
    features_df['AGE'] = features_df.apply(
        lambda x: abs(x['ADMITTIME'].year - x['DOB'].year) - 
                ((x['ADMITTIME'].month, x['ADMITTIME'].day) < 
                 (x['DOB'].month, x['DOB'].day)) if pd.notna(x['ADMITTIME']) and pd.notna(x['DOB']) else np.nan,
        axis=1
    )
    
    # Cap age at 90 for privacy and handle invalid ages
    features_df['AGE'] = features_df['AGE'].clip(0, 90)
    
    # Create admission type features (one-hot encoding)
    if 'ADMISSION_TYPE' in features_df.columns:
        admission_type_dummies = pd.get_dummies(features_df['ADMISSION_TYPE'], prefix='ADM_TYPE')
        features_df = pd.concat([features_df, admission_type_dummies], axis=1)
    else:
        print("Warning: ADMISSION_TYPE column not found")
        # Create dummy columns to maintain consistency
        features_df['ADM_TYPE_ELECTIVE'] = 0
        features_df['ADM_TYPE_EMERGENCY'] = 0
        features_df['ADM_TYPE_URGENT'] = 0
    
    # Calculate length of stay safely
    print("Calculating length of stay")
    features_df['LOS_DAYS'] = features_df.apply(
        lambda x: (x['DISCHTIME'] - x['ADMITTIME']).total_seconds() / (24*60*60) 
                 if pd.notna(x['DISCHTIME']) and pd.notna(x['ADMITTIME']) else np.nan,
        axis=1
    )
    
    # Handle any negative LOS (due to data errors) by taking absolute value
    features_df['LOS_DAYS'] = features_df['LOS_DAYS'].abs()
    
    # Cap extreme LOS values
    features_df['LOS_DAYS'] = features_df['LOS_DAYS'].clip(0, 365)  # Cap at 1 year
    
    # Count previous admissions
    features_df['PREV_ADMISSIONS'] = features_df.groupby('SUBJECT_ID')['HADM_ID'].transform(
        lambda x: pd.Series(range(len(x)), index=x.index)
    )
    
    # Ensure no negative values
    features_df['PREV_ADMISSIONS'] = features_df['PREV_ADMISSIONS'].clip(0)
    
    # Count diagnoses per admission
    if not diagnoses_df.empty:
        diagnoses_count = diagnoses_df.groupby('HADM_ID').size().reset_index(name='DIAGNOSIS_COUNT')
        features_df = features_df.merge(diagnoses_count, on='HADM_ID', how='left')
        features_df['DIAGNOSIS_COUNT'] = features_df['DIAGNOSIS_COUNT'].fillna(0)
    else:
        features_df['DIAGNOSIS_COUNT'] = 0
        print("Warning: No diagnoses data available")
    
    # Count procedures per admission
    if not procedures_df.empty:
        procedure_count = procedures_df.groupby('HADM_ID').size().reset_index(name='PROCEDURE_COUNT')
        features_df = features_df.merge(procedure_count, on='HADM_ID', how='left')
        features_df['PROCEDURE_COUNT'] = features_df['PROCEDURE_COUNT'].fillna(0)
    else:
        features_df['PROCEDURE_COUNT'] = 0
        print("Warning: No procedures data available")
    
    # Calculate abnormal lab result percentage (if available)
    if not lab_df.empty and 'FLAG' in lab_df.columns:
        print("Processing lab event data...")
        abnormal_labs = lab_df[lab_df['FLAG'] == 'abnormal'].groupby('HADM_ID').size().reset_index(name='ABNORMAL_LABS')
        total_labs = lab_df.groupby('HADM_ID').size().reset_index(name='TOTAL_LABS')
        
        lab_metrics = abnormal_labs.merge(total_labs, on='HADM_ID', how='right')
        lab_metrics['ABNORMAL_LABS'] = lab_metrics['ABNORMAL_LABS'].fillna(0)
        lab_metrics['ABNORMAL_RATIO'] = lab_metrics['ABNORMAL_LABS'] / lab_metrics['TOTAL_LABS']
        
        features_df = features_df.merge(lab_metrics[['HADM_ID', 'ABNORMAL_RATIO']], on='HADM_ID', how='left')
        features_df['ABNORMAL_RATIO'] = features_df['ABNORMAL_RATIO'].fillna(0)
    else:
        print("Note: Lab events data not available or no FLAG column found")
    
    # Select and clean final feature set
    feature_columns = ['HADM_ID', 'SUBJECT_ID', 'GENDER', 'AGE', 'LOS_DAYS', 
                      'PREV_ADMISSIONS', 'DIAGNOSIS_COUNT', 'PROCEDURE_COUNT',
                      'READMISSION_30D']
    
    # Add admission type columns if they exist
    for col in ['ADM_TYPE_ELECTIVE', 'ADM_TYPE_EMERGENCY', 'ADM_TYPE_URGENT']:
        if col in features_df.columns:
            feature_columns.append(col)
    
    if 'ABNORMAL_RATIO' in features_df.columns:
        feature_columns.append('ABNORMAL_RATIO')
    
    # Create the final dataset
    result_df = features_df[feature_columns].copy()
    
    # Handle missing values
    initial_rows = len(result_df)
    result_df = result_df.dropna(subset=['AGE', 'LOS_DAYS'])
    final_rows = len(result_df)
    
    if initial_rows > final_rows:
        print(f"Dropped {initial_rows - final_rows} rows with missing essential features")
    
    print(f"Final feature dataset: {result_df.shape[0]} rows, {result_df.shape[1]} columns")
    
    return result_df

def add_comorbidity_features(features_df, diagnoses_df):
    """
    Add Elixhauser comorbidity features based on ICD codes
    
    Parameters:
    -----------
    features_df : pandas DataFrame
        Features dataframe with HADM_ID
    diagnoses_df : pandas DataFrame
        Diagnoses dataframe with ICD codes
        
    Returns:
    --------
    DataFrame with added comorbidity features
    """
    print("Adding comorbidity features...")
    
    if diagnoses_df.empty:
        print("Warning: No diagnoses data available for comorbidity calculation")
        features_df['COMORBIDITY_COUNT'] = 0
        return features_df
    
    # Check if the ICD code column exists
    icd_column = None
    for col in ['ICD9_CODE', 'ICD_CODE']:
        if col in diagnoses_df.columns:
            icd_column = col
            break
    
    if icd_column is None:
        print("Warning: No ICD code column found in diagnoses data")
        features_df['COMORBIDITY_COUNT'] = 0
        return features_df
    
    # Define Elixhauser comorbidities with their ICD-9 codes (simplified subset)
    comorbidities = {
        'CHF': ['428'],  # Congestive Heart Failure
        'ARRHY': ['426', '427'],  # Cardiac Arrhythmias
        'VALVE': ['394', '395', '396', '397', '424'],  # Valvular Disease
        'PULMCIRC': ['415', '416', '417'],  # Pulmonary Circulation Disorders
        'PERIVASC': ['440', '441', '442', '443', '444', '447'],  # Peripheral Vascular Disorders
        'HTN': ['401', '402', '403', '404', '405'],  # Hypertension
        'PARA': ['342', '343', '344'],  # Paralysis
        'NEURO': ['330', '331', '332', '333', '334', '335', '336', '337'],  # Neurological Disorders
        'CHRNLUNG': ['490', '491', '492', '493', '494', '495', '496', '500', '501', '502', '503', '504', '505'],  # Chronic Pulmonary Disease
        'DM': ['250'],  # Diabetes
        'RENLFAIL': ['585', '586', 'V56'],  # Renal Failure
        'LIVER': ['570', '571', '572', '573'],  # Liver Disease
        'ULCER': ['531', '532', '533', '534'],  # Peptic Ulcer Disease
        'CANCER': ['140', '141', '142', '143', '144', '145', '146', '147', '148', '149', '150', '151', '152', '153', '154', '155', '156', '157', '158', '159', '160', '161', '162', '163', '164', '165', '166', '167', '168', '169', '170', '171', '172', '173', '174', '175', '176', '177', '178', '179', '180', '181', '182', '183', '184', '185', '186', '187', '188', '189', '190', '191', '192', '193', '194', '195', '196', '197', '198', '199', '200', '201', '202', '203', '204', '205', '206', '207', '208', '209'],  # Cancer
        'DEPRESSION': ['300.4', '301.12', '309.0', '309.1', '311']  # Depression
    }
    
    # Filter diagnoses to relevant columns
    diag_subset = diagnoses_df[['HADM_ID', icd_column]].copy()
    
    # Clean ICD codes (remove dots and convert to string)
    diag_subset[icd_column] = diag_subset[icd_column].astype(str).str.replace('.', '')
    
    # Initialize comorbidity columns
    for comorbidity in comorbidities:
        features_df[f'CM_{comorbidity}'] = 0
    
    # Map diagnoses to comorbidities
    comorbidity_counts = {comorbidity: 0 for comorbidity in comorbidities}
    
    for comorbidity, icd_codes in comorbidities.items():
        for icd_code in icd_codes:
            # Find diagnoses with matching ICD code prefix
            matching_diagnoses = diag_subset[diag_subset[icd_column].str.startswith(icd_code, na=False)]
            
            # Mark the comorbidity for matching hospitalizations
            if not matching_diagnoses.empty:
                hadm_ids_with_comorbidity = matching_diagnoses['HADM_ID'].unique()
                
                # Update the comorbidity column
                mask = features_df['HADM_ID'].isin(hadm_ids_with_comorbidity)
                features_df.loc[mask, f'CM_{comorbidity}'] = 1
                
                # Count the patients with this comorbidity
                comorbidity_counts[comorbidity] += mask.sum()
    
    # Calculate Elixhauser comorbidity index (simplified version)
    comorbidity_columns = [col for col in features_df.columns if col.startswith('CM_')]
    features_df['COMORBIDITY_COUNT'] = features_df[comorbidity_columns].sum(axis=1)
    
    # Print comorbidity statistics
    print("\nComorbidity Statistics:")
    for comorbidity, count in sorted(comorbidity_counts.items(), key=lambda x: x[1], reverse=True):
        if count > 0:
            percentage = (count / len(features_df)) * 100
            print(f"  {comorbidity}: {count} patients ({percentage:.1f}%)")
    
    return features_df

def explore_data(feature_dataset):
    """
    Perform exploratory data analysis on the feature dataset
    """
    print("\nExploratory Data Analysis:")
    
    # Basic statistics
    print("\nFeature dataset summary:")
    print(feature_dataset.describe())
    
    # Check class distribution
    readmission_counts = feature_dataset['READMISSION_30D'].value_counts()
    print("\nClass distribution:")
    for label, count in readmission_counts.items():
        percentage = (count / len(feature_dataset)) * 100
        print(f"  Readmitted = {label}: {count} ({percentage:.1f}%)")
    
    # Gender distribution
    if 'GENDER' in feature_dataset.columns:
        gender_counts = feature_dataset['GENDER'].value_counts()
        print("\nGender distribution:")
        for gender, count in gender_counts.items():
            percentage = (count / len(feature_dataset)) * 100
            print(f"  {gender}: {count} ({percentage:.1f}%)")
    
    # Age distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(data=feature_dataset, x='AGE', hue='READMISSION_30D', bins=20, multiple='dodge')
    plt.title('Age Distribution by Readmission Status')
    plt.xlabel('Age')
    plt.ylabel('Count')
    plt.savefig('plots/age_distribution.png')
    print("Saved age distribution plot to 'plots/age_distribution.png'")
    
    # Length of stay distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(data=feature_dataset, x='LOS_DAYS', hue='READMISSION_30D', bins=20, multiple='dodge')
    plt.title('Length of Stay Distribution by Readmission Status')
    plt.xlabel('Length of Stay (days)')
    plt.ylabel('Count')
    plt.xlim(0, 30)  # Focus on stays up to 30 days
    plt.savefig('plots/los_distribution.png')
    print("Saved length of stay distribution plot to 'plots/los_distribution.png'")
    
    # Correlation heatmap of numerical features
    numerical_features = feature_dataset.select_dtypes(include=['float64', 'int64']).columns
    plt.figure(figsize=(12, 10))
    correlation = feature_dataset[numerical_features].corr()
    sns.heatmap(correlation, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('plots/correlation_heatmap.png')
    print("Saved correlation heatmap to 'plots/correlation_heatmap.png'")
    
    # Comorbidity analysis if available
    comorbidity_cols = [col for col in feature_dataset.columns if col.startswith('CM_')]
    if comorbidity_cols:
        plt.figure(figsize=(12, 8))
        comorbidity_counts = feature_dataset[comorbidity_cols].sum().sort_values(ascending=False)
        sns.barplot(x=comorbidity_counts.values, y=comorbidity_counts.index)
        plt.title('Comorbidity Frequency')
        plt.xlabel('Count')
        plt.tight_layout()
        plt.savefig('plots/comorbidity_frequency.png')
        print("Saved comorbidity frequency plot to 'plots/comorbidity_frequency.png'")
        
        # Comorbidity by readmission status
        comorbidity_by_readmission = feature_dataset.groupby('READMISSION_30D')[comorbidity_cols].mean()
        plt.figure(figsize=(12, 8))
        comorbidity_by_readmission.T.plot(kind='bar')
        plt.title('Comorbidity Frequency by Readmission Status')
        plt.ylabel('Frequency')
        plt.xlabel('Comorbidity')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('plots/comorbidity_by_readmission.png')
        print("Saved comorbidity by readmission plot to 'plots/comorbidity_by_readmission.png'")
    
    return feature_dataset

def train_evaluate_models(features_df, use_smote=True):
    """
    Train and evaluate multiple models for readmission prediction
    
    Parameters:
    -----------
    features_df : pandas DataFrame
        Feature dataset with target variable READMISSION_30D
    use_smote : bool
        Whether to use SMOTE for handling class imbalance
    
    Returns:
    --------
    Dictionary containing trained models and evaluation metrics
    """
    print("\nPreparing data for model training...")
    
    # Prepare the dataset
    X = features_df.drop(['HADM_ID', 'SUBJECT_ID', 'READMISSION_30D'], axis=1)
    y = features_df['READMISSION_30D']
    
    # Convert categorical variables to numeric
    X = pd.get_dummies(X, drop_first=True)
    
    # Impute missing values
    imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    
    # Scale numerical features
    scaler = StandardScaler()
    numerical_cols = X.select_dtypes(include=['float64', 'int64']).columns
    X_imputed[numerical_cols] = scaler.fit_transform(X_imputed[numerical_cols])
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X_imputed, y, test_size=0.25, random_state=42, stratify=y)
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Testing set: {X_test.shape[0]} samples")
    print(f"Positive class (readmissions) in training: {sum(y_train)}/{len(y_train)} ({100*sum(y_train)/len(y_train):.2f}%)")
    
    # Apply SMOTE to handle class imbalance if requested
    if use_smote:
        try:
            from imblearn.over_sampling import SMOTE
            smote = SMOTE(random_state=42)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
            print(f"Applied SMOTE: Training set size increased from {len(X_train)} to {len(X_train_resampled)}")
            print(f"Class distribution after SMOTE: {np.bincount(y_train_resampled)}")
        except ImportError:
            print("Warning: imblearn not installed. Continuing without SMOTE.")
            X_train_resampled, y_train_resampled = X_train, y_train
    else:
        X_train_resampled, y_train_resampled = X_train, y_train
    
    # Initialize models
    models = {
        'logistic': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
        'random_forest': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    }
    
    # Add XGBoost if available
    try:
        import xgboost as xgb
        models['xgboost'] = xgb.XGBClassifier(
            scale_pos_weight=len(y_train[y_train==0])/len(y_train[y_train==1]),
            random_state=42
        )
    except ImportError:
        print("Warning: XGBoost not installed. Skipping XGBoost model.")
    
    # Train and evaluate each model
    results = {}
    for name, model in models.items():
        print(f"\nTraining {name} model...")
        
        # Train the model
        model.fit(X_train_resampled, y_train_resampled)
        
        # Make predictions
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        # Calculate metrics
        auc = roc_auc_score(y_test, y_pred_proba)
        average_precision = average_precision_score(y_test, y_pred_proba)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        # Store results
        results[name] = {
            'model': model,
            'auc': auc,
            'avg_precision': average_precision,
            'confusion_matrix': conf_matrix,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        # Print results
        print(f"Model: {name}")
        print(f"AUC: {auc:.4f}")
        print(f"Average Precision: {average_precision:.4f}")
        print("\nConfusion Matrix:")
        print(conf_matrix)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Plot ROC curve
        plt.figure(figsize=(8, 6))
        RocCurveDisplay.from_predictions(
            y_test,
            y_pred_proba,
            name=f"{name} ROC curve (area = {auc:.3f})",
            plot_chance_level=True
        )
        plt.title(f'ROC Curve - {name}')
        plt.savefig(f'plots/roc_curve_{name}.png')
        print(f"Saved ROC curve to 'plots/roc_curve_{name}.png'")
        
        # For Random Forest and XGBoost, get feature importance
        if name in ['random_forest', 'xgboost']:
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                feature_importance = pd.DataFrame({
                    'Feature': X_train.columns,
                    'Importance': importances
                }).sort_values('Importance', ascending=False)
                
                print("\nTop 10 Important Features:")
                print(feature_importance.head(10))
                
                # Plot feature importance
                plt.figure(figsize=(10, 8))
                sns.barplot(x='Importance', y='Feature', data=feature_importance.head(15))
                plt.title(f'Top 15 Features - {name}')
                plt.tight_layout()
                plt.savefig(f'plots/feature_importance_{name}.png')
                print(f"Saved feature importance plot to 'plots/feature_importance_{name}.png'")
    
    # Compare models
    plt.figure(figsize=(10, 6))
    for name, result in results.items():
        PrecisionRecallDisplay.from_predictions(
            result['y_test'],
            result['y_pred_proba'],
            name=f"{name} (AP = {result['avg_precision']:.3f})"
        )
    plt.title('Precision-Recall Curves for All Models')
    plt.savefig('plots/precision_recall_comparison.png')
    print("Saved precision-recall comparison to 'plots/precision_recall_comparison.png'")
    
    # Determine best model based on AUC
    best_model_name = max(results, key=lambda k: results[k]['auc'])
    print(f"\nBest performing model: {best_model_name} (AUC: {results[best_model_name]['auc']:.4f})")
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    joblib.dump(results[best_model_name]['model'], f'models/readmission_model_{best_model_name}.joblib')
    print(f"Saved best model to 'models/readmission_model_{best_model_name}.joblib'")
    
    return results, X_train.columns

def predict_readmission_risk(model, new_patient_data, feature_names):
    """
    Predict readmission risk for a new patient
    
    Parameters:
    -----------
    model : trained sklearn model
    new_patient_data : dict
        Dictionary containing patient features
    feature_names : list
        List of feature names used by the model
        
    Returns:
    --------
    Readmission probability
    """
    # Convert patient data to DataFrame
    patient_df = pd.DataFrame([new_patient_data])
    
    # Ensure all features are present
    for feature in feature_names:
        if feature not in patient_df.columns:
            patient_df[feature] = 0
    
    # Select only the features used by the model
    patient_df = patient_df[feature_names]
    
    # Make prediction
    readmission_prob = model.predict_proba(patient_df)[0, 1]
    
    return readmission_prob

def main():
    """
    Main function to run the full pipeline
    """
    # Initialize all variables to None
    admissions = None
    patients = None 
    diagnoses = None
    procedures = None
    prescriptions = None
    labevents = None
    admissions_with_labels = None
    feature_dataset = None
    
    # Load the data
    try:
        admissions, patients, diagnoses, procedures, prescriptions, labevents = load_data()
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
    
    # Create readmission labels
    try:
        admissions_with_labels = create_readmission_labels(admissions)
    except Exception as e:
        print(f"Error creating readmission labels: {e}")
        return None
    
    # Create feature dataset
    try:
        feature_dataset = create_features(
            admissions_with_labels, 
            patients, 
            diagnoses, 
            procedures, 
            labevents
        )
    except Exception as e:
        print(f"Error creating features: {e}")
        return None
    
    # Add comorbidity features
    try:
        feature_dataset = add_comorbidity_features(feature_dataset, diagnoses)
    except Exception as e:
        print(f"Error adding comorbidity features: {e}")
        # Continue without comorbidities if there's an error
    
    # Explore the data
    try:
        explore_data(feature_dataset)
    except Exception as e:
        print(f"Warning: Error during data exploration: {e}")
        # Continue even if visualization fails
    
    # Train and evaluate models
    try:
        model_results, feature_names = train_evaluate_models(feature_dataset)
        
        # Example of how to use the model for prediction
        print("\nExample prediction with best model:")
        best_model_name = max(model_results, key=lambda k: model_results[k]['auc'])
        best_model = model_results[best_model_name]['model']
        
        # Example patient
        example_patient = {
            'AGE': 65,
            'GENDER_M': 1,  # Assuming GENDER was one-hot encoded with GENDER_M
            'LOS_DAYS': 5.2,
            'PREV_ADMISSIONS': 2,
            'DIAGNOSIS_COUNT': 8,
            'PROCEDURE_COUNT': 3,
            'COMORBIDITY_COUNT': 4,
            'ADM_TYPE_EMERGENCY': 1,
            'CM_CHF': 1,  # Congestive Heart Failure
            'CM_HTN': 1,  # Hypertension
            'CM_DM': 1    # Diabetes
        }
        
        # Predict readmission risk
        risk = predict_readmission_risk(best_model, example_patient, feature_names)
        print(f"Example patient's 30-day readmission risk: {risk:.2%}")
        
    except Exception as e:
        print(f"Error during model training: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nAnalysis complete!")
    
    # Return the created datasets
    return feature_dataset, admissions, patients, diagnoses, procedures, prescriptions, labevents

if __name__ == "__main__":
    result = main()
    if result is not None:
        feature_dataset, admissions, patients, diagnoses, procedures, prescriptions, labevents = result
        print(f"Successfully processed data. Feature dataset has {len(feature_dataset)} rows.")
    else:
        print("Pipeline execution failed. Please check the error messages above.")