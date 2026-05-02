import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE


def preprocess(data_path="data/Telco-Customer-Churn.csv"):

    df = pd.read_csv(data_path)

    # Fix TotalCharges
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.fillna({'TotalCharges': 0}, inplace=True)

    # Encode target
    df['Churn'] = df['Churn'].replace({'Yes': 1, 'No': 0}).astype(int)

    # Drop customer ID
    df = df.drop('customerID', axis=1)

    # Clean "No internet service" / "No phone service"
    cols_to_clean = [
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies', 'MultipleLines'
    ]
    for col in cols_to_clean:
        df[col] = df[col].replace({'No internet service': 'No', 'No phone service': 'No'})

    # Binary encoding
    binary_cols = {
        'gender': {'Male': 1, 'Female': 0},
        'Partner': {'Yes': 1, 'No': 0},
        'Dependents': {'Yes': 1, 'No': 0},
        'PhoneService': {'Yes': 1, 'No': 0},
        'PaperlessBilling': {'Yes': 1, 'No': 0},
        'OnlineSecurity': {'Yes': 1, 'No': 0},
        'OnlineBackup': {'Yes': 1, 'No': 0},
        'DeviceProtection': {'Yes': 1, 'No': 0},
        'TechSupport': {'Yes': 1, 'No': 0},
        'StreamingTV': {'Yes': 1, 'No': 0},
        'StreamingMovies': {'Yes': 1, 'No': 0},
        'MultipleLines': {'Yes': 1, 'No': 0},
    }
    for col, mapping in binary_cols.items():
        df[col] = df[col].map(mapping)

    # One-hot encode multi-category columns
    df = pd.get_dummies(df, columns=['InternetService', 'Contract', 'PaymentMethod'], drop_first=True)

    # Split features and target
    X = df.drop('Churn', axis=1)
    y = df['Churn']

    # Train / val / test split (60 / 20 / 20)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val
    )

    # Scale numerical features
    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    scaler = StandardScaler()
    X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])
    X_val[numerical_cols] = scaler.transform(X_val[numerical_cols])

    # Apply SMOTE to training set only
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train.astype(int))

    print("Preprocessing complete.")
    print(f"  Train size : {X_train_resampled.shape}")
    print(f"  Val size   : {X_val.shape}")
    print(f"  Test size  : {X_test.shape}")

    return X_train_resampled, X_val, X_test, y_train_resampled, y_val, y_test, scaler