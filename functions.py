import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def create_category_columns(df):
    # Define percentiles for categorization
    low_threshold_N = df['N'].quantile(0.33)
    high_threshold_N = df['N'].quantile(0.67)

    # Create a new column 'N_category' based on the thresholds
    df['N_category'] = pd.cut(df['N'], bins=[-float('inf'), low_threshold_N, high_threshold_N, float('inf')],
                              labels=['Low', 'Moderate', 'High'], include_lowest=True)

    # Define percentiles for categorization
    low_threshold_K = df['K'].quantile(0.33)
    high_threshold_K = df['K'].quantile(0.67)

    # Create a new column 'K_category' based on the thresholds
    df['K_category'] = pd.cut(df['K'], bins=[-float('inf'), low_threshold_K, high_threshold_K, float('inf')],
                              labels=['Low', 'Moderate', 'High'], include_lowest=True)

    # Define percentiles for categorization
    low_threshold_P = df['P'].quantile(0.33)
    high_threshold_P = df['P'].quantile(0.67)

    # Create a new column 'P_category' based on the thresholds
    df['P_category'] = pd.cut(df['P'], bins=[-float('inf'), low_threshold_P, high_threshold_P, float('inf')],
                              labels=['Low', 'Moderate', 'High'], include_lowest=True)

    return df

def train_and_save_n_category(df):
    # Features and target variable
    X = df[['humidity', 'temperature', 'rainfall', 'ph', 'id']]
    y = df['N_category']

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Create and train the Gradient Boosting Classifier model
    gb_classifier = GradientBoostingClassifier(random_state=42)
    gb_classifier.fit(X_scaled, y)

    # Save the model and scaler
    pickle.dump(gb_classifier, open('models/gb_n_category_classifier.pkl', 'wb'))
    pickle.dump(scaler, open('models/scaler_n_category.pkl', 'wb'))

def predict_n_category(humidity, temperature, rainfall, ph, id):
    # Load the model and scaler
    gb_classifier = pickle.load(open('models/gb_n_category_classifier.pkl', 'rb'))
    scaler = pickle.load(open('models/scaler_n_category.pkl', 'rb'))

    # Scale the input values
    input_scaled = scaler.transform([[humidity, temperature, rainfall, ph, id]])

    # Make predictions
    predicted_n_category = gb_classifier.predict(input_scaled)

    return predicted_n_category[0]

def train_and_save_K_category(df):
    # Features and target variable
    X = df[['humidity', 'temperature', 'rainfall', 'ph', 'id']]
    y = df['K_category']

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Create and train the Random Forest Classifier model
    rf_classifier = RandomForestClassifier(random_state=42)
    rf_classifier.fit(X_train_scaled, y_train)

    # Save the model and scaler
    pickle.dump(rf_classifier, open('models/rf_K_category_classifier.pkl', 'wb'))
    pickle.dump(scaler, open('models/scaler_K_category.pkl', 'wb'))

def predict_K_category(humidity, temperature, rainfall, ph, id):
    # Load the model and scaler
    rf_classifier = pickle.load(open('models/rf_K_category_classifier.pkl', 'rb'))
    scaler = pickle.load(open('models/scaler_K_category.pkl', 'rb'))

    # Scale the input values
    input_scaled = scaler.transform([[humidity, temperature, rainfall, ph, id]])

    # Make predictions
    predicted_K_category = rf_classifier.predict(input_scaled)

    return predicted_K_category[0]

def train_and_save_P_category(df):
    # Features and target variable
    X = df[['humidity', 'temperature', 'rainfall', 'ph', 'id']]
    y = df['P_category']

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Create and train the Random Forest Classifier model
    rf_classifier = RandomForestClassifier(random_state=42)
    rf_classifier.fit(X_train_scaled, y_train)

    # Save the model and scaler
    pickle.dump(rf_classifier, open('models/rf_P_category_classifier.pkl', 'wb'))
    pickle.dump(scaler, open('models/scaler_P_category.pkl', 'wb'))

def predict_P_category(humidity, temperature, rainfall, ph, id):
    # Load the model and scaler
    rf_classifier = pickle.load(open('models/rf_P_category_classifier.pkl', 'rb'))
    scaler = pickle.load(open('models/scaler_P_category.pkl', 'rb'))

    # Scale the input values
    input_scaled = scaler.transform([[humidity, temperature, rainfall, ph, id]])

    # Make predictions
    predicted_P_category = rf_classifier.predict(input_scaled)

    return predicted_P_category[0]

def train_and_save_N(df):
    # Features and target variable
    X = df[['humidity', 'temperature', 'rainfall', 'ph', 'id']]
    y = df['N']

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Create and train the Random Forest Regressor model
    rf_model1 = RandomForestRegressor(random_state=42)
    rf_model1.fit(X_train_scaled, y_train)

    # Save the model and scaler
    pickle.dump(rf_model1, open('models/rf_N_regressor.pkl', 'wb'))
    pickle.dump(scaler, open('models/scaler_N.pkl', 'wb'))

def predict_N(humidity, temperature, rainfall, ph, pid):
    # Load the model and scaler
    rf_model1 = pickle.load(open('models/rf_N_regressor.pkl', 'rb'))
    scaler = pickle.load(open('models/scaler_N.pkl', 'rb'))

    # Scale the input values
    input_scaled = scaler.transform([[humidity, temperature, rainfall, ph, pid]])

    # Make predictions
    predicted_N = rf_model1.predict(input_scaled)

    return predicted_N[0]