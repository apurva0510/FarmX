import pandas as pd
from functions import (
    create_category_columns,
    train_and_save_n_category,
    train_and_save_K_category,
    train_and_save_P_category,
    train_and_save_N,
)

def main():
    # Load your dataset
    try:
        df = pd.read_csv('data/crop_recommendation.csv')  # Update the path as necessary
    except FileNotFoundError:
        print("Error: The specified file was not found.")
        return
    except pd.errors.EmptyDataError:
        print("Error: The file is empty.")
        return
    except pd.errors.ParserError:
        print("Error: The file could not be parsed.")
        return
    
    # Print DataFrame columns and head for debugging
    print("DataFrame Columns:", df.columns.tolist())
    print("DataFrame Head:")
    print(df.head())

    # Create category columns
    df = create_category_columns(df)

    # Check for NaN values in important columns
    if df[['N_category', 'K_category', 'P_category']].isnull().values.any():
        print("Error: There are NaN values in the category columns.")
        return
    
    # Check if required columns exist
    required_columns = ['humidity', 'temperature', 'rainfall', 'ph', 'id', 'N', 'K', 'P']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Error: The following required columns are missing: {missing_columns}")
        return

    # Proceed to train models
    try:
        # Call each training function to train and save models
        train_and_save_n_category(df)
        train_and_save_K_category(df)
        train_and_save_P_category(df)
        train_and_save_N(df)
        print("All models trained and saved successfully.")
        
    except ValueError as ve:
        print("ValueError during model training:", ve)
    except Exception as e:
        print("An unexpected error occurred during model training:", e)

if __name__ == "__main__":
    main()