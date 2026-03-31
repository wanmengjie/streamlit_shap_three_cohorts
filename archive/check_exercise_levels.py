import pandas as pd

def check_exercise_levels():
    print("Checking 'exercise' variable levels...")
    try:
        # Try reading the preprocessed data first as it's smaller and cleaner
        df = pd.read_csv('preprocessed_data/CHARLS_final_preprocessed.csv')
        if 'exercise' in df.columns:
            unique_vals = sorted(df['exercise'].unique())
            print(f"Unique values in 'exercise' (preprocessed): {unique_vals}")
            print(f"Value counts:\n{df['exercise'].value_counts().sort_index()}")
        else:
            print("'exercise' column not found in preprocessed data.")
            
    except FileNotFoundError:
        print("Preprocessed data not found. Checking raw data...")
        try:
            df = pd.read_csv('CHARLS.csv', encoding='gb18030') # or try utf-8 if this fails
            if 'exercise' in df.columns:
                unique_vals = sorted(df['exercise'].dropna().unique())
                print(f"Unique values in 'exercise' (raw): {unique_vals}")
                print(f"Value counts:\n{df['exercise'].value_counts().sort_index()}")
            else:
                print("'exercise' column not found in raw data.")
        except Exception as e:
            print(f"Error reading raw data: {e}")

if __name__ == "__main__":
    check_exercise_levels()
