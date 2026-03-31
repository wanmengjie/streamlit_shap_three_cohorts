import pandas as pd
import numpy as np

def check_data_values():
    print("Checking variable values in CHARLS.csv...")
    try:
        # Try reading a subset of columns to save memory/time
        cols_to_check = [
            'edu', 'gender', 'rural', 'marry', 'smokev', 'drinkev', 
            'adlab_c', 'iadl', 'wspeed', 'is_socially_isolated', 
            'chronic_burden', 'hibpe', 'diabe', 'cancre', 'lunge', 
            'hearte', 'stroke', 'psyche', 'arthre'
        ]
        
        # Read first 1000 rows to infer types, or full if possible
        # Since file is large, let's read full but only specific columns if possible.
        # But we don't know if all cols exist. Let's read header first.
        
        header = pd.read_csv('CHARLS.csv', nrows=0, encoding='gb18030')
        existing_cols = [c for c in cols_to_check if c in header.columns]
        
        print(f"Columns found: {existing_cols}")
        
        df = pd.read_csv('CHARLS.csv', usecols=existing_cols, encoding='gb18030')
        
        for col in existing_cols:
            print(f"\n--- {col} ---")
            print(f"Type: {df[col].dtype}")
            unique_vals = sorted(df[col].dropna().unique())
            if len(unique_vals) < 20:
                print(f"Unique values: {unique_vals}")
            else:
                print(f"Unique values (first 10): {unique_vals[:10]} ...")
                print(f"Range: [{df[col].min()}, {df[col].max()}]")
                
            # Check specific coding questions
            if col == 'gender':
                print("Gender counts:")
                print(df[col].value_counts().sort_index())
            if col == 'rural':
                print("Rural counts:")
                print(df[col].value_counts().sort_index())
            if col == 'marry':
                print("Marital status counts:")
                print(df[col].value_counts().sort_index())
            if col == 'edu':
                print("Education counts:")
                print(df[col].value_counts().sort_index())

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_data_values()
