import pandas as pd
import numpy as np

def check_data_quality():
    print("Checking CHARLS.csv data quality...")
    try:
        df = pd.read_csv('CHARLS.csv', encoding='gb18030')
    except Exception:
        df = pd.read_csv('CHARLS.csv', encoding='utf-8')
    
    cols_to_check = ['bmi', 'mwaist', 'systo', 'diasto']
    
    for col in cols_to_check:
        if col in df.columns:
            print(f"\n--- {col} ---")
            print(df[col].describe())
            n_zeros = (df[col] == 0).sum()
            n_nan = df[col].isna().sum()
            n_neg = (df[col] < 0).sum()
            print(f"Zeros: {n_zeros}")
            print(f"NaNs: {n_nan}")
            print(f"Negatives: {n_neg}")
            
            # Check low values
            if col == 'bmi':
                print(f"Values < 15: {(df[col] < 15).sum()}")
            if col == 'systo':
                print(f"Values < 80: {(df[col] < 80).sum()}")

if __name__ == "__main__":
    check_data_quality()
