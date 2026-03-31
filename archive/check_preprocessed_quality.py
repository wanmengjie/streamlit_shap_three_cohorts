import pandas as pd

def check_preprocessed_quality():
    print("Checking preprocessed_data/CHARLS_final_preprocessed.csv data quality...")
    try:
        df = pd.read_csv('preprocessed_data/CHARLS_final_preprocessed.csv')
    except FileNotFoundError:
        print("File not found.")
        return

    cols_to_check = ['bmi', 'mwaist', 'systo', 'diasto']
    
    for col in cols_to_check:
        if col in df.columns:
            print(f"\n--- {col} ---")
            print(df[col].describe())
            n_zeros = (df[col] == 0).sum()
            print(f"Zeros: {n_zeros}")
            
            # Check if it looks like LabelEncoded (integers)
            if pd.api.types.is_integer_dtype(df[col]):
                print("Type: Integer (Suspicious if continuous)")
            else:
                print(f"Type: {df[col].dtype}")

if __name__ == "__main__":
    check_preprocessed_quality()
