import pandas as pd
import os

# Define output directory
OUTPUT_DIR = "FINAL_PAPER_TABLES"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_table_s6_full():
    print(">>> 正在生成包含 95% CI 列的 Table S6...")
    
    # Load the summary table from results
    src_path = r'FINAL_PAPER_TABLES/Table_S6_Causal_Methods_Comparison.csv'
    if not os.path.exists(src_path):
        print(f"Error: Source file not found: {src_path}")
        return

    df = pd.read_csv(src_path)
    
    # Format 95% CI column
    def format_ci(row):
        ate = row['ate']
        lb = row['ate_lb']
        ub = row['ate_ub']
        
        # Check if values are valid
        if pd.isna(lb) or pd.isna(ub):
            return "N/A"
            
        return f"{lb:.4f}, {ub:.4f}"

    df['95% CI'] = df.apply(format_ci, axis=1)
    
    # Map axis names
    cohort_map = {
        'Cohort_A': 'Cohort A (Healthy)',
        'Cohort_B': 'Cohort B (Depression)',
        'Cohort_C': 'Cohort C (Cognition)'
    }
    df['Cohort'] = df['axis'].map(cohort_map).fillna(df['axis'])
    
    # Map intervention names
    intervention_map = {
        'exercise': 'Exercise',
        'sleep_adequate': 'Adequate Sleep',
        'smokev': 'Smoking',
        'drinkev': 'Drinking',
        'bmi_normal': 'Normal BMI',
        'chronic_low': 'Low Chronic Disease'
    }
    df['Intervention'] = df['exposure'].map(intervention_map).fillna(df['exposure'])
    
    # Select final columns
    final_df = df[['Cohort', 'Intervention', 'method', 'ate', '95% CI']].rename(columns={'ate': 'ATE', 'method': 'Method'})
    
    # Sort
    final_df = final_df.sort_values(['Cohort', 'Intervention', 'Method'])
    
    # Save
    output_path = os.path.join(OUTPUT_DIR, 'Table_S6_Causal_Methods_Comparison_Full.csv')
    final_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"Table S6 (Full) generated at: {output_path}")
    print(final_df.head().to_string())

if __name__ == "__main__":
    generate_table_s6_full()
