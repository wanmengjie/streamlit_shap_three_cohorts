import pandas as pd
import os

# Define output directory
OUTPUT_DIR = "FINAL_PAPER_TABLES"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_table_s4_full():
    print(">>> Generating Table S4 (95% CI)...")
    
    # Load the summary table from results
    src_path = r'results/tables/table4_ate_summary.csv'
    if not os.path.exists(src_path):
        print(f"Error: Source file not found: {src_path}")
        return

    df = pd.read_csv(src_path)
    
    # Filter for reliable interventions only (reliable=1)
    df = df[df['reliable'] == 1].copy()
    
    # Format 95% CI (lower, upper), 4 decimal places
    df['95% CI'] = df.apply(lambda row: f"{row['ate_lb']:.4f}, {row['ate_ub']:.4f}" if pd.notna(row['ate_lb']) and pd.notna(row['ate_ub']) else "N/A", axis=1)
    
    # Map axis names
    cohort_map = {
        'Cohort_A': 'Cohort A (Healthy)',
        'Cohort_B': 'Cohort B (Depression)',
        'Cohort_C': 'Cohort C (Cognition)'
    }
    df['Cohort'] = df['axis'].map(cohort_map)
    
    # Map intervention names
    # exercise, sleep_adequate, smokev, drinkev, bmi_normal
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
    final_df = df[['Cohort', 'Intervention', 'ate', '95% CI']].rename(columns={'ate': 'ATE'})
    
    # Sort
    final_df = final_df.sort_values(['Cohort', 'Intervention'])
    
    # Save
    output_path = os.path.join(OUTPUT_DIR, 'Table_S4_Exploratory_Causal_Analysis_Full.csv')
    final_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"Table S4 (Full) generated at: {output_path}")
    print(final_df.head().to_string())

if __name__ == "__main__":
    generate_table_s4_full()
