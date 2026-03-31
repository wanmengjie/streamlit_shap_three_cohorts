import pandas as pd
import os

# Define output directory
OUTPUT_DIR = "FINAL_PAPER_TABLES"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_full_table2():
    print(">>> 正在生成包含完整指标的 Table 2...")
    
    cohorts = [
        ('A', 'Cohort_A_Healthy_Prospective'), 
        ('B', 'Cohort_B_Depression_to_Comorbidity'), 
        ('C', 'Cohort_C_Cognition_to_Comorbidity')
    ]
    
    all_metrics = []
    
    for cohort_label, folder in cohorts:
        path = os.path.join(folder, '01_prediction', 'model_performance_full_is_comorbidity_next.csv')
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                # The first row is the champion model
                best_row = df.iloc[0]
                
                # Extract relevant columns
                # We want the formatted strings (e.g., "0.75 (0.70-0.80)") not the _raw values
                metrics = {
                    'Cohort': cohort_label,
                    'Model': best_row['Model'],
                    'AUC': best_row.get('AUC', 'N/A'),
                    'AUPRC': best_row.get('AUPRC', 'N/A'),
                    'Accuracy': best_row.get('Accuracy', 'N/A'),
                    'F1': best_row.get('F1', 'N/A'),
                    'Precision': best_row.get('Precision', 'N/A'),
                    'Recall': best_row.get('Recall', 'N/A'),
                    'Youden': best_row.get('Youden', 'N/A'),
                    'Brier': best_row.get('Brier', 'N/A')
                }
                all_metrics.append(metrics)
            except Exception as e:
                print(f"Error reading {path}: {e}")
        else:
            print(f"File not found: {path}")

    if all_metrics:
        df_out = pd.DataFrame(all_metrics)
        output_path = os.path.join(OUTPUT_DIR, 'Table_2_Prediction_Performance_Full.csv')
        df_out.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"Table 2 (Full) generated at: {output_path}")
        print(df_out.to_string())
    else:
        print("No data found to generate table.")

if __name__ == "__main__":
    generate_full_table2()
