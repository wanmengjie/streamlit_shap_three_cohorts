import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_study_flow(output_path='FINAL_PAPER_FIGURES/Figure_1_Study_Flow.png'):
    fig, ax = plt.subplots(figsize=(10, 12))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # Define box style
    box_props = dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='black', linewidth=1.5)
    
    # Coordinates
    x_center = 0.5
    y_start = 0.95
    y_step = 0.12
    
    # 1. Raw Data
    ax.text(x_center, y_start, 
            "CHARLS National Survey (Waves 1-4)\nRaw Records (N = 96,628)", 
            ha='center', va='center', bbox=box_props, fontsize=11)
    
    # Arrow 1
    ax.annotate('', xy=(x_center, y_start - 0.08), xytext=(x_center, y_start - 0.04),
                arrowprops=dict(arrowstyle='->', lw=1.5))
    ax.text(x_center + 0.02, y_start - 0.06, "Excluded: Age < 60 years (n = 47,613)", ha='left', va='center', fontsize=9)

    # 2. Age >= 60
    y_2 = y_start - y_step
    ax.text(x_center, y_2, 
            "Age ≥ 60 years\n(N = 49,015)", 
            ha='center', va='center', bbox=box_props, fontsize=11)

    # Arrow 2
    ax.annotate('', xy=(x_center, y_2 - 0.08), xytext=(x_center, y_2 - 0.04),
                arrowprops=dict(arrowstyle='->', lw=1.5))
    ax.text(x_center + 0.02, y_2 - 0.06, "Excluded: Missing Depression (CES-D-10) (n = 5,967)", ha='left', va='center', fontsize=9)

    # 3. Depression Non-missing
    y_3 = y_2 - y_step
    ax.text(x_center, y_3, 
            "Valid Depression Assessment\n(N = 43,048)", 
            ha='center', va='center', bbox=box_props, fontsize=11)

    # Arrow 3
    ax.annotate('', xy=(x_center, y_3 - 0.08), xytext=(x_center, y_3 - 0.04),
                arrowprops=dict(arrowstyle='->', lw=1.5))
    ax.text(x_center + 0.02, y_3 - 0.06, "Excluded: Missing Cognition Score (n = 11,474)", ha='left', va='center', fontsize=9)

    # 4. Cognition Non-missing
    y_4 = y_3 - y_step
    ax.text(x_center, y_4, 
            "Valid Cognition Assessment\n(N = 31,574)", 
            ha='center', va='center', bbox=box_props, fontsize=11)

    # Arrow 4
    ax.annotate('', xy=(x_center, y_4 - 0.08), xytext=(x_center, y_4 - 0.04),
                arrowprops=dict(arrowstyle='->', lw=1.5))
    ax.text(x_center + 0.02, y_4 - 0.06, "Excluded: Missing Outcome (Next Wave) or\nPre-existing Comorbidity (n = 14,588)", ha='left', va='center', fontsize=9)

    # 5. Final Incident Cohort
    y_5 = y_4 - y_step
    ax.text(x_center, y_5, 
            "Final Incident Cohort\n(N = 14,386)", 
            ha='center', va='center', bbox=dict(boxstyle='round,pad=0.5', facecolor='#E8F8F5', edgecolor='black', linewidth=2), fontsize=12, fontweight='bold')

    # Branching Arrows
    y_branch_start = y_5 - 0.05
    y_branch_end = y_5 - 0.15
    
    # Left Branch
    ax.annotate('', xy=(0.2, y_branch_end), xytext=(x_center, y_branch_start),
                arrowprops=dict(arrowstyle='->', lw=1.5, connectionstyle="arc3,rad=0.1"))
    
    # Middle Branch
    ax.annotate('', xy=(0.5, y_branch_end), xytext=(x_center, y_branch_start),
                arrowprops=dict(arrowstyle='->', lw=1.5))
    
    # Right Branch
    ax.annotate('', xy=(0.8, y_branch_end), xytext=(x_center, y_branch_start),
                arrowprops=dict(arrowstyle='->', lw=1.5, connectionstyle="arc3,rad=-0.1"))

    # 6. Cohorts
    y_6 = y_branch_end - 0.02
    
    # Cohort A
    ax.text(0.2, y_6, 
            "Cohort A\nHealthy\n(n = 8,828)", 
            ha='center', va='top', bbox=dict(boxstyle='round,pad=0.3', facecolor='#D5F5E3', edgecolor='green'), fontsize=10)
    
    # Cohort B
    ax.text(0.5, y_6, 
            "Cohort B\nDepression Only\n(n = 3,123)", 
            ha='center', va='top', bbox=dict(boxstyle='round,pad=0.3', facecolor='#FADBD8', edgecolor='red'), fontsize=10)
    
    # Cohort C
    ax.text(0.8, y_6, 
            "Cohort C\nCognition Only\n(n = 2,435)", 
            ha='center', va='top', bbox=dict(boxstyle='round,pad=0.3', facecolor='#D6EAF8', edgecolor='blue'), fontsize=10)

    plt.tight_layout()
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Figure 1 saved to {output_path}")

if __name__ == "__main__":
    draw_study_flow()
