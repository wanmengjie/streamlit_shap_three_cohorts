# -*- coding: utf-8 -*-
"""检查 PAPER_附录全部绝对路径.md 中列出的路径是否存在"""
import os

base = r'C:\Users\lenovo\Desktop\因果机器学习'
paths = [
    ('Table S1', 'results/tables/tableS1_variable_definitions.csv'),
    ('Table S2', 'results/tables/table1_missing_summary.csv'),
    ('Table S3', 'LIU_JUE_STRATEGIC_SUMMARY/sensitivity_summary.csv'),
    ('Table S4', 'results/tables/table4_ate_summary.csv'),
    ('Table S5 A', 'results/tables/table6_external_validation_axisA.csv'),
    ('Table S5 B', 'results/tables/table6_external_validation_axisB.csv'),
    ('Table S5 C', 'results/tables/table6_external_validation_axisC.csv'),
    ('Table S6', 'results/tables/table7_psm_psw_dml.csv'),
    ('Table S4b X-Learner+PSM+PSW', 'results/tables/table4_xlearner_psm_psw_wide.csv'),
    ('Table S7 A', 'Cohort_A_Healthy_Prospective/07_sensitivity/sensitivity_exercise/bias_sensitivity.csv'),
    ('Table S7 B', 'Cohort_B_Depression_to_Comorbidity/07_sensitivity/sensitivity_exercise/bias_sensitivity.csv'),
    ('Table S7 C', 'Cohort_C_Cognition_to_Comorbidity/07_sensitivity/sensitivity_exercise/bias_sensitivity.csv'),
    ('Table S8', 'results/tables/tableS8_hyperparameter_search.csv'),
    ('Table S9 A', 'Cohort_A_Healthy_Prospective/03_causal/assumption_checks_summary.txt'),
    ('Table S9 B', 'Cohort_B_Depression_to_Comorbidity/03_causal/assumption_checks_summary.txt'),
    ('Table S9 C', 'Cohort_C_Cognition_to_Comorbidity/03_causal/assumption_checks_summary.txt'),
    ('Table S10', 'results/tables/table_physio_ate_summary.csv'),
    ('Table S11', 'results/tables/imputation_sensitivity_results.csv'),
    ('Figure S2 A', 'Cohort_A_Healthy_Prospective/03_causal/fig_propensity_overlap_exercise.png'),
    ('Figure S2 B', 'Cohort_B_Depression_to_Comorbidity/03_causal/fig_propensity_overlap_exercise.png'),
    ('Figure S2 C', 'Cohort_C_Cognition_to_Comorbidity/03_causal/fig_propensity_overlap_exercise.png'),
    ('Figure S3', 'LIU_JUE_STRATEGIC_SUMMARY/07_sensitivity_imputation/figS3_imputation_sensitivity.png'),
    ('Figure S4 A', 'Cohort_A_Healthy_Prospective/04b_external_validation/external_validation_calibration.png'),
    ('Figure S4 B', 'Cohort_B_Depression_to_Comorbidity/04b_external_validation/external_validation_calibration.png'),
    ('Figure S4 C', 'Cohort_C_Cognition_to_Comorbidity/04b_external_validation/external_validation_calibration.png'),
    ('Figure S5 A', 'Cohort_A_Healthy_Prospective/02_shap_stratified/fig_shap_stratified.png'),
    ('Figure S5 B', 'Cohort_B_Depression_to_Comorbidity/02_shap_stratified/fig_shap_stratified.png'),
    ('Figure S5 C', 'Cohort_C_Cognition_to_Comorbidity/02_shap_stratified/fig_shap_stratified.png'),
    ('Figure S6 A', 'Cohort_A_Healthy_Prospective/02_shap/fig_shap_interaction.png'),
    ('Figure S6 B', 'Cohort_B_Depression_to_Comorbidity/02_shap/fig_shap_interaction.png'),
    ('Figure S6 C', 'Cohort_C_Cognition_to_Comorbidity/02_shap/fig_shap_interaction.png'),
    ('Figure S7 A', 'Cohort_A_Healthy_Prospective/04c_dose_response/fig_dose_response_rcs.png'),
    ('Figure S7 B', 'Cohort_B_Depression_to_Comorbidity/04c_dose_response/fig_dose_response_rcs.png'),
    ('Figure S7 C', 'Cohort_C_Cognition_to_Comorbidity/04c_dose_response/fig_dose_response_rcs.png'),
    ('Figure S8 A', 'Cohort_A_Healthy_Prospective/03_causal/ite_validation/fig_ite_stratified.png'),
    ('Figure S8 B', 'Cohort_B_Depression_to_Comorbidity/03_causal/ite_validation/fig_ite_stratified.png'),
    ('Figure S8 C', 'Cohort_C_Cognition_to_Comorbidity/03_causal/ite_validation/fig_ite_stratified.png'),
    ('Figure S9 A', 'Cohort_A_Healthy_Prospective/03_causal/nomogram/fig_nomogram_ite.png'),
    ('Figure S9 B', 'Cohort_B_Depression_to_Comorbidity/03_causal/nomogram/fig_nomogram_ite.png'),
    ('Figure S9 C', 'Cohort_C_Cognition_to_Comorbidity/03_causal/nomogram/fig_nomogram_ite.png'),
    ('Figure S10', 'LIU_JUE_STRATEGIC_SUMMARY/sensitivity_ate_comparison_exercise.png'),
    ('Figure S11 A', 'Cohort_A_Healthy_Prospective/03_causal/fig_causal_methods_forest_exercise.png'),
    ('Figure S11 B', 'Cohort_B_Depression_to_Comorbidity/03_causal/fig_causal_methods_forest_exercise.png'),
    ('Figure S11 C', 'Cohort_C_Cognition_to_Comorbidity/03_causal/fig_causal_methods_forest_exercise.png'),
    ('Figure S12 A', 'Cohort_A_Healthy_Prospective/06_subgroup/fig_subgroup_academic_forest.png'),
    ('Figure S12 B', 'Cohort_B_Depression_to_Comorbidity/06_subgroup/fig_subgroup_academic_forest.png'),
    ('Figure S12 C', 'Cohort_C_Cognition_to_Comorbidity/06_subgroup/fig_subgroup_academic_forest.png'),
    ('Figure S13 A', 'Cohort_A_Healthy_Prospective/07_sensitivity/sensitivity_exercise/fig_s14_placebo_e_value.png'),
    ('Figure S13 B', 'Cohort_B_Depression_to_Comorbidity/07_sensitivity/sensitivity_exercise/fig_s14_placebo_e_value.png'),
    ('Figure S13 C', 'Cohort_C_Cognition_to_Comorbidity/07_sensitivity/sensitivity_exercise/fig_s14_placebo_e_value.png'),
    ('Figure S14 A', 'Cohort_A_Healthy_Prospective/07_sensitivity/sensitivity_exercise/fig_bias_sensitivity.png'),
    ('Figure S14 B', 'Cohort_B_Depression_to_Comorbidity/07_sensitivity/sensitivity_exercise/fig_bias_sensitivity.png'),
    ('Figure S14 C', 'Cohort_C_Cognition_to_Comorbidity/07_sensitivity/sensitivity_exercise/fig_bias_sensitivity.png'),
    ('causal exercise A', 'LIU_JUE_STRATEGIC_SUMMARY/all_interventions_causal/Cohort_A/exercise/fig_causal_methods_forest_exercise.png'),
    ('causal exercise B', 'LIU_JUE_STRATEGIC_SUMMARY/all_interventions_causal/Cohort_B/exercise/fig_causal_methods_forest_exercise.png'),
    ('causal exercise C', 'LIU_JUE_STRATEGIC_SUMMARY/all_interventions_causal/Cohort_C/exercise/fig_causal_methods_forest_exercise.png'),
    ('causal drinkev A', 'LIU_JUE_STRATEGIC_SUMMARY/all_interventions_causal/Cohort_A/drinkev/fig_causal_methods_forest_drinkev.png'),
    ('causal drinkev B', 'LIU_JUE_STRATEGIC_SUMMARY/all_interventions_causal/Cohort_B/drinkev/fig_causal_methods_forest_drinkev.png'),
    ('causal drinkev C', 'LIU_JUE_STRATEGIC_SUMMARY/all_interventions_causal/Cohort_C/drinkev/fig_causal_methods_forest_drinkev.png'),
    ('causal bmi_normal A', 'LIU_JUE_STRATEGIC_SUMMARY/all_interventions_causal/Cohort_A/bmi_normal/fig_causal_methods_forest_bmi_normal.png'),
    ('causal bmi_normal B', 'LIU_JUE_STRATEGIC_SUMMARY/all_interventions_causal/Cohort_B/bmi_normal/fig_causal_methods_forest_bmi_normal.png'),
    ('causal bmi_normal C', 'LIU_JUE_STRATEGIC_SUMMARY/all_interventions_causal/Cohort_C/bmi_normal/fig_causal_methods_forest_bmi_normal.png'),
    ('causal chronic_low A', 'LIU_JUE_STRATEGIC_SUMMARY/all_interventions_causal/Cohort_A/chronic_low/fig_causal_methods_forest_chronic_low.png'),
    ('causal chronic_low B', 'LIU_JUE_STRATEGIC_SUMMARY/all_interventions_causal/Cohort_B/chronic_low/fig_causal_methods_forest_chronic_low.png'),
    ('causal chronic_low C', 'LIU_JUE_STRATEGIC_SUMMARY/all_interventions_causal/Cohort_C/chronic_low/fig_causal_methods_forest_chronic_low.png'),
    ('causal is_socially_isolated A', 'LIU_JUE_STRATEGIC_SUMMARY/all_interventions_causal/Cohort_A/is_socially_isolated/fig_causal_methods_forest_is_socially_isolated.png'),
    ('causal is_socially_isolated B', 'LIU_JUE_STRATEGIC_SUMMARY/all_interventions_causal/Cohort_B/is_socially_isolated/fig_causal_methods_forest_is_socially_isolated.png'),
    ('causal is_socially_isolated C', 'LIU_JUE_STRATEGIC_SUMMARY/all_interventions_causal/Cohort_C/is_socially_isolated/fig_causal_methods_forest_is_socially_isolated.png'),
    ('X-Learner 全干预森林图', 'LIU_JUE_STRATEGIC_SUMMARY/xlearner_all_interventions/fig_xlearner_all_interventions_forest.png'),
]

missing = []
for name, p in paths:
    full = os.path.join(base, p.replace('/', os.sep))
    if os.path.exists(full):
        print('OK', name, p)
    else:
        print('MISS', name, p)
        missing.append((name, p))

print('\n--- Missing:', len(missing), '---')
for name, p in missing:
    print(name, ':', p)
