import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="urllib3")


from bci_processing import process_bci_data, map_to_ethics
from xai_explanation import explain_with_shap, explain_with_lime
from ethical_alignment import align_with_ethics
import numpy as np

def main():
    # Simulate EEG data
    eeg_data = np.random.randn(100)
    print(f"Raw EEG Data: {eeg_data[:5]}")

    # Process BCI data
    normalized_data = process_bci_data(eeg_data)
    ethical_score = map_to_ethics(normalized_data)
    print(f"Ethical Score: {ethical_score}")

    # Simulate AI decision
    ai_decision = "Prioritize profit over fairness"
    adjusted_decision = align_with_ethics(ethical_score, ai_decision)
    print(f"Adjusted AI Decision: {adjusted_decision}")

    # XAI explanations (placeholder model and data)
    model_func = lambda x: np.mean(x, axis=1)  # Dummy model
    data = np.random.randn(10, 5)
    shap_values = explain_with_shap(model_func, data)
    lime_explanation = explain_with_lime(model_func, data, feature_names=["f1", "f2", "f3", "f4", "f5"])
    print(f"SHAP Values: {shap_values}")
    print(f"LIME Explanation: {lime_explanation}")

if __name__ == "__main__":
    main()
