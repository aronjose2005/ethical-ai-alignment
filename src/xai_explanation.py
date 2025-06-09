import shap
import numpy as np
from lime.lime_tabular import LimeTabularExplainer

# Explain AI model decisions using SHAP
def explain_with_shap(model_func, data):
    explainer = shap.KernelExplainer(model_func, data)
    shap_values = explainer.shap_values(data)
    return shap_values

# Explain using LIME
def explain_with_lime(model_func, data, feature_names):
    explainer = LimeTabularExplainer(data, feature_names=feature_names, mode="regression")
    explanation = explainer.explain_instance(data[0], model_func, num_features=len(feature_names))
    return explanation.as_list()
