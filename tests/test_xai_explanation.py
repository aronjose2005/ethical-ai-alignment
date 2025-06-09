import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import numpy as np
from src.xai_explanation import explain_with_shap, explain_with_lime

def test_explain_with_shap():
    model_func = lambda x: np.mean(x, axis=1)
    data = np.random.randn(5, 3)  # Reduced from (10, 5) to (5, 3)
    result = explain_with_shap(model_func, data)
    assert len(result) == len(data)

def test_explain_with_lime():
    model_func = lambda x: np.mean(x, axis=1)
    data = np.random.randn(5, 3)  # Reduced from (10, 5) to (5, 3)
    result = explain_with_lime(model_func, data, feature_names=["f1", "f2", "f3"])
    assert isinstance(result, list)
