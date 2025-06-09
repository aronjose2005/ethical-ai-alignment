import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import numpy as np
from src.bci_processing import process_bci_data, map_to_ethics

def test_process_bci_data():
    eeg_data = np.random.randn(100)
    result = process_bci_data(eeg_data)
    assert result.shape == eeg_data.shape
    assert np.isclose(np.mean(result), 0, atol=1e-5)

def test_map_to_ethics():
    data = np.ones(100)
    result = map_to_ethics(data)
    assert isinstance(result, float)
