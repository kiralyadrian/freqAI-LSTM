import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch", reason="PyTorch is not installed")

MODEL_PATH = Path(__file__).resolve().parents[1] / "torch" / "PyTorchLSTMModel.py"
spec = importlib.util.spec_from_file_location("pylstm_module", MODEL_PATH)
pylstm_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pylstm_module)
PyTorchLSTMModel = pylstm_module.PyTorchLSTMModel


def test_pytorch_lstm_model_output_shape():
    input_dim = 4
    output_dim = 2
    hidden_dim = 8
    batch_size = 3
    sequence_length = 5

    model = PyTorchLSTMModel(input_dim=input_dim, output_dim=output_dim, hidden_dim=hidden_dim)
    dummy_input = torch.randn(batch_size, sequence_length, input_dim)
    out = model(dummy_input)

    assert out.shape == (batch_size, output_dim)
