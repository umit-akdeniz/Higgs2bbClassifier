from src.model import build_residual_model

def test_model_build():
    model = build_residual_model(input_dim=27, output_dim=2)
    assert model.output_shape == (None, 2)
