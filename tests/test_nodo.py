import pytest
import numpy as np
from nodo_regulator import NODORegulator

def test_fit_transform_output():
    """Garante que o fit/transform retorna o formato esperado."""
    X = np.random.rand(100, 10)
    model = NODORegulator(n_components=2)
    model.fit(X)
    X_transformed = model.transform(X)
    
    assert X_transformed.shape == (100, 2)
    assert not np.isnan(X_transformed).any()

def test_invalid_parameters():
    """Garante que o modelo reclama de parâmetros inválidos."""
    with pytest.raises(ValueError):
        NODORegulator(n_components=-1).fit(np.random.rand(10, 5))

def test_deterministic_output():
    """Garante que com o mesmo input, o resultado é consistente."""
    X = np.random.rand(50, 5)
    model1 = NODORegulator(n_components=2)
    model2 = NODORegulator(n_components=2)
    
    model1.fit(X)
    model2.fit(X)
    
    assert np.allclose(np.abs(model1.embedding_), np.abs(model2.embedding_))
