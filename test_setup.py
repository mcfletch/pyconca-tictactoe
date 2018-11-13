from mock import Mock
import main
import math
import numpy as np

def test_model_build_output_dimention():
    input_dim = (3,)
    output_dim = 2
    mock_env = Mock(observation_space=Mock(shape=input_dim),action_space=Mock(n=output_dim))
    model = main.build_model(mock_env)
    assert model.output_shape == (None,output_dim)

def test_model_build_input_dimention():
    input_dim = (5,)
    output_dim = 7
    mock_env = Mock(observation_space=Mock(shape=input_dim),action_space=Mock(n=output_dim))
    model = main.build_model(mock_env)
    assert model.input_shape == (None, input_dim[0])

def test_history_generates_batch_of_size():
    history_size = 100
    batch_size = 24
    history = [[1,2,3]] * history_size
    assert len(history) == history_size
    batch_generator = main.generate_batches(history, batch_size)
    i = 0
    remaining_history = history_size
    for batch in batch_generator:
        i += 1
        assert len(batch) == min(batch_size, remaining_history)
        remaining_history -= len(batch)
    assert i == math.ceil(history_size / batch_size)

