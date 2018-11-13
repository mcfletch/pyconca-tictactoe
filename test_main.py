from mock import Mock
import main

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
    assert model.input_shape == (None, input_dim)

def test_history_generates_batch_of_size():
    history_size = 10
    batch_size = 25
    history = [(1,2,3) for _ in range(history_size)]
    assert len(history) == history_size
    batch = main.generate_batches(history, batch_size)
    assert len(batch) == batch_size
    assert len(history) == history_size - batch_size
