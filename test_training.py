import unittest
import main
import numpy as np

class TestNumerics(unittest.TestCase):
    def test_reinforce_correct(self):
        records = [
            {
                'state':[0,0,0,0],
                'new_state':[0,0,0,0],
                'reward': 0,
                'action': 1,
                'done': False,
            },
        ]*5 + [
            {
                'state':[0,0,0,0],
                'new_state':[0,0,0,1],
                'reward': 1,
                'action': 0,
                'done': False,
            },
        ]* 5
        class env:
            class observation_space:
                shape = (4,)
            class action_space:
                n = 2
        model = main.build_model(env)
        main.train_model( model, records, env, batch_size=64)
        prediction = main.predict(model,[0,0,0,0])
        assert np.argmax(prediction) == 0, prediction