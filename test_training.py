import unittest
import main
import numpy as np

class env:
    class observation_space:
        shape = (4,)
    class action_space:
        n = 2

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
        model = main.build_model(env)
        main.train_model( model, records, env, batch_size=64)
        prediction = main.predict(model,[0,0,0,0])
        assert np.argmax(prediction) == 0, prediction
    
    def test_predict_future_reward(self):
        """When predicting future rewards, we want to see the network give correct directions"""
        good_sequence = [
            ([0,0,0,0],1,[0,0,0,1]),
            ([0,0,0,1],0,[1,0,1,0]),
            ([1,0,1,0],1,[1,1,1,1]),
        ]
        bad_sequence = [
            ([0,0,0,0],0,[1,0,0,1]),
            ([1,0,0,1],1,[0,0,1,0]),
            ([0,0,1,0],1,[0,1,1,1]),
        ]
        def expand(r, final_reward):
            results = []
            for i,(state,action,new_state) in enumerate(r):
                record = {
                    'state': np.array(state,'f'),
                    'new_state': np.array(new_state,'f'),
                    'action': action,
                    'done': i >= len(r),
                    'reward': final_reward
                }
                results.append(record)
            assert results[-1]['reward'] == final_reward
            return results 
        records = expand(good_sequence,1.0) + expand(bad_sequence,-1.0)
        print(records)
        records = records * 256
        model = main.build_model(env)
        main.train_model( model, records, env, batch_size=8)
        for (state,action,new_state) in good_sequence:
            prediction = main.predict(model,state)
            assert np.argmax(prediction) == action, (state,action,prediction)
        
        for (state,action,new_state) in bad_sequence:
            prediction = main.predict(model,state)
            assert np.argmax(prediction) != action, (state,action,prediction)
        