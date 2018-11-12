#! /usr/bin/env python
import gym
import numpy as np

from keras.models import Model
from keras.layers import (
    Dense, 
    Input,
    Dropout,
    Activation,
)

def build_model( env ):
    initial = layer = Input(env.observation_space.shape)
    for size in [31,15,15,15]:
        layer = Dense(size)(layer)
        layer = Activation('relu')(layer)
        layer = Dropout(.3)(layer)
    layer = Dense(env.action_space.n)(layer)
    layer = Activation('softmax')(layer)
    model = Model(initial,layer)
    model.compile(
        'adam',
        'mse'
    )
    return model


def run_game( env, model ):
    done = False 
    state = env.reset()
    history = []
    while not done:
        env.render()
        if np.random.random() < 0.1:
            action = env.action_space.sample()
            random_trial = True
        else:
            state = np.array(state,'f').reshape((1,-1))
            action = np.argmax( model.predict(state) )
            random_trial = False
        new_state,reward,done,_ = env.step(action)
        history.append({
            'state': state,
            'new_state': new_state,
            'action': action,
            'random_trial': random_trial,
            'reward': reward,
            'done': done,
        })
    history = apply_decay(history)
    return history

def apply_decay(history, decay = 0.01):
    if history:
        final_reward = history[-1]['reward']
        result = [
            record.copy().update({
                'reward': record.get('reward', 0) + (decay * neg_index * final_reward)
            })
            for neg_index,record in enumerate(history[::-1])
        ]
        return result 
    return history 

def main():
    env = gym.make('CartPole-v1')
    model = build_model( env )
    for epoch in range(200):
        overall_history = []
        for i in range(200):
            history = run_game( env, model )
            overall_history.append( history )
        #train_model( model, history )



if __name__ == "__main__":
    main()
    