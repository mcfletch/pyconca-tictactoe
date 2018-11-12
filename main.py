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
    while not done:
        env.render()
        if np.random.random() < 0.1:
            action = env.action_space.sample()
        else:
            state = np.array(state,'f').reshape((1,-1))
            action = np.argmax( model.predict(state) )
        state,reward,done,_ = env.step(action)


def main():
    env = gym.make('CartPole-v1')
    model = build_model( env )
    for i in range(700):
        run_game( env, model )


if __name__ == "__main__":
    main()
    