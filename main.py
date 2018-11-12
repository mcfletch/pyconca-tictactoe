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
    for size in [1023,1023,1023,256]:
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

    overall_reward = 0
    while not done:
        env.render()
        if np.random.random() < .99:
            action = env.action_space.sample()
            random_trial = True
        else:
            state = np.array(state,'f').reshape((1,-1))
            action = np.argmax( model.predict(state) )
            random_trial = False
        new_state,reward,done,_ = env.step(action)
        overall_reward += reward
        history.append({
            'state': state,
            'new_state': new_state,
            'action': action,
            'random_trial': random_trial,
            'reward': overall_reward,
            'done': done,
        })
    # history = apply_decay(history)
    return history

def apply_decay(history, decay = 0.01):
    if history:
        final_reward = history[-1]['reward']
        result = []
        for neg_index,record in enumerate(history[::-1][1:]):
            record = record.copy()
            record['reward'] = record.get('reward', 0) + (1.0-(decay * neg_index) * final_reward)
            result.append(record)
        print([record['reward'] for record in result])
        return result 
    return history 

def generate_batches(epoch_history, batch_size):
    np.random.shuffle(epoch_history)
    while epoch_history:
        yield epoch_history[:batch_size]
        epoch_history = epoch_history[batch_size:]

def train_model( model, epoch_history, env, batch_size=256):
    states = np.zeros((batch_size,)+env.observation_space.shape,'f')
    actions = np.zeros((batch_size,env.action_space.n),'f')
    for batch in generate_batches(epoch_history, batch_size):
        if len(batch) < batch_size:
            break
        for index,record in enumerate(batch):
            states[index] = record['state']
            action_reward = np.zeros((env.action_space.n,), 'f')
            action_reward[record['action']] = record['reward']
            actions[index] = action_reward
        model.train_on_batch(
            states, 
            actions,
        )


def main():
    env = gym.make('CartPole-v1')
    model = build_model( env )
    for epoch in range(200):
        overall_history = []
        scores = []
        for i in range(200):
            history = run_game( env, model )
            overall_history.extend( history )
            scores.append(history[-1]['reward'])
        print('Epoch Score: ',np.mean(scores))
        train_model( model, overall_history, env )




if __name__ == "__main__":
    main()
    