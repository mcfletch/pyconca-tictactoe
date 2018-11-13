#! /usr/bin/env python
import gym
import numpy as np
import bisect
import random
import os
from collections import deque
from keras.models import Model
from keras.layers import (
    Dense, 
    Input,
    Dropout,
    Activation,
)

def predict(model, state):
    """Predict a single state's future reward"""
    state = np.array(state,'f').reshape((1,-1))
    action_weights = model.predict(state)
    return action_weights[0]

def build_model( env ):
    """Build a Q function that predicts reward for a given state"""
    initial = layer = Input(env.observation_space.shape)
    for size in [63,15,]:
        layer = Dense(size)(layer)
        layer = Activation('relu')(layer)
    layer = Dense(env.action_space.n)(layer)
    layer = Activation('linear')(layer)
    model = Model(initial,layer)
    model.compile(
        'adam',
        'mse'
    )
    return model


def run_game( env, model, epoch=0, exploit=.9 ):
    done = False 
    state = env.reset()
    history = []

    overall_reward = 0
    choices = []
    while not done:
        if not epoch % 100:
            env.render()
        if np.random.random() > exploit:
            action = env.action_space.sample()
            random_trial = True
        else:
            state = np.array(state,'f').reshape((1,-1))
            action_weights = predict( model, state)
            action = np.argmax( action_weights )
            random_trial = False
        choices.append(action)
        new_state,reward,done,_ = env.step(action)
        overall_reward += reward
        history.append({
            'state': state,
            'new_state': new_state,
            'action': action,
            'random_trial': random_trial,
            'overall_reward': overall_reward,
            'reward': reward,
            'done': done,
        })
        state = new_state
        # exploit *= max((.995,exploit*1.1))
    # print('%s/%s chose 0'%(choices.count(0), len(choices)))
    return history

def generate_batches(epoch_history, batch_size):
    yield random.sample(epoch_history, min([len(epoch_history),batch_size]))
    # while epoch_history:
    #     yield epoch_history[:batch_size]
    #     epoch_history = epoch_history[batch_size:]

def train_model( model, epoch_history, env, batch_size=1024):
    states = np.zeros((batch_size,)+env.observation_space.shape,'f')
    actions = np.zeros((batch_size,env.action_space.n),'f')
    for batch in generate_batches(epoch_history, batch_size):
        if len(batch) < batch_size:
            break
        for index,record in enumerate(batch):
            states[index] = record['state']
            action_reward = predict(model,record['state'])
            if not record['done']:
                action_reward[record['action']] = record['reward'] + 1.0 * np.max(
                    predict(model,record['new_state'])
                )
            else:
                # assert not np.max(action_reward) > 1.0, action_reward
                action_reward[record['action']] = record['reward']
            actions[index] = action_reward
        model.fit(
            states, 
            actions,
            verbose=0
        )

def insort_left(target, record):
    """Insort left *only* comparing record[0] values"""
    for i in range(len(target)):
        if target[i][0] > record[0]:
            target.insert(i,record)
            return 
    target.append(record)
    return

def verify(env, model):
    history = run_game(env, model, epoch=0, exploit=1.0)
    score = history[-1]['overall_reward']
    return score

def main(env_name='CartPole-v1'):
    env = gym.make(env_name)
    model = build_model( env )
    filename = '%s-weights.hd5'%(env_name)
    if os.path.exists(filename):
        model.load_weights(filename)
    scores = deque(maxlen=100)
    overall_history = []
    epsilon_decay = .02
    epsilon_min = 0.05
    epsilon_max = .995
    epsilon = epsilon_max
    for epoch in range(10000):
        epoch_scores = []
        epsilon = np.max([
            epsilon_min, np.min([
                epsilon, 
                1.0 - np.log10((epoch + 1) * epsilon_decay ),
                epsilon_max,
            ]),
        ])
        exploit = 1.0- epsilon
        # while len(overall_history) < :
        history = run_game( env, model, epoch, exploit )
        score = history[-1]['overall_reward']
        scores.append(score)
        overall_history.extend( history )
        train_model( model, overall_history, env, batch_size=64 )
        if not epoch % 100:
            avg = np.mean(scores)
            print('Avg Score on last 100 tests: ', avg)
            if avg > 195:
                print('Success at epoch %s'%(epoch,))
                model.save_weights(filename)
                print('Verification...')
                verification = [
                    verify(env, model)
                    for i in range(20)
                ]
                print('Verification: mean %s stddev=%s'%(
                    np.mean(verification),
                    np.std(verification),
                ))
                return verification



if __name__ == "__main__":
    main()
    