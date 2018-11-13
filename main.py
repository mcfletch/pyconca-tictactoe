#! /usr/bin/env python
import gym
import numpy as np
import bisect
from keras.models import Model
from keras.layers import (
    Dense, 
    Input,
    Dropout,
    Activation,
)

def build_model( env ):
    initial = layer = Input(env.observation_space.shape)
    for size in [63,63]:
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


def run_game( env, model, epoch=0 ):
    done = False 
    state = env.reset()
    history = []

    explore = np.max(( 
        np.min( (
            .99,
            np.cos(epoch*.02*np.pi)
        )),
        0.05
    ))
    overall_reward = 0
    choices = []
    while not done:
        if epoch and not epoch%10:
            env.render()
        if np.random.random() < explore:
            action = env.action_space.sample()
            random_trial = True
        else:
            state = np.array(state,'f').reshape((1,-1))
            action_weights = model.predict(state)
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
    history = apply_decay(history)
    # print('%s/%s chose 0'%(choices.count(0), len(choices)))
    return history

def apply_decay(history, decay = 0.01):
    if history:
        final_reward = history[-1]['overall_reward']
        result = []
        for neg_index,record in enumerate(history[::-1][1:]):
            record = record.copy()
            record['reward'] = record.get('reward', 0) + ((1.0-(decay * neg_index)) * final_reward)
            result.append(record)
        return result[::-1]
    return history 

def generate_batches(epoch_history, batch_size):
    np.random.shuffle(epoch_history)
    while epoch_history:
        yield epoch_history[:batch_size]
        epoch_history = epoch_history[batch_size:]

def train_model( model, epoch_history, env, batch_size=1024):
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
        model.fit(
            states, 
            actions,
        )

def insort_left(target, record):
    """Insort left *only* comparing record[0] values"""
    for i in range(len(target)):
        if target[i][0] > record[0]:
            target.insert(i,record)
            return 
    target.append(record)
    return

def main():
    env = gym.make('CartPole-v1')
    model = build_model( env )
    scores = []
    for epoch in range(200):
        overall_history = []
        epoch_scores = []
        while len(overall_history) < (1024*10):
            history = run_game( env, model, epoch )
            score = history[-1]['overall_reward']
            epoch_scores.append(score)
            if not scores or score > scores[0][0]:
                print(f'Score: {score: 3.1f}')
                # Numpy arrays are pedantic about allowing comparisons with bisect, sigh
                insort_left(scores,(score,history))
                del scores[:-100]
                overall_history.extend( history )
            elif score >= 100:
                print('O',end='')
            else:
                print('.',end='')
        print('Epoch Score: ',np.mean(epoch_scores))
        for _,history in scores:
            overall_history.extend(history)
        train_model( model, overall_history, env )




if __name__ == "__main__":
    main()
    