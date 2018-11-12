#! /usr/bin/env python
import gym
import numpy as np

def run_game( env ):
    env.reset()
    done = False 
    while not done:
        env.render()
        action = env.action_space.sample()
        state,reward,done,_ = env.step(action)


def main():
    env = gym.make('CartPole-v1')
    for i in range(700):
        run_game( env )


if __name__ == "__main__":
    main()
    