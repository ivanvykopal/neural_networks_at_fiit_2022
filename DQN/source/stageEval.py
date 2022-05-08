#------------------------------------------------------------------------------
#
#   DQN Demo
#
#   Author : Igor Janos
#
#------------------------------------------------------------------------------

import gym
import yaml

import numpy as np
import torch
import cv2

from itertools import count
from backstage.model import DQN
from backstage.utils import EnvState, preprocessFrame, visualize, GreedyPolicy, EpsilonGreedyPolicy, RandomPolicy


#------------------------------------------------------------------------------
#
#   Evaluator class
#
#------------------------------------------------------------------------------

class Evaluator:

    def __init__(self, config):

        self.config = config
        self.shape = config["shape"]

        # Torch
        self.device = "cpu"

        # Environment
        self.env = gym.make(config["envName"], render_mode='human')

        # Drzime poslednych N frejmov
        self.envState = EnvState(nFrames=4, shape=self.shape)

        # Zistime rozmery v ktorych sa to hra
        h,w = self.shape
        self.nActions = self.env.action_space.n        

        # Loadneme z modeloveho suboru
        self.netQ = DQN(h, w, self.envState.nFrames, self.nActions)
        checkpoint = torch.load(config["modelOutput"], map_location=torch.device('cpu'))
        self.netQ.load_state_dict(checkpoint)
        self.netQ = self.netQ.to(self.device)
        self.netQ.eval()

        # Nasa policy
        self.policy = EpsilonGreedyPolicy(self.nActions, self.netQ, epsilon=0.02)
        #self.policy = GreedyPolicy(self.nActions, self.netQ)
        #self.policy = RandomPolicy(self.nActions)


    def playGame(self):

        for iEpisode in count():

            # Zaciatok epizody
            observation = self.env.reset()
            self.envState.reset()
            state = self.envState.push(observation, observation)
            score = 0

            lastObservation = observation
            lastImg = None
        
            # Pocitame kroky v epizodach
            for t in count():

                state = state.to(self.device)
                action = self.policy.selectAction(state)
                action = action.cpu().detach()

                observation, reward, done, _ = self.env.step(action.item())
                score += reward
                reward = torch.tensor([reward])

                # Vytiahneme novy video frejm a odlozime do pamate
                state = self.envState.push(observation, lastObservation)
                lastObservation = observation

                if (True):
                    vis = visualize(state)
                    cv2.imshow("frame", vis)
                    o = cv2.cvtColor(observation, cv2.COLOR_RGB2BGR)
                    cv2.imshow("observation", o)


                # Koncime epizodu ?
                if (done):
                    print("Episode: {} - {} steps - SCORE: {}".format(iEpisode, t+1, score))
                    break




#------------------------------------------------------------------------------
#   Evaluation Stage
#------------------------------------------------------------------------------

def stage_eval(configFile):

    # load config file
    with open(configFile) as cf:
        config = yaml.safe_load(cf)

    eval = Evaluator(config)
    eval.playGame()



if (__name__ == "__main__"):
    stage_eval("params.yaml")
