#------------------------------------------------------------------------------
#
#   DQN Demo
#
#   Author : Igor Janos
#
#------------------------------------------------------------------------------

import os
import gym
import yaml
import numpy as np
import torch
import random
import math

from itertools import count
from torchsummary import summary

from backstage.replayMemory import ReplayMemory, Transition
from backstage.model import DQN
from backstage.utils import EnvState, preprocessFrame, EpsilonGreedyPolicy

import wandb



#------------------------------------------------------------------------------
#
#   Trainer class
#
#------------------------------------------------------------------------------

class Trainer:

    def __init__(self, config):

        self.config = config
        self.batchSize = config["batchSize"]
        self.shape = config["shape"]

        # Torch
        self.device = "cuda"

        # Environment
        self.env = gym.make(config["envName"], render_mode='rgb_array')
        self.frameSkip = config["frameSkip"]

        # Drzime poslednych N frejmov
        self.envState = EnvState(nFrames=config["nFrames"], shape=self.shape)

        # Zistime rozmery v ktorych sa to hra
        h,w = self.shape
        self.nActions = self.env.action_space.n        
        self.netQ = DQN(h, w, self.envState.nFrames, self.nActions)

        if (os.path.isfile(config["modelOutput"])):
            checkpoint = torch.load(config["modelOutput"], map_location=torch.device('cpu'))
            self.netQ.load_state_dict(checkpoint)

        self.netQ = self.netQ.to(self.device)
        self.netQ.train()
        self.netV = DQN(h, w, self.envState.nFrames, self.nActions)
        self.netV = self.netV.to(self.device)
        self.netV.train()

        summary(self.netQ, input_size=(self.envState.nFrames,h,w))

        self._updateNetworks()

        self.memory = ReplayMemory(config["memorySize"])
        self.policy = EpsilonGreedyPolicy(
                            self.nActions, self.netQ, 
                            epsilon=config["epsEnd"], epsilonUpper=config["epsStart"],
                            decay=config["epsDecay"]
                        )


        # Optimizer
        self.optimizer = torch.optim.RMSprop(
                            self.netQ.parameters(),
                            lr = config["learningRate"]
                            )
        self.criterion = torch.nn.SmoothL1Loss().to(self.device)
        self.steps = 0




    def train(self):

        avgScore = []
        print("Training for {} episodes: ", self.config["episodes"])
        for iEpisode in range(self.config["episodes"]):

            # Reset the environment
            observation = self.env.reset()
            lastObservation = observation
            self.envState.reset()
            state = self.envState.push(observation, lastObservation)
            score = 0

            # Pocitame kroky v epizodach
            for t in count():

                state = state.to(self.device)
                self.netQ.eval()
                action = self.policy.selectAction(state)
                self.netQ.train()
                action = action.cpu().detach()

                # Frame skipping!
                reward = 0.0
                for _ in range(self.frameSkip):
                    lastObservation = observation
                    observation, r, done, _ = self.env.step(action.item())
                    reward += r
                    if (done): break
                    
                # Reward processing normujeme na male cislo
                if (reward > 0): reward = 1
                if (reward > 100): reward = 10       # Bonus za mothership!

                # Mensi penalty za nic nerobenie
                reward = reward - 0.02
                
                score += reward
                reward = torch.tensor([reward])

                # Vytiahneme novy video frejm a odlozime do pamate
                if (done):
                    nextState = None
                else:
                    nextState = self.envState.push(observation, lastObservation)

                state = state.cpu().detach()
                self.memory.push(state, action, nextState, reward)
                state = nextState

                # Ak mame co, ucime
                self._optimizeModel()

                # Koncime epizodu ?
                if (done):
                    avgScore.append(score)
                    print("Episode Num:{}  Steps: {}  Score: {}".format(iEpisode, t+1, score))

                    if (self.config["wandb"]):
                        wandb.log(
                            {
                                "epizode": iEpisode,
                                "score": score,
                                "epsilon": self.policy.currentEpsilon,
                                "memory": len(self.memory)
                            }
                        )

                    break

            # Raz za cas updatneme siet na odhad value
            if (iEpisode % self.config["targetUpdate"] == 0):
                self._updateNetworks()
                print("---------------")
                print("Mean score: ", np.mean(avgScore))
                print("Total optimizer steps: {}".format(self.steps))
                avgScore = []
                self._makeSnapshot(intermediate=False)
                print("---------------")



    #--------------------------------------------------------------------------
    #   Private methods
    #--------------------------------------------------------------------------
        

    def _optimizeModel(self):

        # Potrebujeme mat aspon BATCH_SIZE vzoriek v pamati
        if (len(self.memory) < self.batchSize):
            return

        # Vytiahneme data
        transitions = self.memory.sample(self.batchSize)
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        nonFinalMask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool)
        nonFinalNextStates = torch.cat([s for s in batch.next_state if s is not None])
        stateBatch = torch.cat(batch.state)
        actionBatch = torch.cat(batch.action)
        rewardBatch = torch.cat(batch.reward)

        nonFinalMask = nonFinalMask.to(self.device)
        nonFinalNextStates = nonFinalNextStates.to(self.device)
        stateBatch = stateBatch.to(self.device)
        actionBatch = actionBatch.to(self.device)
        rewardBatch = rewardBatch.to(self.device)

        # Ucime !
        stateActionValues = self.netQ(stateBatch).gather(1, actionBatch)
        nextStateValues = torch.zeros(self.batchSize).to(self.device)
        nextStateValues[nonFinalMask] = self.netV(nonFinalNextStates).max(1)[0].detach()

        # expected Q values
        expectedStateActionValues = (nextStateValues * self.config["gamma"]) + rewardBatch
        expectedStateActionValues = expectedStateActionValues.unsqueeze(1)

        # Huber loss
        loss = self.criterion(stateActionValues, expectedStateActionValues)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # inkrementujeme pocitadlo
        self.steps += 1
        if (self.steps % self.config["snapshotInterval"] == 0):
            self._makeSnapshot(intermediate=True)

        
    def _updateNetworks(self):
        self.netV.load_state_dict(self.netQ.state_dict())


    def _makeSnapshot(self, intermediate=True):

        if (intermediate):
            modelOutput = os.path.join("models", "snapshot-{}.pkl".format(self.steps))
        else:
            modelOutput = self.config["modelOutput"]

        print("Saving model: ", modelOutput)
        torch.save(self.netQ.state_dict(), modelOutput)



#------------------------------------------------------------------------------
#   Training Stage
#------------------------------------------------------------------------------

def stage_train(configFile):


    # load config file
    with open(configFile) as cf:
        config = yaml.safe_load(cf)

    if (config["wandb"]):
        wandb.init(project="invaders", entity="igorjanos")
        wandb.config = config

    trainer = Trainer(config)
    trainer.train()



if (__name__ == "__main__"):
    stage_train("params.yaml")

