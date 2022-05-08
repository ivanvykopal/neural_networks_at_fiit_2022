#------------------------------------------------------------------------------
#
#   DQN Demo
#
#   Author : Igor Janos
#
#------------------------------------------------------------------------------

import numpy as np
import torch
import random
import math

from skimage.transform import resize



#------------------------------------------------------------------------------
#
#   RandomPolicy class
#
#------------------------------------------------------------------------------

class RandomPolicy:
    def __init__(self, nActions):
        self.nActions = nActions

    def selectAction(self, state):
        return torch.tensor([[random.randrange(self.nActions)]], dtype=torch.long)

    def reset(self):
        pass


#------------------------------------------------------------------------------
#
#   GreedyPolicy class
#
#------------------------------------------------------------------------------

class GreedyPolicy:

    def __init__(self, nActions, netQ):
        self.nActions = nActions
        self.netQ = netQ

    def selectAction(self, state):
        with torch.no_grad():
            actionDistribution = self.netQ(state)
            act = actionDistribution.max(1)     # Highest value
            act = act[1].view(1, 1)             # Index of highest value
            return act
        
    def reset(self):
        pass


#------------------------------------------------------------------------------
#
#   EpsilonGreedyPolicy class
#
#------------------------------------------------------------------------------

class EpsilonGreedyPolicy:

    def __init__(self, nActions, netQ, epsilon=0.1, epsilonUpper=0.1, decay=0.0):
        self.nActions = nActions
        self.netQ = netQ
        self.epsilon = epsilon
        self.epsilonUpper = epsilonUpper
        self.decay = decay
        self.steps = 0
        self.currentEpsilon = epsilon

    def selectAction(self, state):

        # Casom klesa ?
        if (self.decay > 0.0):
            epsilon = self.epsilon + (self.epsilonUpper - self.epsilon)*math.exp(-1.0 * self.steps / self.decay)
        else:
            epsilon = self.epsilon

        self.steps += 1
        self.currentEpsilon = epsilon

        sample = random.random()
        if (sample > epsilon):
            with torch.no_grad():
                actionDistribution = self.netQ(state)
                act = actionDistribution.max(1)     # Highest value
                act = act[1].view(1, 1)             # Index of highest value
            return act
        else:
            return torch.tensor([[random.randrange(self.nActions)]], dtype=torch.long)




#------------------------------------------------------------------------------
#   
#   EnvState class
#
#------------------------------------------------------------------------------

class EnvState:
    def __init__(self, nFrames, shape):
        self.nFrames = nFrames
        self.shape = shape
        self.frames = []

    def reset(self):
        self.frames = []

    def push(self, frame, lastFrame):

        # vypadne (h,w)
        frame = preprocessFrame(frame, shape=self.shape)
        lastFrame = preprocessFrame(lastFrame, shape=self.shape)
        frame = np.maximum(frame, lastFrame)

        # Vyhodime prvy
        if (len(self.frames) > 0): self.frames.pop(0)

        # Dokladame na koniec - uchovavame ako uint8 v pamati    
        while (len(self.frames) < self.nFrames):
            self.frames.append(torch.from_numpy(frame).unsqueeze(0))

        # Vratime von - 4 posledne frejmy
        frames = [
            self.frames[3],
            self._difFrame(self.frames[3], self.frames[2]),
            self._difFrame(self.frames[2], self.frames[1]),
            self._difFrame(self.frames[1], self.frames[0])
        ]

        result = torch.cat(frames, dim=0)     
        result = result.unsqueeze(0)   
        result = result.to(torch.uint8)
        return result


    def _difFrame(self, f1, f2):
        return (128 + 0.5*(f1.float()-f2.float())).to(torch.uint8)


def preprocessFrame(frame, shape=(84,84)):

    '''
        Frame - (h, w, ch)
        Skonvertujeme do GrayScale, a potom orezeme na 
        192 x 160 rozmer
    '''

    h, w, ch = frame.shape

    # Odrezeme nadbytocnosti
    nLines = 192
    nFirst = (h - nLines) // 2
    frame = frame[nFirst:(nFirst+nLines),:,:]

    # Skonvertujeme na ciernobiely obrazok <0;1>
    frame = np.max(frame, axis=2)
    frame = frame.astype(np.float32) / 255.0
    frame = resize(frame, shape, anti_aliasing=False)
    frame = ((frame > 0) * 255.0).astype(np.uint8)

    return frame


def visualize(state):

    # Skonvertujeme stav do podoby vhodnej na vizualizaciu
    bs, ch, h, w = state.shape
    
    frames = [ state[0,0], state[0,1], state[0,2], state[0,3] ]
    singleFrame = torch.cat(frames, dim=1)
    return singleFrame.cpu().detach().numpy()

