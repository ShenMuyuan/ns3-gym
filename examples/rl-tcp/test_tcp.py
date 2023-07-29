#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from ns3gym import ns3env
from agents import TcpRlAgent
import torch
import numpy as np

__author__ = "Piotr Gawlowicz"
__copyright__ = "Copyright (c) 2018, Technische Universität Berlin"
__version__ = "0.1.0"
__email__ = "gawlowicz@tkn.tu-berlin.de"

# parser = argparse.ArgumentParser(description='Start simulation script on/off')
# parser.add_argument('--start',
#                     type=int,
#                     default=1,
#                     help='Start ns-3 simulation script 0/1, Default: 1')
# parser.add_argument('--iterations',
#                     type=int,
#                     default=1,
#                     help='Number of iterations, Default: 1')
# args = parser.parse_args()
# startSim = bool(args.start)
# iterationNum = int(args.iterations)

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int,
                    help='set seed for reproducibility')
args = parser.parse_args()
my_seed = 42
if args.seed:
    my_seed = args.seed
print("Using random seed {} for reproducibility.".format(my_seed))
np.random.seed(my_seed)
torch.manual_seed(my_seed)

startSim = False
iterationNum = 1

port = 5555
simTime = 10  # seconds
stepTime = 0.5  # seconds
seed = 12
simArgs = {"--duration": simTime, }
debug = False

env = ns3env.Ns3Env(port=port, stepTime=stepTime, startSim=startSim, simSeed=seed, simArgs=simArgs, debug=debug)
# simpler:
# env = ns3env.Ns3Env()
env.reset()

ob_space = env.observation_space
ac_space = env.action_space
# print("Observation space: ", ob_space, ob_space.dtype)
# print("Action space: ", ac_space, ac_space.dtype)

stepIdx = 0
currIt = 0


def get_agent(obs):
    socketUuid = obs[0]
    tcpEnvType = obs[1]
    tcpAgent = get_agent.tcpAgents.get(socketUuid, None)
    if tcpAgent is None:
        if tcpEnvType == 0:
            # event-based = 0
            tcpAgent = TcpRlAgent()
        else:
            # time-based = 1
            tcpAgent = TcpRlAgent()
        # tcpAgent.set_spaces(get_agent.ob_space, get_agent.ac_space)
        get_agent.tcpAgents[socketUuid] = tcpAgent

    return tcpAgent


# initialize variable
get_agent.tcpAgents = {}
# get_agent.ob_space = ob_space
# get_agent.ac_space = ac_space

np.random.seed(10)
torch.manual_seed(10)

try:
    while True:
        # print("Start iteration: ", currIt)
        obs = env.reset()
        info = {}
        reward = 0
        done = False
        # print("Step: ", stepIdx)
        # print("---obs: ", obs)

        # get existing agent of create new TCP agent if needed
        tcpAgent = get_agent(obs)

        while True:
            stepIdx += 1
            action = tcpAgent.get_action(obs, reward, done, info)
            # print("---action: ", action)

            # print("Step: ", stepIdx)
            obs, reward, done, info = env.step(action)
            # print("---obs, reward, done, info: ", obs, reward, done, info)

            # get existing agent of create new TCP agent if needed
            tcpAgent = get_agent(obs)

            if done:
                stepIdx = 0
                if currIt + 1 < iterationNum:
                    env.reset()
                break

        currIt += 1
        if currIt == iterationNum:
            break

except KeyboardInterrupt:
    print("Ctrl-C -> Exit")
finally:
    env.close()
    print("Done")
