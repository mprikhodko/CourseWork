import nimlearn as nim

import torch
import torch.nn as nn
import random
from collections import namedtuple
from collections import deque
import sys

REPLAY_N = 10000
Transition = namedtuple('Transition', ('st', 'action', 'r', 'st_next', 'ended'))

Dqmodel = nn.Sequential(nn.Linear(1,1))
Loss_fn = nn.MSELoss()
Optimizer = torch.optim.Adam(Dqmodel.parameters(), 1e-3)

Eps_greedy, Eps_min, Eps_decay = 1.0, 0.01, 0.99
Eps = Eps_greedy


def replay_fill():
    mem = deque(maxlen=REPLAY_N)
    while len(mem) < REPLAY_N:
        # first state is the starting position
        st1 = nim.game_init()
        while True:  # while game not finished
            # make a random move - exploration
            pile, move = nim.nagent_random(st1)
            st2 = list(st1)
            # make the move
            st2[pile] -= move  # --> last move I made
            if st2 == [0, 0, 0]:  # game ends
                mem.append(
                    Transition(st1, nim.action2index((pile, move)), nim.Reward, st2, True))
                break  # new game
            mem.append(
                Transition(st1, nim.action2index((pile, move)), 0, st2, False))
            # Switch sides for play and learning
            st1 = st2
    return mem


def nagent_dq(_st:list)->(int,int):
    global Dqmodel, Eps
    # if np.random.rand() <= Eps:
    #     return nim.nagent_random(_st)
    with torch.no_grad():
        q_values = Dqmodel(torch.tensor(_st, dtype=torch.float32))
    # check if game illegal move
    pile, move = nim.index2action(torch.argmax(q_values).item())
    if move <= 0 or _st[pile] < move:
        pile, move = nim.nagent_random(_st)
    return pile, move


def dqmodel_init(_lr):
    global Dqmodel, Loss_fn, Optimizer
    Dqmodel = nn.Sequential(
        nn.Linear(nim.PILES_N, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, nim.PILES_N*nim.ITEMS_MX))
    Loss_fn = nn.MSELoss()
    Optimizer = torch.optim.Adam(Dqmodel.parameters(), _lr)


def dqmodel_train(_X):
    def adjust_eps():
        global Eps
        if Eps > Eps_min:
            Eps *= Eps_decay

    global Dqmodel, Loss_fn, Optimizer

    batch_states, batch_targets = [], []
    for transition in _X:
        st, a, r, st_next, ended = transition
        with torch.no_grad():
            if ended:
                target = r
            else:
                pred = Dqmodel(torch.tensor(st_next, dtype=torch.float32))
                target = r + nim.Gamma*pred.max()

        target_all = Dqmodel(torch.tensor(st, dtype=torch.float32))
        target_all[a] = target

        batch_states.append(st)
        batch_targets.append(target_all)
        # adjust_eps()
    Optimizer.zero_grad()
    pred = Dqmodel(torch.tensor(batch_states, dtype=torch.float32))
    loss = Loss_fn(pred, torch.stack(batch_targets))
    loss.backward()
    Optimizer.step()
    return loss.item()


def train_dqlearner(_replay, lr=0.01, epochs=100, bs=200, info=True):
    dqmodel_init(lr)
    loss_ = []
    for e in range(epochs):
        samples = random.sample(_replay, bs)
        loss = dqmodel_train(samples)
        loss_ += [loss]
        if info:
            sys.stderr.write(f"\r{e+1:02d}/{epochs:02d} | Loss: {loss:<6.4f}")
            sys.stderr.flush()
    return loss_


if __name__ == '__main__':
    engines = {'Random': nim.nagent_random, 'Guru': nim.nagent_guru,
               'Qlearner': nim.nagent_q, 'Dqlearner': nagent_dq}
    nim.train_qlearner(1000)
    replay = replay_fill()
    losses = train_dqlearner(replay)
    print()
    nim.play_games(100, 'Dqlearner', 'Random', engines)
    nim.play_games(100, 'Qlearner', 'Dqlearner', engines)
    nim.play_games(100, 'Dqlearner', 'Qlearner', engines)
    nim.play_games(100, 'Dqlearner', 'Guru', engines)
