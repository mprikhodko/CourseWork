import numpy as np
from random import randint, choice

# The number of piles
PILES_N = 3  # data structures below are set to 3

# max number of items per pile
ITEMS_MX = 10

Qtable = np.zeros((0,0,0,0))

Alpha, Gamma, Reward = 1.0, 0.8, 100.0


# action is tuple (pile,move)
# a Nim game move - take move-many stones from pile
def index2action(_index:int)->(int,int):
    pile, move = _index//ITEMS_MX, _index%ITEMS_MX + 1
    return pile, move


def action2index(_action:(int,int))->int:
    index = _action[0]*ITEMS_MX + _action[1]-1
    return index


# Random Nim player
def nagent_random(_st:list)->(int,int):
    pile = choice([i for i in range(PILES_N) if _st[i]>0])  # find the non-empty piles
    return pile, randint(1, _st[pile])  # random action


# Based on X-or ing the item counts in piles - mathematical solution
def nagent_guru(_st:list)->(int,int):
    xored = _st[0] ^ _st[1] ^ _st[2]
    if xored == 0:
        return nagent_random(_st)
    for pile in range(PILES_N):
        s = _st[pile] ^ xored
        if s <= _st[pile]:
            return pile, _st[pile]-s  # best action


def nagent_q(_st:list)->(int,int):
    global Qtable
    # pick the best rewarding action, equation 1
    ix_a = np.argmax(Qtable[_st[0], _st[1], _st[2]])  # exploitation
    pile, move = index2action(ix_a)
    # check if qtable has generated a random but game illegal move - we have not explored there yet
    if move <= 0 or _st[pile] < move:
        pile, move = nagent_random(_st)  # exploration
    return pile, move  # action


# Initialize a starting position
def game_init()->list:
    return [randint(1,ITEMS_MX), randint(1,ITEMS_MX), randint(1,ITEMS_MX)]


def game(_a:str, _b:str, _engines):
    state, side = game_init(), 'A'
    while True:
        engine = _engines[_a] if side == 'A' else _engines[_b]
        pile, move = engine(state)
        # print(state, move, pile)  # debug purposes
        state[pile] -= move  # game move
        if state == [0, 0, 0]:  # game ends
            return side  # winning side
        side = 'B' if side == 'A' else 'A'  # switch sides


def play_games(_n:int, _a:str, _b:str, _engines, info:bool=True)->(int,int):
    from collections import defaultdict
    wins = defaultdict(int)
    for _ in range(_n):
        wins[game(_a, _b, _engines)] += 1
    if info:
        print(f"{_n} games, {_a:>8s}{wins['A']:5d}  {_b:>8s}{wins['B']:5d}")
    return wins['A'], wins['B']


# learn from _n games, randomly played to explore the possible states
def train_qlearner(_n:int):
    global Qtable

    def qtable_update(r:float, _st1:list, _action:(int, int), q_future_best:float):
        global Qtable
        ix_a = action2index(_action)
        Qtable[_st1[0], _st1[1], _st1[2], ix_a] = Alpha * (r + Gamma*q_future_best)

    # based on max items per pile
    Qtable = np.zeros((ITEMS_MX+1, ITEMS_MX+1, ITEMS_MX+1, PILES_N*ITEMS_MX), dtype=np.float32)
    # play _n games
    for _ in range(_n):
        # first state is the starting position
        st1 = game_init()
        while True:  # while game not finished
            # make a random move - exploration
            pile, move = nagent_random(st1)
            st2 = list(st1)
            # make the move
            st2[pile] -= move  # --> last move I made
            if st2 == [0, 0, 0]:  # game ends
                qtable_update(Reward, st1, (pile, move), 0)  # I won
                break  # new game

            qtable_update(0, st1, (pile, move), np.max(Qtable[st2[0], st2[1], st2[2]]))

            # Switch sides for play and learning
            st1 = st2


# Function to print the entire set of states
def qtable_log(_fn:str):
    global Qtable
    with open(_fn, 'w') as fout:
        s = f'state'
        for ix_a in range(PILES_N*ITEMS_MX):
            pile, move = index2action(ix_a)
            s += f',{move:02d}_{pile:01d}'
        print(s, file=fout)
        for i, j, k in [(i,j,k) for i in range(ITEMS_MX+1) for j in range(ITEMS_MX+1) for k in range(ITEMS_MX+1)]:
            s = f'{i:02d}_{j:02d}_{k:02d}'
            for ix_a in range(PILES_N*ITEMS_MX):
                r = Qtable[i, j, k, ix_a]
                s += f',{r:.1f}'
            print(s, file=fout)


if __name__ == '__main__':
    engines = {'Random': nagent_random, 'Guru': nagent_guru, 'Qlearner': nagent_q}
    train_qlearner(1000)
    # Play games
    play_games(100, 'Guru', 'Random', engines)
    play_games(100, 'Guru', 'Guru', engines)
    play_games(100, 'Qlearner', 'Random', engines)
    play_games(100, 'Qlearner', 'Guru', engines)

    qtable_log('qtable_debug.txt')
