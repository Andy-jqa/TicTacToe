__author__ = 'Qiao Jin'

import numpy as np
import random as rd
import os.path

class Board:
    def __init__(self, state=np.zeros((3,3))):
        self.state = state
        self.win = 0
        self.draw = 0
        self.cont = 1

    def change(self, coor, player):
        assert self.state[coor[0]][coor[1]] == 0
            
        self.state[coor[0]][coor[1]] = player

    def display(self):
        self.judge()

        if self.draw != 0:
            print('----Draw----')
        elif self.win != 0:
            print('--\'%s\' wins!--'%self.value2pieces(self.win))
        else:
            print('-------------')
        print('%s %s %s\n%s %s %s\n%s %s %s'%tuple(self.value2pieces(v) for v in list(self.state.reshape(9))))

    def value2pieces(self,v):
        if v == 1:
            return('+')
        elif v == -1:
            return('-')
        else:
            return('0')

    def judge(self):
        ones = np.array([1,1,1])
        check1 = np.matmul(self.state,ones)
        check2 = np.matmul(ones,self.state)
        check3 = np.sum(self.state * np.identity(3))
        check4 = np.sum(self.state * np.fliplr(np.identity(3)))
        
        checker = list(check1)+list(check2)+[check3]+[check4]

        assert not (3 in checker and -3 in checker)

        if 3 in checker:
            self.win = 1
            self.cont = 0
        elif -3 in checker:
            self.win = -1
            self.cont = 0
        elif 0 not in self.state:
            self.draw = 1
            self.cont = 0

class GameTree:
    def __init__(self):
        start = np.zeros((9))
        self.state2idx = {tuple(start):0}
        self.idx2child = {}
        self.step(start,1)
        self.idx2state = {v:k for k,v in self.state2idx.items()}
        self.act2value = {}
        for key,value in self.idx2child.items():
            for v in value:
                self.act2value[(key,v)] = 0

    def step(self,state,who):

        idx = self.state2idx[tuple(state)]

        if idx not in self.idx2child:
            self.idx2child[idx] = []

        for i in range(9):
            if state[i] == 0:
                tmp = np.copy(state)
                tmp[i] = who

                if tuple(tmp) not in self.state2idx:
                    self.state2idx[tuple(tmp)] = len(self.state2idx)

                child_idx = self.state2idx[tuple(tmp)]
                if child_idx not in self.idx2child[idx]:
                    self.idx2child[idx] += [child_idx]

                self.step(tmp,-who)


class Player:
    def __init__(self):
        self.tree = GameTree()

    def play(self,board,eps):

        idx = self.tree.state2idx[tuple(board.state.reshape((9)))]
        choices = self.tree.idx2child[idx]
        values = [self.tree.act2value[(idx,_id)] for _id in choices]
        print(values)

        if rd.random() < eps or max(values)==0:
            step = rd.sample(choices,1)[0]
        else:
            step = choices[values.index(max(values))]

        board.state = np.array(self.tree.idx2state[step]).reshape((3,3))
        board.display()

        return idx,step

    def learn(self,result,lr):
        seq = result[0]
        ret = result[1]

        for idx,step in seq:
            self.tree.act2value[(idx,step)] += lr * ((ret * (0.9 ** (len(seq)))) - self.tree.act2value[(idx,step)])