__author__ = 'Qiao Jin'

from utils import *

# Monte-Carlo Control Learning

player1 = Player()
player2 = Player()

num_iter = 1000000

for i in range(num_iter):
	board = Board()

	p1 = [[]]
	p2 = [[]]

	while True:
		if board.cont:
			p1[0].append(player1.play(board,eps=1/num_iter))
		else:
			p1.append(board.win)
			p2.append(-board.win)
			break

		if board.cont:
			p2[0].append(player2.play(board,eps=0.1/num_iter))
		else:
			p1.append(board.win)
			p2.append(-board.win)
			break

	player1.learn(p1,lr=0.1)
	player2.learn(p2,lr=0)