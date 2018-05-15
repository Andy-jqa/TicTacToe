__author__ = 'Qiao Jin'

from utils import *

# Monte-Carlo Control Learning

player1 = Player()
player2 = Player()

num_iter = 1000000

black_win = 0
draw = 0
white_win = 0

for i in range(num_iter):
	board = Board()

	p1 = [[]]
	p2 = [[]]

	while True:
		if board.cont:
			p1[0].append(player1.play(board,eps=1/(0.01*i+1)))
		else:
			p1.append(board.win)
			p2.append(-board.win)
			break

		if board.cont:
			p2[0].append(player2.play(board,eps=1/(0.01*i+1)))
		else:
			p1.append(board.win)
			p2.append(-board.win)
			break

	if board.win == -1: black_win += 1
	elif board.win == -1: white_win += 1
	else: draw += 1

	player1.learn(p1,lr=0.1)
	player2.learn(p2,lr=0.1)

	print('black win %.2f, white win %.2f, draw %.2f'%(black_win/(i+1), white_win/(i+1), draw/(i+1)))