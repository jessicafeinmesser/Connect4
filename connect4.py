import copy
import numpy as np
import random
from termcolor import colored  # can be taken out if you don't like it...

# # # # # # # # # # # # # # global values  # # # # # # # # # # # # # #
ROW_COUNT = 6
COLUMN_COUNT = 7

RED_CHAR = colored('X', 'red')  # RED_CHAR = 'X'
BLUE_CHAR = colored('O', 'blue')  # BLUE_CHAR = 'O'

EMPTY = 0
RED_INT = 1
BLUE_INT = 2

VIC = 10**20  # The value of a winning board (for max)
LOSS = -VIC  # The value of a losing board (for max)
TIE = 0  # The value of a tie
SIZE = 4  # The length of a winning sequence
COMPUTER = SIZE + 1  # Marks the computer's cells on the board
HUMAN = 1  # Marks the human's cells on the board


# # # # # # # # # # # # # # functions definitions # # # # # # # # # # # # # #

def create_board():
    """create empty board for new game"""
    board = np.zeros((ROW_COUNT, COLUMN_COUNT), dtype=int)
    return board


def drop_chip(board, row, col, chip):
    """place a chip (red or BLUE) in a certain position in board"""
    board[row][col] = chip


def is_valid_location(board, col):
    """check if a given column in the board has a room for extra dropped chip"""
    return board[ROW_COUNT - 1][col] == 0


def get_next_open_row(board, col):
    """assuming column is available to drop the chip,
    the function returns the lowest empty row  """
    for r in range(ROW_COUNT):
        if board[r][col] == 0:
            return r


def print_board(board):
    """print current board with all chips put in so far"""
    # print(np.flip(board, 0))
    print(" 1 2 3 4 5 6 7 \n" "|" + np.array2string(np.flip(np.flip(board, 1)))
          .replace("[", "").replace("]", "").replace(" ", "|").replace("0", "_")
          .replace("1", RED_CHAR).replace("2", BLUE_CHAR).replace("\n", "|\n") + "|")


def game_is_won(board, chip):
    """check if current board contain a sequence of 4-in-a-row of in the board
     for the player that play with "chip"  """

    winning_Sequence = np.array([chip, chip, chip, chip])
    # Check horizontal sequences
    for r in range(ROW_COUNT):
        if "".join(list(map(str, winning_Sequence))) in "".join(list(map(str, board[r, :]))):
            return True
    # Check vertical sequences
    for c in range(COLUMN_COUNT):
        if "".join(list(map(str, winning_Sequence))) in "".join(list(map(str, board[:, c]))):
            return True
    # Check positively sloped diagonals
    for offset in range(-2, 4):
        if "".join(list(map(str, winning_Sequence))) in "".join(list(map(str, board.diagonal(offset)))):
            return True
    # Check negatively sloped diagonals
    for offset in range(-2, 4):
        if "".join(list(map(str, winning_Sequence))) in "".join(list(map(str, np.flip(board, 1).diagonal(offset)))):
            return True


def get_valid_locations(board):
    valid_locations = []
    for col in range(COLUMN_COUNT):
        if is_valid_location(board, col):
            valid_locations.append(col)
    return valid_locations


def moveRandom(board, color):
    valid_locations = get_valid_locations(board)
    column = random.choice(
        valid_locations)  # you can replace with input if you like... -- line updated with Gilad's code-- thanks!
    row = get_next_open_row(board, column)
    drop_chip(board, row, column, color)


def evaluateBoard(s):
    dr = [-SIZE + 1, -SIZE + 1, 0, SIZE - 1]
    dc = [0, SIZE - 1, SIZE - 1, SIZE - 1]
    heuristic_value = 0
    board = s[0]
    rows, cols = board.shape

    for row in range(rows):
        for col in range(cols):
            for i in range(len(dr)):
                r2 = row + dr[i]
                c2 = col + dc[i]
                if 0 <= r2 < rows and 0 <= c2 < cols:
                    sum = 0
                    valid_sequence = True
                    for k in range(SIZE):
                        nr = row + k * dr[i]
                        nc = col + k * dc[i]
                        if 0 <= nr < rows and 0 <= nc < cols:
                            sum += s[0][nr][nc]
                        else:
                            valid_sequence = False
                            break
                    if valid_sequence:
                        if sum == COMPUTER * SIZE:
                            return VIC
                        if sum == HUMAN * SIZE:
                            heuristic_value -= 10**15  # Large negative value for blocking
                        heuristic_value += sum

    # Add more weight to the center column
    center_col = cols // 2
    center_array = [int(i == center_col) for i in range(cols)]
    center_score = np.sum(board * center_array) * 3
    heuristic_value += center_score

    return heuristic_value


def alpha_beta_pruning(s, depth, alpha, beta, maximizingPlayer):
    valid_locations = get_valid_locations(s[0])
    is_terminal = game_is_won(s[0], RED_INT) or game_is_won(s[0], BLUE_INT) or len(valid_locations) == 0

    if depth == 0 or is_terminal:
        if is_terminal:
            if game_is_won(s[0], RED_INT):
                return None, VIC
            elif game_is_won(s[0], BLUE_INT):
                return None, LOSS
            else:  # Game is over, no more valid moves
                return None, TIE
        else:  # Depth is zero
            return None, evaluateBoard(s)

    if maximizingPlayer:
        value = -float('inf')
        column = random.choice(valid_locations)
        for col in valid_locations:
            row = get_next_open_row(s[0], col)
            b_copy = copy.deepcopy(s[0])
            drop_chip(b_copy, row, col, RED_INT)
            new_s = [b_copy, s[1], BLUE_INT, ROW_COUNT * COLUMN_COUNT - np.count_nonzero(b_copy)]
            new_score = alpha_beta_pruning(new_s, depth - 1, alpha, beta, False)[1]
            if new_score > value:
                value = new_score
                column = col
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return column, value
    else:  # Minimizing player
        value = float('inf')
        column = random.choice(valid_locations)
        for col in valid_locations:
            row = get_next_open_row(s[0], col)
            b_copy = copy.deepcopy(s[0])
            drop_chip(b_copy, row, col, BLUE_INT)
            new_s = [b_copy, s[1], RED_INT, ROW_COUNT * COLUMN_COUNT - np.count_nonzero(b_copy)]
            new_score = alpha_beta_pruning(new_s, depth - 1, alpha, beta, True)[1]
            if new_score < value:
                value = new_score
                column = col
            beta = min(beta, value)
            if alpha >= beta:
                break
        return column, value


def makeAlphaBetaMove(s, playerChip, depth):
    col, minimax_score = alpha_beta_pruning(s, depth, -float('inf'), float('inf'), True)
    if col is not None:
        row = get_next_open_row(s[0], col)
        drop_chip(s[0], row, col, playerChip)
        s[3] -= 1
        s[2] = HUMAN if playerChip == COMPUTER else COMPUTER  # Switch turns
        s[1] = evaluateBoard(s)
        if s[3] == 0 and s[1] not in [LOSS, VIC]:
            s[1] = TIE

# # # # # # # # # # # # # # main execution of the game # # # # # # # # # # # # # #
blueWins = 0
redWins = 0
ties = 0
for counter in range(100):
    turn = 0

    board = create_board()
    print_board(board)
    game_over = False

    s = [board, 0, COMPUTER, ROW_COUNT * COLUMN_COUNT]  # [board, heuristic_value, current_player, empty_cells_count]

    while not game_over:
        if turn % 2 == 0:
            makeAlphaBetaMove(s, RED_INT, 5)  # RED bot with depth 5 Alpha-Beta pruning

        if turn % 2 == 1 and not game_over:
            moveRandom(board, BLUE_INT)  # random bot for BLUE

        print_board(board)

        if game_is_won(board, RED_INT):
            game_over = True
            print(colored("Red wins!", 'red'))
            redWins += 1
        if game_is_won(board, BLUE_INT):
            game_over = True
            print(colored("Blue wins!", 'blue'))
            blueWins += 1
        if len(get_valid_locations(board)) == 0:
            game_over = True
            print(colored("Draw!", 'blue'))
            ties += 1

        s = [board, s[1], COMPUTER if turn % 2 == 0 else HUMAN, ROW_COUNT * COLUMN_COUNT - np.count_nonzero(board)]
        turn += 1
print("Red wins: " + str(redWins))
print("Blue wins: " + str(blueWins))
print("Ties: " + str(ties))