import pygame
import copy
import numpy as np
from constants import BLAC, WHIT, PADDING_X, PADDING_Y, NUM_COLS, WHITE, NOBODY, BLACK, TIE, WIN_VAL, DIRECTIONS, \
    LIGHT_BLUE, MAROON

pygame.font.init()


class Grid:
    def __init__(self, rows, cols, width, height, window):
        self.rows = rows
        self.cols = cols
        self.width = width
        self.height = height
        self.window = window
        self.selected = None
        self.board = starting_board()

    def draw(self):
        gap = self.width / 8

        for i in range(0, 9):
            thick = 1

            # horizontal lines
            pygame.draw.line(self.window, BLAC, (0, i * gap),
                             (self.width, i * gap), thick)
            # vertical lines
            pygame.draw.line(self.window, BLAC, (i * gap, 0),
                             (i * gap, self.height), thick)

        # Draw Pieces
        for row in range(8):
            for col in range(8):
                if self.board[row][col] == WHITE:
                    pygame.draw.circle(self.window, WHIT, (PADDING_X + (col * 80), PADDING_Y + (row * 80)), 27)
                elif self.board[row][col] == BLACK:
                    pygame.draw.circle(self.window, BLAC, (PADDING_X + (col * 80), PADDING_Y + (row * 80)), 27)

    def click(self, pos):
        row, col = pos[0], pos[1]

        if row < self.width and col < self.height:
            gap = self.width / 8
            y = row // gap
            x = col // gap
            return int(x), int(y)
        else:
            return False


def redraw_window(grid, white_score, black_score, main_button):
    grid.window.fill((207, 185, 151))
    font = pygame.font.SysFont("Arial", 45)

    text_white = font.render("White: " + str(white_score), 1, WHIT)
    text_black = font.render("Black: " + str(black_score), 1, BLAC)
    grid.window.blit(text_white, (0, grid.height))
    grid.window.blit(text_black, (grid.width - text_black.get_width(), grid.height))

    pygame.draw.rect(grid.window, LIGHT_BLUE, main_button)
    fnt = pygame.font.SysFont('Arial', 25)
    main_menu_text = fnt.render('Main Menu', 1, WHIT)
    grid.window.blit(main_menu_text, (grid.width / 2 - main_menu_text.get_width() / 2, grid.height + 10))
    grid.draw()
    pygame.display.update()


def check_game_over(board):
    """Returns the current winner of the board - WHITE, BLACK, TIE, NOBODY"""

    # It's not over if either player still has legal moves
    white_legal_moves = generate_legal_moves(board, True)
    if white_legal_moves:  # Python idiom for checking for empty list
        return NOBODY
    black_legal_moves = generate_legal_moves(board, False)
    if black_legal_moves:
        return NOBODY
    # I guess the game's over
    return find_winner(board)


def starting_board():
    """Returns a board with the traditional starting positions in Othello."""
    board = np.zeros((NUM_COLS, NUM_COLS))
    board[3][3] = WHITE
    board[3][4] = BLACK
    board[4][3] = BLACK
    board[4][4] = WHITE
    return board


def can_capture(board, row, col, white_turn):
    """ Helper that checks capture in each of 8 directions.

    Args:
        board (numpy 2D int array) - othello board
        row (int) - row of move
        col (int) - col of move
        white_turn (bool) - True if it's white's turn
    Returns:
        True if capture is possible in any direction
    """
    for r_delta, c_delta in DIRECTIONS:
        if captures_in_dir(board, row, r_delta, col, c_delta, white_turn):
            return True
    return False


def generate_legal_moves(board, white_turn):
    """Returns a list of (row, col) tuples representing places to move.

    Args:
        board (numpy 2D int array):  The othello board
        white_turn (bool):  True if it's white's turn to play
    """

    legal_moves = []
    for row in range(8):
        for col in range(8):
            if board[row][col] != 0:
                continue  # Occupied, so not legal for a move
            # Legal moves must capture something
            if can_capture(board, row, col, white_turn):
                legal_moves.append((row, col))
    return legal_moves


def play_move(board, move, white_turn):
    """Handles the logic of putting down a new piece and flipping captured pieces.

    The board that is returned is a copy, so this is appropriate to use for search.

    Args:
        board (numpy 2D int array):  The othello board
        move ((int,int)):  A (row, col) pair for the move
        white_turn:  True iff it's white's turn
    Returns:
        board (numpy 2D int array)
    """
    new_board = copy.deepcopy(board)
    new_board[move[0]][move[1]] = WHITE if white_turn else BLACK
    new_board = capture(new_board, move[0], move[1], white_turn)
    return new_board


def minimax_value(board, white_turn, search_depth, alpha, beta):
    """Return the value of the board, up to the maximum search depth.

    Assumes white is MAX and black is MIN (even if black uses this function).

    Args:
        board (numpy 2D int array) - The othello board
        white_turn (bool) - True iff white would get to play next on the given board
        search_depth (int) - the search depth remaining, decremented for recursive calls
        alpha (int or float) - Lower bound on the value:  MAX ancestor forbids lower results
        beta (int or float) - Upper bound on the value:  MIN ancestor forbids larger results
    """

    state = check_game_over(board)
    if state != NOBODY:
        if state == WHITE:
            return WIN_VAL
        elif state == BLACK:
            return -WIN_VAL
        else:
            return 0

    if search_depth == 0:
        return evaluation_function(board)

    moves = generate_legal_moves(board, white_turn)

    if not moves:
        return minimax_value(board, not white_turn, search_depth, alpha, beta)

    if white_turn:
        best_val = float('-inf')

        for move in moves:
            newBoard = play_move(board, move, white_turn)
            best_val = max(best_val, minimax_value(newBoard, False, search_depth - 1, alpha, beta))

            if best_val >= beta:
                return best_val

            alpha = max(best_val, alpha)
    else:
        best_val = float('inf')

        for move in moves:
            newBoard = play_move(board, move, white_turn)
            best_val = min(best_val, minimax_value(newBoard, True, search_depth - 1, alpha, beta))

            if best_val <= alpha:
                return best_val

            beta = min(best_val, beta)

    return best_val


def evaluation_function(board):
    """Returns the difference in piece count - an easy evaluation function for minimax"""

    # We could count with loops, but we're feeling fancy
    return np.count_nonzero(board == WHITE) - np.count_nonzero(board == BLACK)


def find_winner(board):
    """Return identity of winner, assuming game is over.

    Args:
        board (numpy 2D int array):  The othello board, with WHITE/BLACK/NOBODY in spaces

    Returns:
        int constant:  WHITE, BLACK, or TIE.
    """
    # Slick counting of values:  np.count_nonzero counts vals > 0, so pass in
    # board == WHITE to get 1 or 0 in the right spots
    white_count = np.count_nonzero(board == WHITE)
    black_count = np.count_nonzero(board == BLACK)
    if white_count > black_count:
        return WHITE
    if white_count < black_count:
        return BLACK
    return TIE


def captures_in_dir(board, row, row_delta, col, col_delta, white_turn):
    """Returns True iff capture possible in direction described by delta parameters

    Args:
        board (numpy 2D int array) - othello board
        row (int) - row of original move
        row_delta (int) - modification needed to row to move in direction of capture
        col (int) - col of original move
        col_delta (int) - modification needed to col to move in direction of capture
        white_turn (bool) - True iff it's white's turn
    """

    # Can't capture if headed off the board
    if (row + row_delta < 0) or (row + row_delta >= 8):
        return False
    if (col + col_delta < 0) or (col + col_delta >= 8):
        return False

    # Can't capture if piece in that direction is not of appropriate color or missing
    enemy_color = BLACK if white_turn else WHITE
    if board[row + row_delta][col + col_delta] != enemy_color:
        return False

    # At least one enemy piece in this direction, so just need to scan until we
    # find a friendly piece (return True) or hit an empty spot or edge of board
    # (return False)
    friendly_color = WHITE if white_turn else BLACK
    scan_row = row + 2 * row_delta  # row of first scan position
    scan_col = col + 2 * col_delta  # col of first scan position
    while 0 <= scan_row < 8 and 0 <= scan_col < 8:
        if board[scan_row][scan_col] == NOBODY:
            return False
        if board[scan_row][scan_col] == friendly_color:
            return True
        scan_row += row_delta
        scan_col += col_delta
    return False


def capture(board, row, col, white_turn):
    """Destructively change a board to represent capturing a piece with a move at (row,col).

    The board's already a copy made specifically for the purpose of representing this move,
    so there's no point in copying it again.  We'll return the board anyway.

    Args:
        board (numpy 2D int array) - The Othello board - will be destructively modified
        row (int) - row of move
        col (int) - col of move
        white_turn (bool) - True iff it's white's turn
    Returns:
        The board, though this isn't necessary since it's destructively modified
    """

    # Check in each direction as to whether flips can happen -- if they can, start flipping
    enemy_color = BLACK if white_turn else WHITE
    for row_delta, col_delta in DIRECTIONS:
        if captures_in_dir(board, row, row_delta, col, col_delta, white_turn):
            flip_row = row + row_delta
            flip_col = col + col_delta
            while board[flip_row][flip_col] == enemy_color:
                board[flip_row][flip_col] = -enemy_color
                flip_row += row_delta
                flip_col += col_delta
    return board


def ai_play(depth, grid):
    """Interactive play, for demo purposes.  Assume AI is white and goes first."""
    # White turn (AI)
    legal_moves = generate_legal_moves(grid.board, True)
    if legal_moves:  # (list is non-empty)
        best_val = float("-inf")
        best_move = None
        for move in legal_moves:
            new_board = play_move(grid.board, move, True)
            move_val = minimax_value(new_board, True, depth, float("-inf"), float("inf"))
            if move_val > best_val:
                best_move = move
                best_val = move_val
        grid.board = play_move(grid.board, best_move, True)


def redraw_main_menu(grid, easy_button, medium_button, hard_button, expert_button):
    grid.window.fill((207, 185, 151))
    font = pygame.font.SysFont("Arial", 90)
    title = font.render("Othello", 3, WHIT)
    grid.window.blit(title, (grid.width / 2 - title.get_width() / 2, 30))

    pygame.draw.rect(grid.window, BLAC, easy_button)
    pygame.draw.rect(grid.window, BLAC, medium_button)
    pygame.draw.rect(grid.window, BLAC, hard_button)
    pygame.draw.rect(grid.window, BLAC, expert_button)

    fnt = pygame.font.SysFont('Arial', 60)
    easy_text = fnt.render('Easy', 2, WHIT)
    medium_text = fnt.render('Medium', 2, WHIT)
    hard_text = fnt.render('Hard', 2, WHIT)
    expert_text = fnt.render('Expert', 2, WHIT)

    grid.window.blit(easy_text, (grid.width / 2 - easy_text.get_width() / 2,
                                 easy_button.centery - easy_text.get_height() / 2))
    grid.window.blit(medium_text, (grid.width / 2 - medium_text.get_width() / 2, medium_button.centery -
                                   medium_text.get_height() / 2))
    grid.window.blit(hard_text, (grid.width / 2 - hard_text.get_width() / 2, hard_button.centery -
                                 hard_text.get_height() / 2))
    grid.window.blit(expert_text, (grid.width / 2 - expert_text.get_width() / 2, expert_button.centery -
                                   expert_text.get_height() / 2))
    pygame.display.update()


def main_menu():
    window = pygame.display.set_mode((640, 690))
    pygame.display.set_caption("Othello MiniMax")
    grid = Grid(8, 8, 640, 640, window)
    run = True
    clock = pygame.time.Clock()
    clock.tick(60)
    height = 100
    width = 400

    easy_button = pygame.Rect((grid.width / 2 - width / 2, 70 + height, width, height))
    medium_button = pygame.Rect((grid.width / 2 - width / 2, 100 + height * 2, width, height))
    hard_button = pygame.Rect((grid.width / 2 - width / 2, 130 + height * 3, width, height))
    expert_button = pygame.Rect((grid.width / 2 - width / 2, 160 + height * 4, width, height))

    redraw_main_menu(grid, easy_button, medium_button, hard_button, expert_button)

    while run:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

            if event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                x, y = pos[0], pos[1]

                if easy_button.collidepoint(x, y):
                    game(grid, 1)
                    grid = Grid(8, 8, 640, 640, grid.window)
                    redraw_main_menu(grid, easy_button, medium_button, hard_button, expert_button)
                elif medium_button.collidepoint(x, y):
                    game(grid, 3)
                    grid = Grid(8, 8, 640, 640, grid.window)
                    redraw_main_menu(grid, easy_button, medium_button, hard_button, expert_button)
                elif hard_button.collidepoint(x, y):
                    game(grid, 5)
                    grid = Grid(8, 8, 640, 640, grid.window)
                    redraw_main_menu(grid, easy_button, medium_button, hard_button, expert_button)
                elif expert_button.collidepoint(x, y):
                    game(grid, 7)
                    grid = Grid(8, 8, 640, 640, grid.window)
                    redraw_main_menu(grid, easy_button, medium_button, hard_button, expert_button)


def game(grid, depth):
    white_turn = True
    white_score = 0
    black_score = 0
    menu_button_width = 150
    menu_button_height = 40
    main_menu_button = pygame.Rect((grid.width / 2 - menu_button_width / 2, grid.height + 5, menu_button_width,
                                    menu_button_height))
    redraw_window(grid, white_score, black_score, main_menu_button)
    run = True

    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

            if event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                x, y = pos[0], pos[1]

                if main_menu_button.collidepoint(x, y):
                    run = False

            if check_game_over(grid.board) == NOBODY:
                if white_turn:
                    ai_play(depth, grid)
                    pygame.time.delay(1000)
                    white_turn = False
                    white_score = np.count_nonzero(grid.board == WHITE)
                    black_score = np.count_nonzero(grid.board == BLACK)
                    redraw_window(grid, white_score, black_score, main_menu_button)

                legal_moves = generate_legal_moves(grid.board, False)

                if legal_moves:
                    button = pygame.mouse.get_pressed()
                    if button[0] and not white_turn:
                        pos = pygame.mouse.get_pos()
                        player_move = grid.click(pos)

                        if player_move and player_move in legal_moves:
                            grid.board = play_move(grid.board, player_move, False)
                            white_turn = True
                            white_score = np.count_nonzero(grid.board == WHITE)
                            black_score = np.count_nonzero(grid.board == BLACK)
                            redraw_window(grid, white_score, black_score, main_menu_button)
                            break
                else:
                    white_turn = True

            else:
                winner = find_winner(grid.board)
                if winner == WHITE:
                    print("White won!")
                elif winner == BLACK:
                    print("Black won!")
                else:
                    print("Tie!")


main_menu()
pygame.quit()
