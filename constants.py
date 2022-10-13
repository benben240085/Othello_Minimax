BLAC = (0, 0, 0)
WHIT = (255, 255, 255)
LIGHT_BLUE = (137, 160, 190)
MAROON = (46, 0, 0)

PADDING_X = 40
PADDING_Y = 40

NUM_COLS = 8
# With these constant values for players, flipping ownership is just a sign change
WHITE = 1
NOBODY = 0
BLACK = -1

TIE = 2  # NOT the value of a tie, which is 0 - just an arbitrary enum for end-of-game

WIN_VAL = 100
WHITE_TO_PLAY = True

# We'll sometimes iterate over this to look in all 8 directions from a particular square.
# The values are the "delta" differences in row, col from the original square.
# (Hence no (0,0), which would be the same square.)
DIRECTIONS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]