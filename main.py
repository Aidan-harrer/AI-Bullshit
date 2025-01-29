import numpy as np

# Constants
BOARD_SIZE = 8
EMPTY, BLACK, WHITE = 0, 1, 2

class Reversi:
    def __init__(self):
        """Initialize an 8x8 board with the starting Reversi configuration."""
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)
        self.current_player = BLACK
        self.initialize_board()

    def initialize_board(self):
        """Set up the initial 4 pieces in the center."""
        mid = BOARD_SIZE // 2
        self.board[mid-1][mid-1] = WHITE
        self.board[mid][mid] = WHITE
        self.board[mid-1][mid] = BLACK
        self.board[mid][mid-1] = BLACK

    def display_board(self):
        """Display the board in a readable format."""
        print("  a b c d e f g h")
        for i in range(BOARD_SIZE):
            print(i+1, end=" ")
            for j in range(BOARD_SIZE):
                symbol = "."
                if self.board[i][j] == BLACK:
                    symbol = "B"
                elif self.board[i][j] == WHITE:
                    symbol = "W"
                print(symbol, end=" ")
            print()

    def make_move(self, row, col):
        """Make a move and flip pieces."""
        if not self.is_valid_move(row, col):
            return False
        self.board[row][col] = self.current_player
        self.flip_pieces(row, col)
        self.switch_player()
        return True

    def switch_player(self):
        """Switch to the other player."""
        self.current_player = BLACK if self.current_player == WHITE else WHITE

    
    def is_valid_move(self, row, col):
        """Check if a move is valid."""
        if self.board[row][col] != EMPTY:
            return False
        return True
    
    def get_valid_moves(self):
        """Get all valid moves for the current player."""
        # Placeholder for actual valid move logic
        return [(0, 0)]
    
    def flip_pieces(self, row, col):
        """Flip pieces after a valid move."""
        # Placeholder for actual piece flipping logic
        pass

    def play(self):
            """Game loop."""
            #while True:
            self.display_board()
            print(f"Player {'Black' if self.current_player == BLACK else 'White'}'s turn.")
            # Get valid moves (not yet implemented)
            valid_moves = self.get_valid_moves()
            if not valid_moves:
                print("No valid moves available. Skipping turn.")
                self.switch_player()
                #continue
            
            # Player input (Placeholder)
            move = input("Enter your move (e.g., e3): ")
            # Convert move and apply logic (not yet implemented)
            row,col = move[0], move[1]
            self.make_move(row, col)
            self.switch_player()

if __name__ == "__main__":
    game = Reversi()
    game.play()
