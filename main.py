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

    def switch_player(self):
        """Switch to the other player."""
        self.current_player = BLACK if self.current_player == WHITE else WHITE

    def is_valid_move(self, row, col):
        """Check if a move is valid (must flip at least one opponent piece)."""
        if self.board[row][col] != EMPTY:
            return False
        
        opponent = WHITE if self.current_player == BLACK else BLACK
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        
        for dx, dy in directions:
            x, y = row + dx, col + dy
            has_opponent_between = False

            while 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE and self.board[x][y] == opponent:
                x += dx
                y += dy
                has_opponent_between = True

            if has_opponent_between and 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE and self.board[x][y] == self.current_player:
                return True  # Valid move found
        
        return False  # No valid direction found

    def get_valid_moves(self):
        """Get all valid moves for the current player."""
        valid_moves = []
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                if self.is_valid_move(row, col):
                    valid_moves.append((row, col))
        return valid_moves

    def flip_pieces(self, row, col):
        """Flip pieces after a valid move."""
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        opponent = WHITE if self.current_player == BLACK else BLACK
        
        for dx, dy in directions:
            x, y = row + dx, col + dy
            pieces_to_flip = []

            while 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE and self.board[x][y] == opponent:
                pieces_to_flip.append((x, y))
                x += dx
                y += dy

            if pieces_to_flip and 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE and self.board[x][y] == self.current_player:
                for fx, fy in pieces_to_flip:
                    self.board[fx][fy] = self.current_player

    def make_move(self, row, col):
        """Make a move and flip pieces."""
        if not self.is_valid_move(row, col):
            return False
        
        self.board[row][col] = self.current_player
        self.flip_pieces(row, col)
        self.switch_player()
        return True

    def parse_input(self, move):
        """Convert chess notation (e.g., 'e3') to board indices."""
        if len(move) != 2:
            return None
        col = ord(move[0].lower()) - ord('a')  # Convert 'a' to 0, 'b' to 1, ...
        row = int(move[1]) - 1  # Convert '1' to 0, '2' to 1, ...
        if 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE:
            return row, col
        return None

    def play(self):
        """Game loop."""
        while True:
            self.display_board()
            valid_moves = self.get_valid_moves()
            
            if not valid_moves:
                print(f"No valid moves for {'Black' if self.current_player == BLACK else 'White'}. Skipping turn.")
                self.switch_player()
                if not self.get_valid_moves():  # No moves for both players → Game Over
                    print("Game Over!")
                    break
                continue

            print(f"Player {'Black' if self.current_player == BLACK else 'White'}'s turn.")
            move = input("Enter your move (e.g., e3): ").strip().lower()

            if move == "q":
                print("Game exited.")
                break

            parsed_move = self.parse_input(move)
            if parsed_move is None or not self.make_move(*parsed_move):
                print("Invalid move. Try again.")
                continue

if __name__ == "__main__":
    game = Reversi()
    game.play()
