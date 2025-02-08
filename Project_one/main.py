import numpy as np
import random

# Constants
BOARD_SIZE = 8
EMPTY, BLACK, WHITE = 0, 1, 2

class Reversi:
    def __init__(self, ai_black=None, ai_white=None):
        """Initialize an 8x8 board with optional AI players."""
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)
        self.current_player = BLACK
        self.ai_players = {BLACK: ai_black, WHITE: ai_white}
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
                symbol = "." if self.board[i][j] == EMPTY else "B" if self.board[i][j] == BLACK else "W"
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
        return [(row, col) for row in range(BOARD_SIZE) for col in range(BOARD_SIZE) if self.is_valid_move(row, col)]

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
                    print("\nGame Over!")
                    self.display_board()
                    self.display_winner()
                    break
                continue

            # Check if AI should move
            ai_player = self.ai_players[self.current_player]
            if ai_player:
                print(f"AI ({'Black' if self.current_player == BLACK else 'White'}) is thinking...")
                ai_player.move(self)
            else:
                valid_moves = self.get_valid_moves()
                formatted_moves = [(chr(col + ord('a')), row + 1) for row, col in valid_moves]
                print(f"Valid moves for {'Black' if self.current_player == BLACK else 'White'}: {formatted_moves}")
                move = input("Enter your move (e.g., e3 or 'q' to quit): ").strip().lower()
                if move == "q":
                    print("Game exited.")
                    break
                parsed_move = self.parse_input(move)
                if parsed_move is None or not self.make_move(*parsed_move):
                    print("Invalid move. Try again.")
                    continue

    def display_winner(self):
        """Determine and display the winner based on piece count."""
        black_count = np.sum(self.board == BLACK)
        white_count = np.sum(self.board == WHITE)

        print("\nFinal Score:")
        print(f"Black: {black_count} | White: {white_count}")

        if black_count > white_count:
            print("🏆 Black Wins!")
        elif white_count > black_count:
            print("🏆 White Wins!")
        else:
            print("It's a Tie!")



# AI Players
class RandomAI:
    def __init__(self, player):
        self.player = player

    def move(self, env):
        """Selects a random valid move."""
        moves = env.get_valid_moves()
        if moves:
            row, col = random.choice(moves)
            env.make_move(row, col)


class MiniMaxAI:
    def __init__(self, player, depth=3):
        self.player = player
        self.depth = depth

    def move(self, env):
        """Minimax-based move selection with Alpha-Beta Pruning."""
        best_move = None
        best_score = -np.inf
        alpha, beta = -np.inf, np.inf

        for move in env.get_valid_moves():
            env_copy = self.simulate_move(env, move)  # Create a simulated board
            score = self.min_player(env_copy, self.depth - 1, alpha, beta)

            if score > best_score:
                best_score = score
                best_move = move

        if best_move:
            env.make_move(*best_move)  # AI executes the move in the real game

    def min_player(self, env, depth, alpha, beta):
        """Minimizing opponent’s best possible move."""
        if depth == 0 or not env.get_valid_moves():
            return self.evaluate_board(env)

        best_score = np.inf
        for move in env.get_valid_moves():
            env_copy = self.simulate_move(env, move)
            score = self.max_player(env_copy, depth - 1, alpha, beta)
            best_score = min(best_score, score)
            beta = min(beta, score)
            if beta <= alpha:  # Alpha-Beta pruning
                break
        return best_score

    def max_player(self, env, depth, alpha, beta):
        """Maximizing AI’s best move."""
        if depth == 0 or not env.get_valid_moves():
            return self.evaluate_board(env)

        best_score = -np.inf
        for move in env.get_valid_moves():
            env_copy = self.simulate_move(env, move)
            score = self.min_player(env_copy, depth - 1, alpha, beta)
            best_score = max(best_score, score)
            alpha = max(alpha, score)
            if beta <= alpha:  # Alpha-Beta pruning
                break
        return best_score

    def evaluate_board(self, env):
        """Basic board evaluation: count the difference in pieces."""
        return np.sum(env.board == self.player) - np.sum(env.board == (BLACK if self.player == WHITE else WHITE))

    def simulate_move(self, env, move):
        """Creates a simulated copy of the environment to test a move."""
        env_copy = Reversi(env.ai_players[BLACK], env.ai_players[WHITE])
        env_copy.board = env.board.copy()
        env_copy.current_player = env.current_player
        env_copy.make_move(*move)  # Apply the move in the simulated environment
        return env_copy



if __name__ == "__main__":
    print("Welcome to Reversi!")
    print("Choose AI players:")
    print("1. Human vs Human")
    print("2. Human vs AI (AI as White)")
    print("3. AI vs Human (AI as Black)")
    print("4. AI vs AI (Both AI)")
    print("5. AI vs Random AI")
    print("6. Random AI vs Random AI")

    choice = input("Enter a number (1-6): ").strip()

    ai_black = None
    ai_white = None
    if choice == "2":
        ai_white = MiniMaxAI(WHITE, depth=3)  # AI as White
    elif choice == "3":
        ai_black = MiniMaxAI(BLACK, depth=3)  # AI as Black
    elif choice == "4":
        ai_black = MiniMaxAI(BLACK, depth=3)  # AI as Black
        ai_white = MiniMaxAI(WHITE, depth=3)  # AI as White
    elif choice == "5":
        ai_black = MiniMaxAI(BLACK, depth=3)
        ai_white = RandomAI(WHITE)
    elif choice == "6":
        ai_black = RandomAI(BLACK)
        ai_white = RandomAI(WHITE)
    game = Reversi(ai_black, ai_white)
    game.play()
