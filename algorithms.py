class HumanPlayer:
    def __init__(self, player, ui):
        self.player = player
        self.pos = None
        self.ui = ui
        ui.register_click_callback(self.click_callback)

    def click_callback(self, row, col):
        self.pos = (row, col)

    def get_click(self):
        self.pos = None
        while self.pos is None: self.ui.update()
        return self.pos

    def move(self, env):
        board = env.get_board()
        pieces = set(zip(*np.where(board == self.player)))
        empty = set(zip(*np.where(board == 0)))

        try:
            move = {
                'player': self.player
            }
            if len(pieces) >= 3:
                pos = self.get_click()
                while pos not in pieces: pos = self.get_click()
                move['take'] = pos

            pos = self.get_click()
            while pos not in empty: pos = self.get_click()
            move['put'] = pos

            env.make_move(move)
        except: pass

class RandomPlayer:
    def __init__(self, player):
        self.player = player

    def move(self, env):
        moves = env.get_possible_moves(self.player)
        move = np.random.choice(moves, 1)[0]
        env.make_move(move)

class MiniMax:
    def __init__(self, player):
        self.player = player

    def move(self, env):
        board = env.get_board()
        best_score = -np.inf
        best_move = None
        for move in possible_moves(board, self.player):
            new_board = update_board(board, move)
            score = self.min_player(new_board)
            if score > best_score:
                best_score = score
                best_move = move
        env.make_move(best_move)

    def min_player(self, board):
        if has_won(board, self.player): return 1
        elif has_won(board, -self.player): return -1
        scores = [self.max_player(update_board(board, move)) for move in possible_moves(board, -self.player)]
        return min(scores)

    def max_player(self, board):
        if has_won(board, self.player): return 1
        elif has_won(board, -self.player): return -1
        scores = [self.min_player(update_board(board, move)) for move in possible_moves(board, -self.player)]
        return max(scores)

class EvalMiniMax:
    def __init__(self, player, max_depth=3):
        self.player = player
        self.max_depth = max_depth
        self.piece_scores = np.array([[1, 0, 1],
                                      [0, 2, 0],
                                      [1, 0, 1]])

    def move(self, env):
        board = env.get_board()
        best_score = -np.inf
        best_move = None
        for move in possible_moves(board, self.player):
            score = self.min_player(update_board(board, move), 1)
            if score > best_score:
                best_score = score
                best_move = move
        env.make_move(best_move)

    def min_player(self, board, depth):
        if has_won(board, self.player): return 10
        elif has_won(board, -self.player): return -10
        elif depth == self.max_depth: return np.sum(self.player * board * self.piece_scores)
        scores = [self.max_player(update_board(board, move), depth+1) for move in possible_moves(board, -self.player)]
        return min(scores)

    def max_player(self, board, depth):
        if has_won(board, self.player): return 10
        elif has_won(board, -self.player): return -10
        elif depth == self.max_depth: return np.sum(self.player * board * self.piece_scores)
        scores = [self.min_player(update_board(board, move), depth+1) for move in possible_moves(board, self.player)]
        return max(scores)

class EvalAlphaBeta:
    def __init__(self, player, max_depth=3):
        self.player = player
        self.max_depth = max_depth
        self.piece_scores = np.array([[1, 0, 1],
                                      [0, 2, 0],
                                      [1, 0, 1]])

    def move(self, env):
        board = env.get_board()
        best_score = -np.inf
        best_move = None
        for move in possible_moves(board, self.player):
            score = self.min_player(update_board(board, move), best_score, np.inf, 1)
            if score > best_score:
                best_score = score
                best_move = move
        env.make_move(best_move)

    def min_player(self, board, alpha, beta, depth):
        if has_won(board, self.player): return 10
        elif has_won(board, -self.player): return -10
        elif depth == self.max_depth: return np.sum(self.player * board * self.piece_scores)
        score = np.inf
        for move in possible_moves(board, -self.player):
            score = min(score, self.max_player(update_board(board, move), alpha, beta, depth+1))
            beta = min(beta, score)
            if score <= alpha: break
        return score

    def max_player(self, board, alpha, beta, depth):
        if has_won(board, self.player): return 10
        elif has_won(board, -self.player): return -10
        elif depth == self.max_depth: return np.sum(self.player * board * self.piece_scores)
        score = -np.inf
        for move in possible_moves(board, self.player):
            score = max(score, self.max_player(update_board(board, move), alpha, beta, depth+1))
            alpha = max(alpha, score)
            if score >= beta: break
        return score

if __name__ == "__main__":
    print("This module provides AI algorithms for the Reversi game.")
