
delta = 5

def _run_game_eval(game):
    # for every move
        # get current SF eval
        # Run predict move thread with delta
        # get new SF eval (pred pos)

        # make real delta moves
        # get real SF eval

        # compare real SF eval w/
            # new SF eval & current
    stockfish = Stockfish('/local/stockfish/stockfish-ubuntu-x86-64-avx2')
    board = game.board()
    M=[]
    M.append(board.fen())
    evals=[]
    evals_dif=[]
    for move in game.mainline_moves():
        board.push(move)
        M.append(board.fen())
        stockfish.set_fen_position(board.fen())
        evals.append(stockfish.get_evaluation())
    
    for i in range(len(M)-delta):
        cur_eval=evals[i]

        cur_board=chess.Board(fen=M[i])
        for j in range(delta):
            cur_board.push(predict_move(weights_path, cur_board.fen(), 1))
        
        stockfish.set_fen_position(cur_board.fen())
        lc0_eval= stockfish.get_evaluation()
        real_eval=evals[M[i+delta]]
        evals_dif.append([abs(real_eval-cur_eval), abs(real_eval-lc0_eval)])
    return evals_dif