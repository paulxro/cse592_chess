import argparse
import chess
import chess.pgn
import threading
import statistics
import pprint
import sys

from stockfish import Stockfish
from run_lc0 import predict_move
from test import LCEngine

game_results_mut = threading.Lock()

DELTA: int = 1
ELO_DELTA: int = 500

TEST_DELTA: list = [1, 3, 5, 7]

TEST_NUM_GAMES: int = 25

# 3 x 4 x 25

default_fen: str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

maia_ratings: list = [1200, 1500, 1800]

game_results: list = []

def _run_position_eval(fen: str, sf, lc, delta: int) -> list:
    sf.set_fen_position(fen)
    sf_eval = sf.get_evaluation()

    evals: list = []


    board = chess.Board(fen=fen)

    for _ in range(delta):
        try:
            board.push(chess.Move.from_uci(
                lc.predict_move(board.fen(), 1)
            ))
            if board.is_checkmate():
                break
            sf.set_fen_position(board.fen())
            timestep_eval = sf.get_evaluation()

            if 'mate' in timestep_eval['type']:
                continue

            evals.append(timestep_eval['value'])
        except Exception as e:
            print(e)
            return [None, None]
    
    # sf.set_fen_position(board.fen())
    # lc_eval = sf.get_evaluation()

    if not len(evals):
        return [None, None]

    lc_eval = statistics.mean(evals)

    if 'mate' in sf_eval['type']:
        return [None, None]

    # return [sf_eval['value'], lc_eval['value']]
    return [sf_eval['value'], lc_eval]


def _run_game_eval(game, depth: int = 15, delta: int = DELTA, res: dict = None):
    if not game.headers['WhiteElo'].isnumeric() or not game.headers['BlackElo'].isnumeric():
        return []

    white_elo: float = float(game.headers['WhiteElo'])
    black_elo: float = float(game.headers['BlackElo'])

    avg_elo: float = (white_elo + black_elo) / 2

    sf_engine = Stockfish('/local/stockfish/stockfish-ubuntu-x86-64-avx2', depth=depth)
    lc_engine = LCEngine(avg_elo)

    board = game.board()

    positions: list = []
    results: list   = []

    for move in game.mainline_moves():
        positions.append(board.fen())
        board.push(move)
    
    for i in range(len(positions) - delta):
        sf_engine.set_fen_position(positions[i + delta])
        actual_eval = sf_engine.get_evaluation()

        if 'mate' in actual_eval['type']:
            continue
    
        actual_eval = actual_eval['value']

        sf_eval, lc_eval = _run_position_eval(
            positions[i],
            sf_engine,
            lc_engine,
            delta=delta
        )

        if sf_eval is None or lc_eval is None:
            continue
        
        results.append(
            [abs(sf_eval - actual_eval), abs(lc_eval - actual_eval), i]
        )
    
    sf_avg: float = 0.0
    lc_avg: float = 0.0

    for result in results:
        sf_dev, lc_dev, _ = result

        sf_avg += sf_dev
        lc_avg += lc_dev
    
    # print(f'[{avg_elo}]: {sf_avg / len(results)} {lc_avg / len(results)}')

    # sf_res = []
    # lc_res = []
    
    # sf_res_initialthird = []
    # lc_res_initialthird = []
    
    # sf_res_middlethird = []
    # lc_res_middlethird = []
    
    # sf_res_finalthird = []
    # lc_res_finalthird = []

    # count = 0
    # n = len(results)
    # for result in results:
    #     sf_dev, lc_dev, _ = result
    #     if count < (n // 3):
    #         sf_res_initialthird.append(sf_dev)
    #         lc_res_initialthird.append(lc_dev)
    #     elif count < ((2 * n) // 3):
    #         sf_res_middlethird.append(sf_dev)
    #         lc_res_middlethird.append(lc_dev)
    #     else:
    #         sf_res_finalthird.append(sf_dev)
    #         lc_res_finalthird.append(lc_dev)
    #     sf_res.append(sf_dev)
    #     lc_res.append(lc_dev)
    #     count += 1
    
    # sf_res_means = [statistics.mean(sf_res_initialthird),
    #                 statistics.mean(sf_res_middlethird),
    #                 statistics.mean(sf_res_finalthird)]
    # lc_res_means = [statistics.mean(lc_res_initialthird),
    #                 statistics.mean(lc_res_middlethird),
    #                 statistics.mean(lc_res_finalthird)]
    
    # with game_results_mut:
        # game_results.append([sf_avg / len(results), lc_avg / len(results)])
        # game_results.append([statistics.median(sf_res), statistics.median(lc_res)])
        # game_results.append([sf_res_means, lc_res_means])
    with res['mut']:
        res['data'].append([sf_avg / len(results), lc_avg / len(results)])

    # print(f"Done game[{game.game_id}]")    

        



# def _run_game_eval(game):
    # for every move
        # get current SF eval
        # Run predict move thread with delta
        # get new SF eval (pred pos)

        # make real delta moves
        # get real SF eval

        # compare real SF eval w/
            # new SF eval & current
    
    # avg_elo = (int(game.headers['WhiteElo']) + int(game.headers['BlackElo'])) / 2

    # stockfish = Stockfish('/local/stockfish/stockfish-ubuntu-x86-64-avx2', depth=15)
    # opt_stockfish = Stockfish('/local/stockfish/stockfish-ubuntu-x86-64-avx2', depth=15)
    # lc = LCEngine(avg_elo)
    # board = game.board()
    # M=[]
    # evals=[]
    # evals_dif=[]

    # for move in game.mainline_moves():
    #     board.push(move)
    #     M.append(board.fen())
    #     stockfish.set_fen_position(board.fen())
    #     evals.append(stockfish.get_evaluation())
    
    # for i in range(len(M)-DELTA):
    #     cur_eval=evals[i]

    #     cur_board=chess.Board(fen=M[i])
    #     try:
    #         for j in range(DELTA):
    #             cur_board.push(chess.Move.from_uci(lc.predict_move(cur_board.fen(), 1)))
    #     except Exception as e:
    #         print(e)
    #         break
        
    #     opt_stockfish.set_fen_position(cur_board.fen())
    #     lc0_eval= opt_stockfish.get_evaluation()
    #     real_eval=evals[i+DELTA]
    #     if real_eval['type']=='mate' or cur_eval['type']=='mate' or lc0_eval['type']=='mate':
    #         evals_dif.append([None,None,i+1])
    #     else:
    #         evals_dif.append([abs(real_eval['value']-cur_eval['value']), abs(real_eval['value']-lc0_eval['value']),i+1])
    
    # sf_eval_total = 0
    # lc_eval_total = 0
    # count = 0

    # for diff in evals_dif:
    #     sf_eval, lc_eval, move = diff
        
    #     if not sf_eval or not lc_eval:
    #         continue
            
    #     count += 1
    #     sf_eval_total += sf_eval
    #     lc_eval_total += lc_eval

    # if count == 0:
    #     return evals_dif

    # avg_sf_diff = sf_eval_total / count
    # avg_lc_diff = lc_eval_total / count

    # # if avg_sf_diff < avg_lc_diff:
    # #     return []

    # print(f'[{avg_elo}]', avg_sf_diff, avg_lc_diff)
    
    # return evals_dif



def _run_predict(fen: str, weights_path: str, idx: int, predicted) -> None:
    predicted[idx] = predict_move(weights_path, fen, 1)

def _get_nearest_rating(elo: int) -> int:
    min_diff_idx: int = 0

    for i, weight in enumerate(maia_ratings):
        if abs(weight - elo) < abs(maia_ratings[min_diff_idx] - elo):
            min_diff_idx = i
    
    return maia_ratings[min_diff_idx]


def main(args):
    results: dict = {}

    for rating in maia_ratings:
        game_count: int = 0
        results[rating] = {}

        with open(args.pgn_file) as pgn:
            while game_count < TEST_NUM_GAMES:
                if not pgn:
                    break

                game = chess.pgn.read_game(pgn)

                if not game.headers['WhiteElo'].isnumeric() or not game.headers['BlackElo'].isnumeric():
                    continue
                
                if len(list(game.mainline_moves())) < max(TEST_DELTA):
                    continue

                white_elo: float = float(game.headers['WhiteElo'])
                black_elo: float = float(game.headers['BlackElo'])

                avg_elo:   float = (white_elo + black_elo) / 2

                if _get_nearest_rating(avg_elo) != rating:
                    continue

                if abs(white_elo - black_elo) > ELO_DELTA:
                    continue
                
                game.__setattr__('game_id', 'lol')
                delta_tpool: list = []

                for delta in TEST_DELTA:
                    if delta not in results[rating].keys():
                        results[rating][delta] = {}
                        results[rating][delta]['data'] = []
                        results[rating][delta]['mut'] = threading.Lock()
                    
                    delta_thread = threading.Thread(
                        target=(_run_game_eval),
                        args=(game, 15, delta, results[rating][delta])
                    )

                    delta_thread.start()

                    delta_tpool.append(delta_thread)

                for thread in delta_tpool:
                    thread.join()
                
                pprint.pprint(results)
                print("---------------------")

                print(f'[{rating}] Done game {game_count}', file=sys.stderr)
                
                game_count += 1


             
                    

                    
                    

                



    # with open(args.pgn_file) as pgn:

    #     thread_pool = []

    #     for game_idx in range(350):
    #         game = chess.pgn.read_game(pgn)

    #         game.__setattr__('game_id', game_idx)

    #         if not game.headers['WhiteElo'].isnumeric() or not game.headers['BlackElo'].isnumeric():
    #             continue
                
    #         if abs((int(game.headers['WhiteElo']) - int(game.headers['BlackElo']))) > ELO_DELTA:
    #             continue

    #         avg_elo = (int(game.headers['WhiteElo']) + int(game.headers['BlackElo'])) / 2

    #         # if avg_elo > 1200 or len(list(game.mainline_moves())) <= DELTA:
    #         #     continue

    #         if (not (1300 < avg_elo < 1500)) or len(list(game.mainline_moves())) <= DELTA:
    #             continue
        
    #         # print(game_idx)

    #         # if game_idx in [4, 5,6]:
    #         #     continue


    #         eval_thread = threading.Thread(
    #             target=_run_game_eval,
    #             args=(game,)
    #         )

    #         eval_thread.start()

    #         thread_pool.append(eval_thread)

    #         # break
        
    #     for thread in thread_pool:
    #         thread.join()

    #     sf_avg: float = 0.0
    #     lc_avg: float = 0.0
        
    #     sf_avgs = [0] * 3
    #     lc_avgs = [0] * 3
        
    #     for game_res in game_results:
    #         sf_res, lc_res = game_res
    #         for i, v in enumerate(sf_res):
    #             sf_avgs[i] += v
    #         for i, v in enumerate(lc_res):
    #             lc_avgs[i] += v
            
        
            
    #     with open(f"out_delta_{DELTA}_thirds.txt", "w") as f:
    #         for i in range(3):
    #             f.write(f"{i}, {sf_avgs[i] / len(game_results)} {lc_avgs[i] / len(game_results)}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights_path", type=str, default='/local/maia-chess/maia_weights/'
    )
    parser.add_argument(
        "--rating", type=int, default=0
    )
    parser.add_argument(
        "--pgn_file", type=str, default="/local/cse592_chess/data/out.pgn"
    )
    parser.add_argument(
        "--nodes", type=int, default=1
    )
    args = parser.parse_args()

    main(args)