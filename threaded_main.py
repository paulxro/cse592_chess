import argparse
import os
import pathlib
import chess
import threading
import chess.pgn

from stockfish import Stockfish

# from run_lc0 import LCEngine
from test import LCEngine

engines = {}


default_fen: str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
maia_ratings: list = [1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900]

def _get_nearest_rating(elo: int) -> int:
    min_diff_idx: int = 0

    for i, weight in enumerate(maia_ratings):
        if abs(weight - elo) < abs(maia_ratings[min_diff_idx] - elo):
            min_diff_idx = i
    
    return maia_ratings[min_diff_idx]


def _run_predict(fen: str, rating: int, idx: int, predicted) -> None:
    # print(f'RATING={rating}, FEN={fen}')
    predicted[idx] = engines[rating].predict_move(fen, 1)


def main(args):
    for elo in maia_ratings:
        engines[elo] = LCEngine(elo)

    with open(args.pgn_file) as pgn:
        num_considered: int = 0
        num_correct_predicted: int = 0
        num_correct_optimal: int = 0

        for game_idx in range(5):
            game = chess.pgn.read_game(pgn)
            board = game.board()

            thread_pool = []


            inputs: list[str] = []
            outputs: list[str] = []

            for move in game.mainline_moves():
                inputs.append(board.fen())
                board.push(move)
                outputs.append(board.fen())

            predicted: list[str] = [''] * len(outputs)

            rating = -1

            if not args.rating:
                avg_elo = (int(game.headers['WhiteElo']) + int(game.headers['BlackElo'])) / 2
                rating = _get_nearest_rating(avg_elo)
            else:
                rating = args.rating
            
            print(f'Game [{game_idx}] -- AVG ELO: {avg_elo}; using {rating}')

            board.reset()

            for i, move in enumerate(game.mainline_moves()):

                pred_thread = threading.Thread(
                    target=(_run_predict),
                    args=(inputs[i], rating, i, predicted)
                )

                print(id(pred_thread))

                pred_thread.start()

                thread_pool.append(pred_thread)

                if i == 1:
                    break
            
            for thread in thread_pool:
                thread.join()
            
            for move in predicted:
                print(move)





        #         stockfish = Stockfish('/local/stockfish/stockfish-ubuntu-x86-64-avx2')
        #         stockfish.set_fen_position(initial_board)
        #         board.push(chess.Move.from_uci(
        #             stockfish.get_best_move()
        #         ))
        #         best_board = board.fen()

        #         board.pop()

        #         predicted_move: str = engines[rating].predict_move(
        #             initial_board, args.nodes
        #         )

        #         board.push(chess.Move.from_uci(
        #             predicted_move
        #         ))
        #         predicted_board = board.fen()

        #         board.pop()

        #         board.push(move)
        #         output_board = board.fen()

                # if predicted_board == output_board:
                #     num_correct_predicted += 1
        #         if best_board == output_board:
        #             num_correct_optimal += 1
        #         num_considered += 1
        
        # if num_considered == 0:
        #     print("No moves have been looked at.")
        #     return

        # print(
        #     f'Lc0\tcorrect: {num_correct_predicted} / {num_considered} ({num_correct_predicted/num_considered})'
        # )
        # print(
        #     f'SF\tcorrect: {num_correct_optimal} / {num_considered} ({num_correct_optimal/num_considered})'
        # )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights_path", type=pathlib.Path, default='/local/maia-chess/maia_weights/'
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
