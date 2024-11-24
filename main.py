import argparse
import chess
import chess.pgn

from run_lc0 import predict_move

default_fen: str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def main(args):
    with open(args.pgn_file) as pgn:
        num_considered: int = 0
        num_correct: int = 0

        for _ in range(1):
            game = chess.pgn.read_game(pgn)
            board = game.board()

            for move in game.mainline_moves():
                initial_board = board.fen()

                predicted_move: str = predict_move(
                    args.weights_path,
                    initial_board,
                    args.nodes
                )
                board.push(chess.Move.from_uci(
                    predicted_move
                ))
                predicted_board = board.fen()

                board.pop()

                board.push(move)
                output_board = board.fen()

                if predicted_board == output_board:
                    num_correct += 1
                num_considered += 1
        
        if num_considered == 0:
            print("No moves have been looked at.")
            return

        print(num_correct / num_considered)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights_path", type=str, default="/local/maia-chess/maia_weights/maia-1200.pb.gz"
    )
    parser.add_argument(
        "--pgn_file", type=str, default="/local/rel_eval/data/out.pgn"
    )
    parser.add_argument(
        "--nodes", type=int, default=1
    )
    args = parser.parse_args()

    main(args)
