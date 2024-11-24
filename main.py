import argparse
import chess
from run_lc0 import get_best_move

default_fen: str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def main(args):
    best_move: str = get_best_move(
        args.weights_path,
        args.from_fen,
        args.nodes
    )

    print(best_move)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights_path", type=str, default="/local/maia-chess/maia_weights/maia-1200.pb.gz"
    )
    parser.add_argument(
        "--nodes", type=int, default=1
    )
    parser.add_argument(
        "--from_fen", type=str, default=default_fen
    )
    args = parser.parse_args()

    main(args)
