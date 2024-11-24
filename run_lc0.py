import subprocess
import threading
from typing import TextIO, Optional

def _parse_best_move(pipe: TextIO, ret: dict) -> None:
    best_move: str = ''
    while 'bestmove' not in best_move:
        best_move = pipe.readline()

    ret['best_move'] = best_move.split()[1]


def get_best_move(weights_path: str, fen: str, nodes: int = 1) -> Optional[str]:
    process = subprocess.Popen(
        ['lc0', f'--weights={weights_path}'],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )

    if process.poll() is not None:
        print("Failed to start lc0.")
        return

    process.stdin.write(f'position fen {fen}\n')
    process.stdin.flush()

    prompt_cmd: str = f'go nodes {nodes}\n'

    try:
        process.stdin.write(prompt_cmd)
        process.stdin.flush()

        lc0_result: dict[str, str] = {}

        lc0_thread = threading.Thread(
            target=_parse_best_move,
            args=(process.stdout, lc0_result)
        )

        lc0_thread.start()

        lc0_thread.join(5)
        
        if 'best_move' not in lc0_result.keys():
            return None

        return lc0_result['best_move']

    finally:
        process.stdin.close()
        process.stdout.close()
        process.stderr.close()
        process.terminate()



