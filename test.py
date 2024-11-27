import subprocess
import threading
from typing import TextIO, Optional

LC0_TIMEOUT: int = 2 # Seconds

class LCEngine():
    def __init__(self, elo):
        self.elo = elo
        nearest_rating = self._get_nearest_rating(self.elo)
        weights_path = "/local/maia-chess/maia_weights/" + f'maia-{nearest_rating}.pb.gz'
        
        self.process = subprocess.Popen(
            ['lc0', f'--weights={weights_path}'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        if self.process.poll() is not None:
            print("Failed to start lc0.")
            return
        
    def _get_nearest_rating(self, elo: int) -> int:
        maia_ratings: list = [1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900]
        min_diff_idx: int = 0
        for i, weight in enumerate(maia_ratings):
            if abs(weight - elo) < abs(maia_ratings[min_diff_idx] - elo):
                min_diff_idx = i
        
        return maia_ratings[min_diff_idx]
    
    def _parse_best_move(self, pipe: TextIO, ret: dict) -> None:
        best_move: str = ''
        while 'bestmove' not in best_move:
            best_move = pipe.readline()

        ret['best_move'] = best_move.split()[1]
    
    def predict_move(self, fen, nodes):
        self.process.stdin.write(f'position fen {fen}\n')
        self.process.stdin.flush()
        prompt_cmd: str = f'go nodes {nodes}\n'
        try:
            self.process.stdin.write(prompt_cmd)
            self.process.stdin.flush()

            lc0_result: dict[str, str] = {}

            lc0_thread = threading.Thread(
                target=self._parse_best_move,
                args=(self.process.stdout, lc0_result)
            )

            lc0_thread.start()

            lc0_thread.join(LC0_TIMEOUT)
            
            if 'best_move' not in lc0_result.keys():
                return None

            return lc0_result['best_move']
        except Exception as e:
            print(e)
            return None
    
        # finally:
        #     self.process.stdin.close()
        #     self.process.stdout.close()
        #     self.process.stderr.close()
        #     self.process.terminate()
        return None
        
    
        

def _parse_best_move(pipe: TextIO, ret: dict) -> None:
    best_move: str = ''
    while 'bestmove' not in best_move:
        best_move = pipe.readline()

    ret['best_move'] = best_move.split()[1]


def predict_move(weights_path: str, fen: str, nodes: int = 1) -> Optional[str]:
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

        lc0_thread.join(LC0_TIMEOUT)
        
        if 'best_move' not in lc0_result.keys():
            return None

        return lc0_result['best_move']

    finally:
        process.stdin.close()
        process.stdout.close()
        process.stderr.close()
        process.terminate()



