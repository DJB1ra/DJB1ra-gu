import numpy as np
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='pi_montecarlo',
        description='jeff'
    )

    parser.add_argument('n', help='number of steps', type=int)
    args=parser.parse_args()

    n=args.n
    m=0
    rng=np.random.default_rng()
    for _ in range(n):
        x=rng.random()
        y=rng.random()
        if x*x+y*y < 1:
            m += 1
        print(4*m/n)

        