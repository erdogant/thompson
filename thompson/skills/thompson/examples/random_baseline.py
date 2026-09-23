"""Randomized baseline on the built-in ads dataset."""

import thompson as th


def main():
    df = th.import_example()
    out = th.UCB_random(df)

    print(f"Method: {out['methodtype']}")
    print(f"Total reward: {out['total_reward']}")
    print(f"Trials: {len(out['cols_selected'])}")

    th.plot(out)


if __name__ == "__main__":
    main()
