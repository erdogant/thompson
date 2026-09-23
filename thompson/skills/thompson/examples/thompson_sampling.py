"""Thompson Sampling on the built-in ads dataset."""

import thompson as th


def main():
    df = th.import_example()
    out = th.thompson(df)

    print(f"Method: {out['methodtype']}")
    print(f"Total reward: {out['total_reward']}")
    print(f"Trials: {len(out['cols_selected'])}")
    print(f"Arms: {list(out['columns'])}")
    print(f"Successes per arm: {out['cols_rewards_1']}")

    th.plot(out)


if __name__ == "__main__":
    main()
