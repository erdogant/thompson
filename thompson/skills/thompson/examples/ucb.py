"""Upper Confidence Bound on the built-in ads dataset."""

import thompson as th


def main():
    df = th.import_example()
    out = th.UCB(df)

    print(f"Method: {out['methodtype']}")
    print(f"Total reward: {out['total_reward']}")
    print(f"Selections per arm: {out['num_selections']}")
    print(f"Sum of rewards per arm: {out['sum_rewards']}")

    th.plot(out)


if __name__ == "__main__":
    main()
