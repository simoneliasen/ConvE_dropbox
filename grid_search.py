import argparse
import itertools as it
import math
import pathlib as pl


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("dataset", choices=["wn18rr", "fb15k237"])
    parser.add_argument("output")
    parser.add_argument("--dimensions", type=int, nargs="*", default=[50])
    parser.add_argument("--learning-rate", type=float, nargs="*", default=[0.1])
    parser.add_argument("--batch-size", type=int, nargs="*", default=[64])
    parser.add_argument("--epochs", type=int, nargs="*", default=[500])
    parser.add_argument("--input-dropout", type=float, nargs="*", default=[0])
    parser.add_argument("--feature-dropout", type=float, nargs="*", default=[0])
    parser.add_argument("--hidden-dropout", type=float, nargs="*", default=[0])
    parser.add_argument("--smoothing", type=float, nargs="*", default=[0])

    args = parser.parse_args()

    filtered_args = {
        key: value
        for key, value in vars(args).items()
        if value is not None and key not in ["output", "dataset"]
    }

    keys, values = zip(*sorted(filtered_args.items()))

    output = pl.Path(args.output)
    output.mkdir(exist_ok=True)

    if args.dataset == "wn18rr":
        dataset = "WN18RR"
    elif args.dataset == "fb15k237":
        dataset = "FB15k-237"

    for value in it.product(*values):
        config = dict(zip(keys, value))

        config_key = "-".join(
            ["{}-{}".format(key, value) for key, value in config.items()]
        )

        script_file = output / f"{config_key}.sh"
        output_file = output / f"{config_key}.out"

        arguments = {
            "model": "conve",
            "data": dataset,
            "batch-size": config["batch_size"],
            "lr": config["learning_rate"],
            "embedding-dim": config["dimensions"],
            "embedding-shape1": 16,
            "hidden-drop": config["hidden_dropout"],
            "input-drop": config["input_dropout"],
            "feat-drop": config["feature_dropout"],
            "label-smoothing": config["smoothing"],
        }

        argument_string = " ".join(
            [f"--{key} {value}" for key, value in arguments.items()]
        )

        with open(script_file, "w") as file:
            file.write(
                f"(time CUDA_VISIBLE_DEVICES=0 python main.py {argument_string}) &> {output_file}"
            )


if __name__ == "__main__":
    main()
