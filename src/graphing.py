import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# This program extracts the desired variables from the output of training the model to avoid doing things by hand.
# Paste the output into a file starting at the first iteration, and supply the filename in main
# Then the extract method will fill lists of the values of each desired training statistic
# Then you just need to modify the plotting code to customize the plots


def extract(
    filename,
    approx_kl,
    entropy_loss,
    explained_variance,
    loss,
    policy_gradient_loss,
    value_loss,
    iterations,
):
    f = open(filename, "r")
    lines = f.readlines()
    for i in range(39):
        if i > 18 and iterations == 19:
            approx_kl.append(0)
            entropy_loss.append(0)
            explained_variance.append(0)
            loss.append(0)
            policy_gradient_loss.append(0)
            value_loss.append(0)
        else:
            # Add values to list
            addition = (18 * i) + 7  # New line calculation between iterations
            print("addition" + str(addition))
            print("Iters" + str(iterations))
            print("i = " + str(i))
            print(lines[7 + addition])
            approx_kl_curr = float(re.findall("\d+\.\d+", lines[7 + addition])[0])
            nums = re.findall("0\d+", lines[7 + addition])
            multiplier = int(str(nums[0])[1])
            approx_kl_curr = approx_kl_curr * 10**-multiplier
            approx_kl.append(approx_kl_curr)
            entropy_loss.append(
                float(re.findall("-?\d+(?:\.\d+)?", lines[10 + addition])[0])
            )
            explained_variance.append(
                float(re.findall("-?\d+(?:\.\d+)?", lines[11 + addition])[0])
            )
            loss.append(float(re.findall("-?\d+(?:\.\d+)?", lines[13 + addition])[0]))
            policy_gradient_loss.append(
                float(re.findall("-?\d+(?:\.\d+)?", lines[15 + addition])[0])
            )
            value_loss.append(
                float(re.findall("-?\d+(?:\.\d+)?", lines[16 + addition])[0])
            )


def plot(x_data, y_data, y_data2, label1, label2, x_label, y_label, title):
    fig, ax = plt.subplots()
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.plot(x_data, y_data, marker="o", label=label1, drawstyle="steps-post")
    ax.plot(x_data, y_data2, marker="o", label=label2, drawstyle="steps-post")
    ax.legend()
    plt.savefig(f"../archive/plots/{title}.png")
    plt.show()


def main():
    iterations = [i for i in range(1, 40)]
    # Supply filenames
    (
        filename,
        approx_kl,
        entropy_loss,
        explained_variance,
        loss,
        policy_gradient_loss,
        value_loss,
    ) = ("../run_logs/rundata1.txt", [], [], [], [], [], [])
    (
        filename2,
        approx_kl2,
        entropy_loss2,
        explained_variance2,
        loss2,
        policy_gradient_loss2,
        value_loss2,
    ) = ("../run_logs/rundata2.txt", [], [], [], [], [], [])
    # Extract data
    extract(
        filename,
        approx_kl,
        entropy_loss,
        explained_variance,
        loss,
        policy_gradient_loss,
        value_loss,
        19,
    )
    extract(
        filename2,
        approx_kl2,
        entropy_loss2,
        explained_variance2,
        loss2,
        policy_gradient_loss2,
        value_loss2,
        19,
    )

    # Plot data
    # This could be much shorter and better code if it was in a function, but alas there are only so many hours in the day.
    # approx_kl
    plot(
        iterations,
        approx_kl,
        approx_kl2,
        "10k Timesteps",
        "20k Timesteps",
        "Iteration",
        "Approx_kl",
        "Approx_kl",
    )
    plot(
        iterations,
        entropy_loss,
        entropy_loss2,
        "10k Timesteps",
        "20k Timesteps",
        "Iteration",
        "Entropy loss",
        "Entropy loss",
    )
    plot(
        iterations,
        explained_variance,
        explained_variance2,
        "10k Timesteps",
        "20k Timesteps",
        "Iteration",
        "Explained Variance",
        "Explained Variance",
    )
    plot(
        iterations,
        loss,
        loss2,
        "10k Timesteps",
        "20k Timesteps",
        "Iteration",
        "loss",
        "loss2",
    )
    plot(
        iterations,
        policy_gradient_loss,
        policy_gradient_loss2,
        "10k Timesteps",
        "20k Timesteps",
        "Iteration",
        "Policy gradient loss",
        "Policy gradient loss",
    )
    plot(
        iterations,
        value_loss,
        value_loss2,
        "10k Timesteps",
        "20k Timesteps",
        "Iteration",
        "Value Loss",
        "Value Loss2",
    )


if __name__ == "__main__":
    main()
