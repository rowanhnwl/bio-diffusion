from matplotlib import pyplot as plt
import numpy as np

if __name__ == "__main__":

    # Data
    categories = ["Acute Toxicity",
      "Molecular Weight",
      "TPSA",
      "Caco2 Permeability",
      "Lipophilicity",
      "Rotatable Bond Count",
      "Volume Distribution at Steady State",
      "XLogP"]  # X-axis labels
    
    num_bars_per_set = 4  # Each set has 4 bars

    # Sample data (replace with your actual values)
    values = np.array([
        [206, 196, 186, 86],
        [54, 59, 43, 1],
        [175, 174, 180, 44],
        [199, 199, 196, 119],
        [182, 174, 178, 70],
        [202, 209, 205, 209],
        [128, 109, 100, 8],
        [29, 34, 36, 94]
    ]) / 250
    values = np.transpose(values)

    # Bar width
    bar_width = 0.2
    x = np.arange(len(categories))  # X positions for sets

    # Colors and labels for bars in each set
    bar_labels = ["Best", "2nd Best", "3rd Best", "Benchmark"]
    colors = ["blue", "green", "red", "orange"]

    # Plot each bar group
    fig, ax = plt.subplots(figsize=(8, 5))
    for i in range(num_bars_per_set):
        ax.bar(x + i * bar_width, values[i], width=bar_width, label=bar_labels[i], color=colors[i])

    # Formatting
    ax.set_xticks(x + bar_width * (num_bars_per_set / 2 - 0.5))
    ax.set_xticklabels(categories)
    ax.set_ylabel("n (good molecules)")
    ax.set_title("Task arithmetic performance by constraint type")
    ax.legend(title="Hyperparameters")
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    # Show plot
    plt.show()