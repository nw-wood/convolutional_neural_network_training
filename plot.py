import os                           # Provides functions to interact with the operating system.
import re                           # Module for working with regular expressions (pattern matching).
import matplotlib.pyplot as plt     # Library for creating static, animated, and interactive visualizations in Python.
import numpy as np                  # Fundamental package for numerical computations and array manipulations.

# Directory structure
base_dir = os.path.join(os.path.dirname(__file__), "evaluations")
graphs_dir = os.path.join(os.path.dirname(__file__), "graphs")

# Ensure the graphs directory exists
os.makedirs(graphs_dir, exist_ok=True)

# Data storage
data = []

# Traverse directories and extract data
for root, dirs, files in os.walk(base_dir):
    for file in files:
        if file == "model_1_accuracy.txt":
            batch_size_match = re.search(r"batch_size_(\d+)", root)
            epochs_match = re.search(r"epochs_(\d+)", root)
            if batch_size_match and epochs_match:
                batch_size = int(batch_size_match.group(1))
                epochs = int(epochs_match.group(1))
                with open(os.path.join(root, file), 'r') as f:
                    content = f.read()
                    accuracy_match = re.search(r"model_1\.pt:\s*(\d+\.\d+)%", content)
                    if accuracy_match:
                        accuracy = float(accuracy_match.group(1))
                        data.append((epochs, batch_size, accuracy))
                    else:
                        print(f"No accuracy found in {os.path.join(root, file)}")
            else:
                print(f"Failed to match batch size or epochs in path: {root}")

# Organize data
data = sorted(data, key=lambda x: (x[0], x[1]))

# Prepare for plotting
epochs_set = sorted(set(d[0] for d in data))
batch_sizes_set = sorted(set(d[1] for d in data))

# Create heatmap of accuracy
heatmap_data = np.zeros((len(epochs_set), len(batch_sizes_set)))
for epochs, batch_size, accuracy in data:
    i = epochs_set.index(epochs)
    j = batch_sizes_set.index(batch_size)
    heatmap_data[i, j] = accuracy

# Debugging outputs
print("Epochs:", epochs_set)
print("Batch Sizes:", batch_sizes_set)
print("Heatmap Data:")
print(heatmap_data)

# Plotting heatmap
fig, ax = plt.subplots(figsize=(10, 8))
cax = ax.imshow(heatmap_data, cmap="viridis", aspect="auto")

# Add labels
ax.set_xticks(range(len(batch_sizes_set)))
ax.set_yticks(range(len(epochs_set)))
ax.set_xticklabels(batch_sizes_set)
ax.set_yticklabels(epochs_set)
ax.set_xlabel("Batch Size")
ax.set_ylabel("Epochs")
ax.set_title("Model Accuracy (%)")

# Add colorbar
fig.colorbar(cax, label="Accuracy (%)")

# Annotate heatmap
for i in range(len(epochs_set)):
    for j in range(len(batch_sizes_set)):
        ax.text(j, i, f"{heatmap_data[i, j]:.2f}", ha="center", va="center", color="white")

plt.tight_layout()
heatmap_path = os.path.join(graphs_dir, "heatmap_accuracy.png")
plt.savefig(heatmap_path)
plt.close()

# Line plot for each batch size
plt.figure(figsize=(10, 6))
for batch_size in batch_sizes_set:
    accuracies = [d[2] for d in data if d[1] == batch_size]
    plt.plot(epochs_set, accuracies, marker='o', label=f"Batch Size {batch_size}")

plt.xlabel("Epochs")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy vs. Epochs for Different Batch Sizes")
plt.legend()
plt.grid()
plt.tight_layout()
line_plot_path = os.path.join(graphs_dir, "line_plot_accuracy.png")
plt.savefig(line_plot_path)
plt.close()
