import numpy as np
import os


def get_embedding_size(file_path):
    try:
        with open(file_path, "r") as f:
            floats = [float(x) for x in f.read().strip().split()]
        return len(floats)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return 0


# Example paths
ada_file = "dataset/ghostbuster/wp/gpt/1-ada.txt"
davinci_file = "dataset/ghostbuster/wp/gpt/1-davinci.txt"
combined_ada_file = "dataset/ghostbuster/wp/gpt/combined-ada.txt"
combined_davinci_file = "dataset/ghostbuster/wp/gpt/combined-davinci.txt"


print(f"Ada embedding size: {get_embedding_size(ada_file)}")
print(f"Davinci embedding size: {get_embedding_size(davinci_file)}")
print(f"Combined Ada size: {get_embedding_size(combined_ada_file)}")
print(f"Combined Davinci size: {get_embedding_size(combined_davinci_file)}")

gpt_folder = "dataset/ghostbuster/wp/gpt"
ada_files = [
    f
    for f in os.listdir(gpt_folder)
    if f.endswith("-ada.txt") and not f.startswith("combined")
]
print(f"Number of ada files: {len(ada_files)}")

with open("dataset/ghostbuster/wp/gpt/combined-ada.txt", "r") as f:
    floats = [float(x) for x in f.read().strip().split()]
    print(f"Total floats: {len(floats)}, Vectors: {len(floats) // 833}")
