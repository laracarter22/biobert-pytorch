import sys
import os
from transformers import AutoTokenizer

# Set environment variable
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Path to dataset folder
#dataset_folder = sys.argv[1]
dataset_folder = './lara_preprocessed_conll'

# Model and max length from command-line arguments
model_name_or_path = sys.argv[2]
max_len = int(sys.argv[3])

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
max_len -= tokenizer.num_special_tokens_to_add()

# Iterate through all files in the folder
for filename in os.listdir(dataset_folder):
    file_path = os.path.join(dataset_folder, filename)

    # Ensure we are only processing files
    if not os.path.isfile(file_path):
        continue

    print(f"Processing file: {file_path}")
    subword_len_counter = 0

    with open(file_path, "rt") as f_p:
        for line in f_p:
            line = line.rstrip()

            if not line:
                print(line)
                subword_len_counter = 0
                continue

            token = line.split()[0]

            current_subwords_len = len(tokenizer.tokenize(token))

            # Token contains strange control characters
            # Just filter out the complete line
            if current_subwords_len == 0:
                continue

            if (subword_len_counter + current_subwords_len) > max_len:
                print("")
                print(line)
                subword_len_counter = current_subwords_len
                continue

            subword_len_counter += current_subwords_len

            print(line)
