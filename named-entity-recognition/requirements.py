import subprocess
import sys

#I need transformers, datasets, seqeval, 

# Install the transformers package
try:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers"])
    print("Successfully installed 'transformers'!")
except subprocess.CalledProcessError as e:
    print(f"An error occurred while installing 'transformers': {e}")

try:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "datasets"])
    print("Successfully installed 'datasets'!")
except subprocess.CalledProcessError as e:
    print(f"An error occurred while installing 'datasets': {e}")

try:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "seqeval"])
    print("Successfully installed 'seqeval'!")
except subprocess.CalledProcessError as e:
    print(f"An error occurred while installing 'seqeval': {e}")
