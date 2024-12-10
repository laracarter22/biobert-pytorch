import subprocess
import sys

# Install the transformers package
try:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers"])
    print("Successfully installed 'transformers'!")
except subprocess.CalledProcessError as e:
    print(f"An error occurred while installing 'transformers': {e}")


