import re
import pathlib
import os

def main():
    dir_path = pathlib.Path(os.path.dirname(os.path.realpath(__file__)))
    for file in dir_path.glob("*.txt"):
        with open(file, 'r') as f:
            numbers = re.findall("\d+\.\d+", "".join(f.readlines()))
            numbers = list(map(lambda s: float(s), numbers))
            max_percent = max(numbers)
            print(f"{file}: {max_percent}, train_mps_thread_percent={((1 - max_percent) * 100):.2f}")

if __name__ == "__main__":
    main()

