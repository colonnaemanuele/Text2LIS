import sys
import subprocess

def main():
    if len(sys.argv) != 2:
        print("Usage: python __main__.py <mode>")
        print("Modes: inference, train")
        sys.exit(1)

    mode = sys.argv[1]

    if mode == "inference":
        subprocess.run(["python", "inference.py"])
    elif mode == "train":
        subprocess.run(["python", "train.py"])
    else:
        print("Invalid mode. Please choose 'inference' or 'train'.")
        sys.exit(1)

if __name__ == "__main__":
    main()