import sys
import subprocess
import lovely_tensors as lt
lt.monkey_patch()
sys.path.append('text2lis/model')
sys.path.append('text2lis/data')

def main():
    if len(sys.argv) != 2:
        print("Usage: python __main__.py <mode>")
        print("Modes: inference, train")
        sys.exit(1)

    mode = sys.argv[1]

    if mode == "inference":
        subprocess.run(["/home/emanuele/miniconda3/envs/text2lis/bin/python", "inference.py"])
    elif mode == "train":
        subprocess.run(["/home/emanuele/miniconda3/envs/text2lis/bin/python", "train.py"])
    else:
        print("Invalid mode. Please choose 'inference' or 'train'.")
        sys.exit(1)

if __name__ == "__main__":
    main()