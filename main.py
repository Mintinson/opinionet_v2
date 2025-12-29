from pathlib import Path

def main():
    print("Hello from opinionet-v2!")
    path = Path("models/roberta_large_makeup/20251226_160310/training.log")
    print(path.parent)


if __name__ == "__main__":
    main()

