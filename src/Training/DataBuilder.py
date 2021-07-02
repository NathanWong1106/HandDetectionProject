import os
import pandas as pd

DATA_DIR_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), "Data")
OUTPUT_FILE_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), "ConcatData", "concat.csv")

def main():
    concat = pd.DataFrame()

    print("Concat:")
    for csv in os.listdir(DATA_DIR_PATH):
        print(csv)
        concat = concat.append(pd.read_csv(os.path.join(DATA_DIR_PATH, csv)))

    concat.to_csv(OUTPUT_FILE_PATH, index=False)
    print(f"Saved to: {OUTPUT_FILE_PATH}")

if __name__ == "__main__":
    main()