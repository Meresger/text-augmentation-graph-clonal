import os
import pandas as pd
import clonalg as model

def process_row(row):
    try:
        result = model.get_clone_text_no_iter(row['text'])
        return result
    except Exception as e:
        print("Error processing row: ", row)
        return []

def process_csv(input_file):
    data = pd.read_csv(input_file)
    processed_rows = []
    for _, row in data.iterrows():
        processed_rows.extend(process_row(row))
    output_file = os.path.splitext(input_file)[0] + "_augmented.csv"
    data_augmented = pd.DataFrame()
    data_augmented['text'] = processed_rows
    data_augmented.to_csv(output_file, index=False)

input_file = "../data/your_file.csv"  # Specify the path to your CSV file
process_csv(input_file)
