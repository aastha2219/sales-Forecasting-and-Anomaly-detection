import pandas as pd

encodings = ['utf-8', 'latin-1', 'windows-1252', 'utf-16']
for enc in encodings:
    try:
        df = pd.read_csv('train.csv', encoding=enc)
        print(f"Success with encoding: {enc}")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        break
    except Exception as e:
        print(f"Failed with {enc}: {e}")
