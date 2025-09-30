import pandas as pd

def load_data(file_path):
    """Load all sheets from Excel file and parse time column."""
    xls = pd.ExcelFile(file_path)
    dfs = {}
    for s in xls.sheet_names:
        df = pd.read_excel(file_path, sheet_name=s)
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'], errors='coerce')
            df = df.dropna(subset=['time']).set_index('time').sort_index()
        dfs[s] = df
    return dfs
