import pandas as pd
import glob

def getAllCsvType(type):
    path = r'data/daily'
    file = '/{}*.csv'.format(type)
    all_files = glob.glob(path + file)

    li = []

    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, header=0)
        li.append(df)

    frame = pd.concat(li, axis=0, ignore_index=True)
    return frame
