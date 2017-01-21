import datetime
import pandas as pd
import pandas_datareader.data as web

from sqlalchemy import create_engine

import matplotlib.pyplot as plt
from matplotlib import style

from src.lib.fetchSymbolList import getSymbolList

Symbols = getSymbolList()

Keys = Symbols.keys()
Values = Symbols.values()

for Sector in Values:
    for Symbol in Sector:
        print(Symbol, "\n")
        start = datetime.datetime(1900, 1, 1)
        end = datetime.datetime(2016, 9, 30)

        df = web.get_data_yahoo(Symbol, start, end)

        df.insert(0, "Symbol", Symbol, True)

        print(df, "\n")

        disk_engine = create_engine('sqlite:///snp500.db')

        df.to_sql("Stocks", disk_engine, if_exists='append');
