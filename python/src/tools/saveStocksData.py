import datetime
import pandas as pd
import pandas_datareader.data as web
from sqlalchemy import create_engine
from src.mylib.fetchSymbolList import getSymbolList

start = datetime.datetime(1975, 1, 1)
end = datetime.datetime(2017, 12, 30)

#Symbols = getSymbolList()
Symbols = {'1': ['AAPL', 'IBM']}

Keys = Symbols.keys()
Values = Symbols.values()

for Sector in Values:
    for Symbol in Sector:
        print(Symbol, "\n")
        df = web.get_data_yahoo(Symbol, start, end)

        df.insert(0, "Symbol", Symbol, True)

        print(df, "\n")

        disk_engine = create_engine('sqlite:///../db/Dow.db')

        df.to_sql("Stocks", disk_engine, if_exists='append');
