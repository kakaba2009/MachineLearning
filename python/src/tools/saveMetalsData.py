import datetime
import quandl
import pandas_datareader.data as web
from sqlalchemy import create_engine

start = datetime.datetime(1975, 1, 1)
end = datetime.datetime(2017, 12, 30)

Symbols = ['CHRIS/CME_GC1']
Symbols = ['CHRIS/CME_SI1']

for Symbol in Symbols:
    print(Symbol, "\n")
    df = quandl.get(Symbol)

    df.insert(0, "Symbol", Symbol, True)

    print(df, "\n")

    disk_engine = create_engine('sqlite:///../db/Metals.db')

    df.to_sql("Metal", disk_engine, if_exists='append');
