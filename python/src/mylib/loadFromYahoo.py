import datetime
import pandas as pd
import pandas_datareader.data as web
from sqlalchemy import create_engine

Symbols = ["ARS=X","RUB=X","SAR=X","ZAR=X","TRY=X"]

#Symbol = "JPY=X"

def loadFromYahoo(Symbol):
    start = datetime.datetime(1900, 1, 1)
    end = datetime.datetime(2016, 9, 30)

    df = web.get_data_yahoo(Symbol, start, end)

    df.insert(0, "Symbol", Symbol, True)

    df.pct_change(periods=1, fill_method='pad')

    #df.to_csv("market.csv")
    
    return df
