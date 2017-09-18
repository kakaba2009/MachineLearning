import pandas as pd
from sqlalchemy import create_engine
from PyQt5.Qt import Qt

def loadCSV(symbol):
    filename = "../db/" + symbol + '.csv'
    
    df = pd.read_csv(filename, index_col=False)
    
    return df
    
def loadData(symbol, path):
    con = create_engine('sqlite:///' + path)

    query = 'SELECT Date, Open, High, Low, Close, Volume FROM FX WHERE Symbol = ' + "'" + symbol + "'"

    df = pd.read_sql_query(query, con, index_col ="Date", parse_dates=["Date"])
    
    #df = df.to_period(freq ="D")

    df = df.drop("Volume", axis=1)
    
    return df

def loadStock(symbol, path="Dow.db"):
    con = create_engine('sqlite:///' + path)

    query1 = 'SELECT Date, Open, High, Low, Close, Volume, "Adj Close" FROM NYSE WHERE Symbol = ' + "'" + symbol + "'"

    df = pd.read_sql_query(query1, con, index_col ="Date", parse_dates=["Date"])
    
    #df = df.to_period(freq ="D")

    return df

def loadSymbolList(path, db = 'FX'):
    con = create_engine('sqlite:///' + path)

    query ="""SELECT DISTINCT Symbol FROM """ + db

    df = pd.read_sql_query(query, con)
    
    return df

def loadAll(path, db = 'FX', dropDate=False):
    con = create_engine('sqlite:///' + path)

    query = 'SELECT Symbol, Date, Open, High, Low, Close FROM ' + db

    df = pd.read_sql_query(query, con, index_col ="Date", parse_dates=["Date"])
    
    #df = df.to_period(freq ="D")

    if(dropDate == True):
        df = df.reset_index()
    
    print(df.info())
    
    return df

def loadAndAppend():
    dfs = []

    ls = loadSymbolList("db/forex.db")

    for s in ls.Symbol:
        df = loadData(s, "db/forex.db")
        df = df.drop("Volume", axis=1)
        dfs.append(df)

    return dfs

def loadOneSymbol(symbol, db="db/forex.db", sub = 'FX'):
    dfs = loadAll(db, sub, dropDate=False)
    dfs = dfs.groupby("Symbol")
    df  = dfs.get_group(symbol).drop("Symbol", axis=1)
    
    return df


def loadClose(symbol, path):
    con = create_engine('sqlite:///' + path)

    query = 'SELECT Date, Close FROM FX WHERE Symbol = ' + "'" + symbol + "'"

    df = pd.read_sql_query(query, con, index_col="Date", parse_dates=["Date"])

    # df = df.to_period(freq ="D")

    #df = df.drop("Volume", axis=1)

    return df
