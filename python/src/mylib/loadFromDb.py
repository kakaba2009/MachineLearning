import pandas as pd
import datetime as dt
from sqlalchemy import create_engine

def loadFromDb(db):
    query = """
            SELECT Symbol, Open, High, Low, Close, Volume, 'Adj Close' 
            FROM NYSE
            WHERE Symbol = 'XOM'
            """

    start = dt.datetime.now()

    disk_engine = create_engine('sqlite:///' + db)
    
    df = pd.read_sql_query(query, disk_engine)

    return df