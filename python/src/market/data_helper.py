from numpy import fromstring, resize
from datetime import datetime as dt
import matplotlib.cbook as cbook
import math

def get_two_stock_data():
    """
    load stock time and price data for two stocks The return values
    (d1,p1,d2,p2) are the trade time (in days) and prices for stocks 1
    and 2 (intc and aapl)
    """
    ticker1, ticker2 = 'CSI300', 'APPL'

    file1 = cbook.get_sample_data('CSI300.csv')
    file2 = cbook.get_sample_data('APPL.csv')
    M1 = fromstring(file1.read(), '<d')

    M1 = resize(M1, (M1.shape[0]/2, 2))

    M2 = fromstring(file2.read(), '<d')
    M2 = resize(M2, (M2.shape[0]/2, 2))

    d1, p1 = M1[:, 0], M1[:, 1]
    d2, p2 = M2[:, 0], M2[:, 1]
    return (d1, p1, d2, p2)


def get_daily_data():
    """
    return stock1 and stock2 instances, each of which have attributes

      open, high, low, close, volume

    as numeric arrays

    """
    class C:
        pass

    def get_ticker(ticker):
        dt1  = dt.strptime('2014-6-9', '%Y-%m-%d')
        #vals  = list()
        dates = list()
        close = list()
        ratio = list()

        datafile = cbook.get_sample_data('%s.csv' % ticker, asfileobj=False)

        lines = open(datafile).readlines()
        
        i = 0
        
        for line in lines[1:]:
            row    = [ val for val in line.split(',')[0:] ]
            row[0] = dt.strptime(row[0], '%Y-%m-%d')
            
            if row[0] >= dt1:
                row[1:] = [ (float(val)*7.0) for val in row[1:] ]
    
            try:
                if i % 1 == 0:
                    dates.append( math.log(row[0].timestamp()) )
                    close.append( float(row[4]) )

                    ratio.append( math.log(float(row[4])) )
            except:
                print( row )
                
            i = i + 1

        #M = array(vals)
        #c = C()
        #c.date   = M[:, 0]
        #c.open   = M[:, 1]
        #c.high   = M[:, 2]
        #c.low    = M[:, 3]
        #c.close  = M[:, 4]
        #c.volume = M[:, 5]
        return (dates, ratio)
    
    return get_ticker('appl')
