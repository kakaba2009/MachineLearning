source('./mylib/mcalc.R')
source('./mylib/mtool.R')

start = "2013-04-01"

#download("WIKI")

#df <- search("HKD", "")
df <- getQuandl("CHRIS/CME_GC1", "daily", start, "raw")

df <- getQuandl("CUR/JPY", "daily", start, "raw")

df <- getQuandl("CUR/CNY", "daily", start, "raw")

df <- getQuandl("YAHOO/GOOG", "monthly", start, "raw")

df <- getQuandl("NSE/OIL", "monthly", start, "raw")

df <- getQuandl("CURRFX/USDJPY", "monthly", start, "raw")

df <- getQuandl("CHRIS/CME_GC1", "monthly", start, "raw")

df <- getQuandl("YAHOO/GOOG", "monthly", start, "raw")
