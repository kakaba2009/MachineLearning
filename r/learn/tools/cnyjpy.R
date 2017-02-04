source('./mylib/mcalc.R')
source('./mylib/mtool.R')

start = "2016-01-01"

jpy <- getQuandl("CUR/JPY", "daily", start, "raw")

cny <- getQuandl("CUR/CNY", "daily", start, "raw")

con <- data.frame(Date=jpy$DATE)

con$RATE <- jpy$RATE / cny$RATE

ord <- con[order(con$RATE),]

big <- max(con$RATE)

big

tail(ord, 3)
