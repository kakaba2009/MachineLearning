library(data.table)
library(dplyr)

source('./mylib/mcalc.R')
source('./mylib/mtool.R')

start = "2013-04-01"

r <- 31.1035 

jpy <- getQuandl("CUR/JPY", "daily", start, "raw")
gld <- getQuandl("CHRIS/CME_GC1", "daily", start, "raw")

setDT(jpy)
setDT(gld)

com <- inner_join(gld, jpy, by=c("Date" = "DATE"))

com$jpn <- com$Settle * com$RATE * r * 1.08

com <- tail(com, 23)

ord <- com[order(com$jpn),]

big <- max(com$jpn)

big
