library(data.table)
library(dplyr)

source('./mylib/mcalc.R')
source('./mylib/mtool.R')

start = "2016-02-10"
end   = "2016-02-28"

r <- 31.1035 

jpy <- getQuandl("CURRFX/USDJPY", "daily", start, end, "raw")
gld <- getQuandl("CHRIS/CME_GC1", "daily", start, end, "raw")

setDT(jpy)
setDT(gld)

com <- inner_join(gld, jpy, by=c("Date" = "DATE"))

com$jpn <- com$Settle * com$RATE * r * 1.08

com <- tail(com, 100)

ord <- com[order(com$jpn),]

big <- max(com$jpn)

big
