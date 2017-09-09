library(TTR)

source('./mylib/mcalc.R')
source('./mylib/mtool.R')

options(max.print=5.5E5)

start = as.Date("1990-01-01")
end   = as.Date("2017-12-31")

df <- getQuandl("CUR/JPY", "daily", start, end, "raw")
df <- df[, c("RATE", "RATE")]

sma <- data.frame(apply(df, 2, SMA, n=314))
ema <- data.frame(apply(df, 2, EMA, n=314))

plot(sma$RATE)
