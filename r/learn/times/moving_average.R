library(data.table)
library(TTR)

source('./mylib/mcalc.R')
source('./mylib/mtool.R')

options(max.print=5.5E5)

df <- loadSymbol('JPY=X')
df <- df[, c("Open", "Close")]

sma <- data.frame(apply(df, 2, SMA, n=314))
ema <- data.frame(apply(df, 2, EMA, n=314))

plot(sma$Close)
