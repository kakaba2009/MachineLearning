library(e1071)
library(xts)

source('./mylib/mcalc.R')
source('./mylib/mtool.R')

options(max.print=5.5E5)

df <- loadSymbol('JPY=X')
df <- df$Close
ts <- as.ts(df)

x <- tail(ts, n=1000)
y <- stft(x, win=6, inc=1, coef=64)
plot(y)