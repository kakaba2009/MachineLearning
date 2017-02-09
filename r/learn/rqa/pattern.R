library(nonlinearTseries)
library(RANN)
library(xts)
library(zoo)

source('./mylib/mcalc.R')
source('./mylib/mtool.R')
source('./mylib/mfractal.R')

options(max.print=5.5E5)

df <- loadSymbol('JPY=X')
df <- df$Close
ts <- as.ts(df)
ts <- head(ts, n=500)

delta   <- diff(ts)
pattern <- sign(delta)
vol     <- rollapply(ts, 20, sd)

n  <- NROW(ts)
ts <- log(ts[-1]/ts[-n])

d <- calcBestDimension(ts, 1, 0.90)
t <- calcTakens(ts, d, 1)
r <- calcInfoDimension(ts, d, 1)
l <- calcLyapunov(ts, 1, r)

rqa <- calcRQA(t, NULL, d, r)
