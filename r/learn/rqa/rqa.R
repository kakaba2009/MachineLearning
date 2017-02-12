library(nonlinearTseries)
library(RANN)
library(xts)

source('./mylib/mcalc.R')
source('./mylib/mtool.R')
source('./mylib/mfractal.R')

options(max.print=5.5E5)

df <- loadSymbol('JPY=X')
df$Time <- ComputeTimeDiff2(df)

df <- df$Close
ts <- as.ts(df)
ts <- head(ts, n=500)

d <- calcBestDimension(ts, 1, 0.90)
t <- calcTakens(ts, d, 1)
m <- MinDistance(df)
r <- calcInfoDimension(ts, d, 1)
l <- calcLyapunov(ts, 1, r)

rqa <- calcRQA(t, NULL, d, r)