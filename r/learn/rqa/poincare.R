library(nonlinearTseries)
library(fractal)
library(xts)

source('./mylib/mcalc.R')
source('./mylib/mtool.R')
source('./mylib/mfractal.R')

options(max.print=5.5E5)

df <- loadSymbol('JPY=X')
df <- df$Close
ts <- as.ts(df)
ts <- head(ts, n=5000)

x <- poincareMap(ts, extrema="max")
y <- embedSeries(x$amplitude, tlag=10, dimension=2)
plot(y, pch=1, cex=1)
plot(x$location, x$amplitude)