library(nonlinearTseries)
library(xts)

source('./mylib/mcalc.R')
source('./mylib/mtool.R')
source('./mylib/mfractal.R')

saveCSV('JPY=X')

df <- loadSymbol('JPY=X')
df <- df$High
ts <- as.ts(df)
ts <- tail(ts, n=1000)
#to.weekly(ts)

#lag <- calcTimeLag(ts)

calcLyapunov(ts, 30)
d <- BoxCountDim(ts)
cd <- CorrDim(ts, d1=5, d2=9, r1=0.05, r2=1, lag=1)
e  <- SampleEntropy(cd, d1=6, d2=9, r1=0.1, r2=1.0)
d <- calcBestDimension(ts, 1, 0.90)
t <- calcTakens(ts, d, 1)
r <- calcInfoDimension(ts, 7, 1)

#calcRecurrPlot(t, ts, d, 1, r*2.0)
#calcRQA(t, ts, d, r*2.0)
#AdditiveNonlinearAutoregressive(ts, 7)
#SpaceTimePlot(t, d, 1000)
#n = NeuralNetworkNonlinearAutoregressiveModel(ts, d, 1, 1, 8)
#f <- FindDominantFrequency(log(ts))