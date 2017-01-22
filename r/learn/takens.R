library(nonlinearTseries)

source('./lib/mcalc.R')
source('./lib/mtool.R')
source('./lib/mfractal.R')

#saveCSV('JPY=X')

x <- loadSymbol('JPY=X')
x <- loadTimeSeries('JPY=X', 'High')
#x <- as.ts(forex['x'])
#x <- x[1:1000,]

#lag <- calcTimeLag(x)

#calcLyapunov(x, 30)
#d <- BoxCountDim(x)
#cd <- CorrDim(x, d1=5, d2=9, r1=0.05, r2=1, lag=1)
#e  <- SampleEntropy(cd, d1=6, d2=9, r1=0.1, r2=1.0)
d <- calcBestDimension(x, 1, 0.90)
t <- calcTakens(x, d, 1)
r <- calcInfoDimension(x, 7, 1)

calcRecurrPlot(t, x, d, 1, r*2.0)
calcRQA(t, x, d, r*2.0)
#AdditiveNonlinearAutoregressive(x, 7)
#SpaceTimePlot(t, d, 1000)
#n = NeuralNetworkNonlinearAutoregressiveModel(x, d, 1, 1, 8)
#f <- FindDominantFrequency(log(x))