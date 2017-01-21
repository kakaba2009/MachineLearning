library(tseriesChaos)
library(data.table)
library(tsDyn)

source("./lib/mtool.R")

data(forex)

x <- as.ts(forex['x'])
x <- x[1:200,]

#mutual(lynx)

#lag.plot(x, lags=9, layout=c(3,3))

recurr(x, m=7, d=1) #levels=c(0,0.2,1)
#mod <- aar(x, m=7)
#Summary informations:
#summary(mod)
#Diagnostic plots:
#plot(mod)
#autopairs(x, lag=30, type="lines")
#autotriples(x)
#autotriples(x, type="persp")
#autotriples(x, type="image")
#fit a Neural Network model
#mod.nnet <- nnetTs(x, m=7, d=1, steps=1, size=16)
#mod.nnet
