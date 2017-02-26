library(tseriesChaos)
library(data.table)
library(tsDyn)

source("./mylib/mtool.R")

options(max.print=5.5E5)

df1 <- loadSymbol('JPY=X')
df1 <- tail(df1, 500)
x   <- df1$Close

#mutual(lynx)

#lag.plot(x, lags=9, layout=c(3,3))

recurr(x, m=6, d=1) #levels=c(0,0.2,1)
#mod <- aar(x, m=7)
#Summary informations:
#summary(mod)
#Diagnostic plots:
#plot(mod)
#autopairs(x, lag=30, type="lines")
#autotriples(x)
#autotriples(x, type="persp")
#autotriples(x, type="image")