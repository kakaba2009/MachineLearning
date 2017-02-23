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

## calculate the scaling exponent for a ts
DFA.walk <- DFA(ts, detrend="poly1", sum.order=1, verbose=TRUE)
## print the results
print(DFA.walk)
## plot a summary of the results
eda.plot(DFA.walk)