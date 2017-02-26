library(dtw)

source('./R/multiview_interface.R')
source('./mylib/mcalc.R')
source('./mylib/mtool.R')

options(max.print=5.5E5)

df1 <- loadSymbol('JPY=X')
df1 <- tail(df1, 3000)
df2 <- loadSymbol('EUR=X')
df2 <- tail(df2, 3000)

query    <- df1$Close
query    <- ComputeChange(query)
reference<- df2$Close
reference<- ComputeChange(reference)

plot(reference); lines(query,col="blue")
## Find the best match
alignment<-dtw(query,reference,keep=TRUE,step=asymmetric,open.end=TRUE,open.begin=TRUE);
## Display the mapping, AKA warping function - may be multiple-valued
## Equivalent to: plot(alignment,type="alignment")
plot(alignment,type="two",off=1)
