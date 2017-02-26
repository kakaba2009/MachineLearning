library(fractal)

source('./R/multiview_interface.R')
source('./mylib/mcalc.R')
source('./mylib/mtool.R')

options(max.print=5.5E5)

df1 <- loadSymbol('JPY=X')
df1 <- tail(df1, 3000)
df1 <- df1$Close

beam.det <- determinism(df1, dimension=8, olag=1)
print(beam.det)
plot(beam.det)