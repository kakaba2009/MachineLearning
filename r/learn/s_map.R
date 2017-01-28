library(rEDM)

source('./mylib/mcalc.R')
source('./mylib/mtool.R')
source('./mylib/mfractal.R')

df <- loadSymbol('JPY=X')
df <- df$Close
ts <- as.ts(df)
ts <- tail(ts, n=5200)

lib  <- c(1, 5000)
pred <- c(5001, 5200)

smap_output <- s_map(ts, lib, pred, E = 5)

par(mar = c(4, 4, 1, 1), mgp = c(2.5, 1, 0))

plot(smap_output$theta, smap_output$rho, type = "l", xlab = "Nonlinearity (theta)", 
     ylab = "Forecast Skill (rho)")
