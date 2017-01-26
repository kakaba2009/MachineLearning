library(rEDM)

source('./mylib/mcalc.R')
source('./mylib/mtool.R')
source('./mylib/mfractal.R')

df <- loadSymbol('JPY=X')
df <- df$High
ts <- as.ts(df)
ts <- tail(ts, n=1000)

lib  <- c(1, 900)
pred <- c(901, 1000)

smap_output <- s_map(ts, lib, pred, E = 3)

par(mar = c(4, 4, 1, 1), mgp = c(2.5, 1, 0))

plot(smap_output$theta, smap_output$rho, type = "l", xlab = "Nonlinearity (theta)", 
     ylab = "Forecast Skill (rho)")
