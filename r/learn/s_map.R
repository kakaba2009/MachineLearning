library(rEDM)

source('./mylib/mcalc.R')
source('./mylib/mtool.R')

df <- loadSymbol('JPY=X')
df <- df$Close

lib  <- c(1, 3000)
pred <- c(3001, 3200)

smap_output <- s_map(df, lib, pred, E = 9)

par(mar = c(4, 4, 1, 1), mgp = c(2.5, 1, 0))

plot(smap_output$theta, smap_output$rho, type = "l", xlab = "Nonlinearity (theta)", 
     ylab = "Forecast Skill (rho)")
