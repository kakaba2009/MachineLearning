library(rEDM)

source('./mylib/mcalc.R')
source('./mylib/mtool.R')

df <- loadSymbol('JPY=X')
df <- df$Close

lib  <- c(1,    2000)
pred <- c(2001, 2500)

bestE <- BestDimEDM(df, lib, pred)

cat("Best Dim:", bestE)

simplex_output <- simplex(df, lib, pred, E = bestE, tp = 1:10)

par(mar = c(4, 4, 1, 1))

plot(simplex_output$tp, simplex_output$rho, type = "l", xlab = "Time to Prediction (tp)", 
     ylab = "Forecast Skill (rho)")
