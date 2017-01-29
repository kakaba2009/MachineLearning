library(rEDM)

source('./mylib/mcalc.R')
source('./mylib/mtool.R')

df <- loadSymbol('HKG=X')
df <- df$Close

lib  <- c(1,    2000)
pred <- c(2001, 2500)

simplex_output <- simplex(df, lib, pred, E = 2:12)

par(mar = c(4, 4, 1, 1), mgp = c(2.5, 1, 0))

plot(simplex_output$E, simplex_output$rho, type = "l", xlab = "Embedding Dimension (E)", 
     ylab = "Forecast Skill (rho)")

#simplex_output <- simplex(ts, lib, pred, E = 3, tp = 1:10)
#par(mar = c(4, 4, 1, 1))
#plot(simplex_output$tp, simplex_output$rho, type = "l", xlab = "Time to Prediction (tp)", 
#     ylab = "Forecast Skill (rho)")

bestE <- simplex_output$E[which.max(simplex_output$rho)]

cat("Best Dim:", bestE)