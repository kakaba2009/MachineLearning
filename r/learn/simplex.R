library(rEDM)

source('./mylib/mcalc.R')
source('./mylib/mtool.R')

start = "1990-01-01"

df <- getQuandl("CURRFX/USDJPY", "daily", start)
df <- getQuandl("CHRIS/CME_GC1", "daily", start, "raw")
#df <- loadSymbol('JPY=X')
#df <- df$Close

lib  <- c(1,    2000)
pred <- c(2001, 2500)

simplex_output <- simplex(df, lib, pred, E = 2:12)

par(mar = c(4, 4, 1, 1), mgp = c(2.5, 1, 0))

plot(simplex_output$E, simplex_output$rho, type = "l", xlab = "Embedding Dimension (E)", 
     ylab = "Forecast Skill (rho)")

bestE <- simplex_output$E[which.max(simplex_output$rho)]

cat("Best Dim:", bestE)
