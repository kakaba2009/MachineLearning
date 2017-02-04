library(rEDM)

source('./mylib/mcalc.R')
source('./mylib/mtool.R')

start = "1990-01-01"

#df <- getQuandl("CURRFX/USDJPY", "daily", start, "raw")
#df <- getQuandl("CHRIS/CME_GC1", "daily", start, "raw")
df <- loadSymbol('JPY=X')
df <- df[ ,c('Close')]
df <- tail(df, 500)

lib  <- c(1, 500)
pred <- c(1, 500)

BestE <- BestDimEDM(df, lib, pred)
cat("BestE: ", BestE, "\n")

lib  <- c(1, 500)
pred <- c(1, 501)
simplex_output <- simplex(df, lib, pred, E=BestE, tp=1, stats_only=FALSE)
observed  <- simplex_output[[1]]$model_output$obs
predicted <- simplex_output[[1]]$model_output$pred
print(tail(observed,  10))
print(tail(predicted, 10))