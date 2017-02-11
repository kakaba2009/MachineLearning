library(data.table)
library(rEDM)

source('./mylib/mcalc.R')
source('./mylib/mtool.R')

start = "1990-01-01"

#df <- getQuandl("CURRFX/USDJPY", "daily", start, "raw")
#df <- getQuandl("CHRIS/CME_GC1", "daily", start, "raw")
df <- loadSymbol('JPY=X')
#df$Close.Chg <- log(df$Close / shift(df$Close, n=1))
df <- df$Close
df <- tail(df, 5000)

tar <- 4500
lib  <- c(1,     tar)
pred <- c(tar+1, 5000)

BestE <- BestDimEDM(df, lib, pred, E=2:12)
cat("BestE: ", BestE, "\n")

simplex_output <- simplex(df, lib, pred, E=BestE, tp=1, stats_only=FALSE)
observed  <- simplex_output[[1]]$model_output$obs
predicted <- simplex_output[[1]]$model_output$pred
print(tail(observed,  5))
print(tail(predicted, 5))
