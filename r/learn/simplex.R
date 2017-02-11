library(data.table)
library(rEDM)

source('./mylib/mcalc.R')
source('./mylib/mtool.R')

options(max.print=5.5E5)

start = "1990-01-01"

#df <- getQuandl("CURRFX/USDJPY", "daily", start, "raw")
#df <- getQuandl("CHRIS/CME_GC1", "daily", start, "raw")
df <- loadSymbol('JPY=X')
#df$Close.Chg <- df$Close - shift(df$Close, n=1)
df <- df$Close
df <- tail(df, 1000)
nr <- NROW(df)

tar <- 975
lib  <- c(1,     tar)
pred <- c(tar+1, nr)

BestE <- BestDimEDM(df, lib, pred, E=2:12)
cat("BestE: ", BestE, "\n")

simplex_output <- simplex(df, lib, pred, E=BestE, tp=1, stats_only=FALSE)
observed  <- simplex_output[[1]]$model_output$obs
predicted <- simplex_output[[1]]$model_output$pred
print(tail(observed,  5))
print(tail(predicted, 5))
par(mar = c(4, 4, 1, 1), pty = "s")
plot_range <- range(c(observed, predicted), na.rm = TRUE)
plot(observed, predicted, xlim = plot_range, ylim = plot_range, xlab = "Observed", ylab = "Predicted")
abline(a = 0, b = 1, lty = 2, col = "blue")