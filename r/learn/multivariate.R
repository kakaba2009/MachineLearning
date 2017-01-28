library(rEDM)
library(data.table)
library(dplyr)

source('./mylib/mcalc.R')
source('./mylib/mtool.R')

JPY <- loadSymbol('JPY=X')
JPY <- tail(JPY, 1000)

EUR <- loadSymbol('EUR=X')
EUR <- tail(EUR, 1000)

setDT(JPY)
setDT(EUR)

df <- left_join(JPY, EUR, by=c("Date"))

lib  <- c(1, NROW(df))
pred <- c(1, NROW(df))

block_lnlp_output <- block_lnlp(df, lib = lib, pred = pred, columns = c("Close.x","Close.y"), 
                        target_column = "Close.x", stats_only = FALSE, first_column_time = TRUE)

observed  <- block_lnlp_output[[1]]$model_output$obs
predicted <- block_lnlp_output[[1]]$model_output$pred

par(mar = c(4, 4, 1, 1), pty = "s")
plot_range <- range(c(observed, predicted), na.rm = TRUE)
plot(observed, predicted, xlim = plot_range, ylim = plot_range, xlab = "Observed", ylab = "Predicted")

abline(a = 0, b = 1, lty = 2, col = "blue")
