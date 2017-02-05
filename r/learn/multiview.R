library(rEDM)
library(data.table)
library(dplyr)

source('./R/multiview_interface.R')
source('./mylib/mcalc.R')
source('./mylib/mtool.R')

options(max.print=5.5E5)
BestE <- 3

df1 <- loadSymbol('JPY=X')
df1 <- tail(df1, 1000)
df1 <- df1[, -1]
df1 <- df1[, c('Close','Open','High','Low')]
df  <- make_block(df1, max_lag=BestE)

lib  <- c(1, floor(NROW(df)*0.9))
pred <- c(floor(NROW(df)*0.9), NROW(df))

multiview_output <- multiview(df, lib=lib, pred=pred, E=BestE, tp=1, target_column=1, 
                              stats_only=FALSE, first_column_time=TRUE)

observed  <- multiview_output[[1]]$model_output$obs
predicted <- multiview_output[[1]]$model_output$pred
print(tail(predicted))
par(mar = c(4, 4, 1, 1), pty = "s")
plot_range <- range(c(observed, predicted), na.rm = TRUE)
plot(observed, predicted, xlim = plot_range, ylim = plot_range, xlab = "Observed", ylab = "Predicted")
abline(a = 0, b = 1, lty = 2, col = "blue")
