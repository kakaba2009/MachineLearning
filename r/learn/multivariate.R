library(rEDM)
library(data.table)
library(dplyr)

options(max.print=5.5E5)

source('./R/multiview_interface.R')
source('./mylib/mcalc.R')
source('./mylib/mtool.R')

df1 <- loadSymbol('JPY=X')
df1  <- tail(df1, 3000)
df2 <- loadSymbol('EUR=X')
df2 <- tail(df2, 3000)
df3 <- loadSymbol('GBP=X')
df3 <- tail(df3, 3000)
df4 <- loadSymbol('CAD=X')
df4 <- tail(df4, 3000)
df5 <- loadSymbol('AUD=X')
df5 <- tail(df5, 3000)


setDT(df1)
setDT(df2)
setDT(df3)
setDT(df4)
setDT(df5)

tmp <- inner_join(df1, df2, by=c("Date"))
df  <- inner_join(tmp, df3, by=c("Date"))
df  <- inner_join(df,  df4, by=c("Date"))
df  <- inner_join(df,  df5, by=c("Date"))

lib  <- c(1, NROW(df))
pred <- c(1, NROW(df))

block_lnlp_output <- block_lnlp(df, lib=lib, pred=pred, method='simplex', tp=1, columns=1:12, 
                        target_column=4, stats_only=FALSE, first_column_time=TRUE)

observed  <- block_lnlp_output[[1]]$model_output$obs
predicted <- block_lnlp_output[[1]]$model_output$pred
print(tail(observed))
print(tail(predicted))
par(mar = c(4, 4, 1, 1), pty = "s")
plot_range <- range(c(observed, predicted), na.rm = TRUE)
plot(observed, predicted, xlim = plot_range, ylim = plot_range, xlab = "Observed", ylab = "Predicted")
abline(a = 0, b = 1, lty = 2, col = "blue")
