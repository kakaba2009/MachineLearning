library(rEDM)

source('./mylib/mcalc.R')
source('./mylib/mtool.R')

df <- loadSymbol('JPY=X')
df <- df[ ,c('Close')]
df <- tail(df, 1000)

lib  <- c(1,   975)
pred <- c(976, 1001)

BestE <- BestDimEDM(df, lib, pred)
BestTheta <- BestThetaEDM(df, lib, pred, BestE)

smap_output <- s_map(df, lib, pred, E=BestE, theta = c(BestTheta), stats_only=FALSE)

observed  <- smap_output[[1]]$model_output$obs
predicted <- smap_output[[1]]$model_output$pred
print(tail(predicted))
par(mar = c(4, 4, 1, 1), pty = "s")
plot_range <- range(c(observed, predicted), na.rm = TRUE)
plot(observed, predicted, xlim = plot_range, ylim = plot_range, xlab = "Observed", ylab = "Predicted")
abline(a = 0, b = 1, lty = 2, col = "blue")