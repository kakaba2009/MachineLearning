library(rEDM)

source('./mylib/mcalc.R')
source('./mylib/mtool.R')

df <- loadSymbol('JPY=X')
df <- df$Close
df <- tail(df, 1000)

lib  <- c(1, 900)
pred <- c(901, 1000)

BestE <- BestDimEDM(df, lib, pred)

cat("BestE: ", BestE, "\n")

smap_output <- s_map(df, lib, pred, E=BestE, stats_only=TRUE)

m <- which.max(smap_output$rho)
cat("Best rho run:", m, "\n")

bestT <- smap_output$theta[m]
cat("Best theta:", bestT, "\n")

par(mar = c(4, 4, 1, 1), mgp = c(2.5, 1, 0))

plot(smap_output$theta, smap_output$rho, type = "l", xlab = "Nonlinearity (theta)", 
     ylab = "Forecast Skill (rho)")

smap_output <- s_map(df, lib, pred, E=BestE, stats_only=FALSE)

observed  <- smap_output[[m]]$model_output$obs
predicted <- smap_output[[m]]$model_output$pred
print(tail(predicted))

par(mar = c(4, 4, 1, 1), pty = "s")
plot_range <- range(c(observed, predicted), na.rm = TRUE)
plot(observed, predicted, xlim = plot_range, ylim = plot_range, xlab = "Observed", ylab = "Predicted")
abline(a = 0, b = 1, lty = 2, col = "blue")