library(rEDM)
library(data.table)
library(dplyr)

source('./mylib/mcalc.R')
source('./mylib/mtool.R')

start = "2000-01-01"

options(max.print=5.5E5)

df1 <- getQuandl("CUR/JPY", "daily", start, "raw")
#df1 <- loadSymbol('JPY=X')
df1 <- df1[, c(1,2)]
df1 <- head(df1, 200)

BestE <- BestDimEDM(df1[,2])

df2 <- getQuandl("CHRIS/CME_GC1", "daily", start, "raw")
#df2 <- loadSymbol('TWD=X')
df2 <- df2[, c(1,2)]
df2 <- head(df2, 200)

setDT(df1)
setDT(df2)

df <- inner_join(df1, df2, by=c('DATE' = 'Date'))

x_xmap_y <- ccm(df, E = BestE, lib_column = 1, target_column = 2, 
                        lib_sizes = seq(10, 200, by = 20), 
                        random_libs = FALSE, first_column_time=TRUE)
y_xmap_x <- ccm(df, E = BestE, lib_column = 2, target_column = 1, 
                        lib_sizes = seq(10, 200, by = 20), 
                        random_libs = FALSE, first_column_time=TRUE)

x_xmap_y_means <- ccm_means(x_xmap_y)
y_xmap_x_means <- ccm_means(y_xmap_x)

par(mar = c(4, 4, 1, 1), mgp = c(2.5, 1, 0))
plot(x_xmap_y_means$lib_size, pmax(0, x_xmap_y_means$rho), type = "l", col = "red", 
     xlab = "Library Size", ylab = "Cross Map Skill (rho)", ylim = c(0, 1.0))
lines(y_xmap_x_means$lib_size, pmax(0, y_xmap_x_means$rho), col = "blue")
legend(x = "topleft", legend = c("x xmap y", "y xmap x"), col = c("red", "blue"), 
       lwd = 1, inset = 0.02, cex = 0.8)
