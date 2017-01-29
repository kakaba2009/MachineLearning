library(rEDM)
library(data.table)
library(dplyr)

source('./mylib/mcalc.R')
source('./mylib/mtool.R')

df1 <- loadSymbol('JPY=X')
df1 <- tail(df1, 1000)

df2 <- loadSymbol('EUR=X')
df2 <- tail(df2, 1000)

setDT(df1)
setDT(df2)

df <- inner_join(df1, df2, by=c("Date"))

data(sardine_anchovy_sst)

anchovy_xmap_sst <- ccm(df, E = 3, lib_column = "Close.x", target_column = "Close.y", 
                        lib_sizes = seq(500, 1000, by = 100), 
                        random_libs = FALSE, first_column_time=TRUE)

sst_xmap_anchovy <- ccm(df, E = 3, lib_column = "Close.y", target_column = "Close.x", 
                        lib_sizes = seq(500, 1000, by = 100), 
                        random_libs = FALSE, first_column_time=TRUE)

a_xmap_t_means <- ccm_means(anchovy_xmap_sst)
t_xmap_a_means <- ccm_means(sst_xmap_anchovy)

par(mar = c(4, 4, 1, 1), mgp = c(2.5, 1, 0))

plot(a_xmap_t_means$lib_size, pmax(0, a_xmap_t_means$rho), type = "l", col = "red", 
     xlab = "Library Size", ylab = "Cross Map Skill (rho)", ylim = c(0, 1.0))

lines(t_xmap_a_means$lib_size, pmax(0, t_xmap_a_means$rho), col = "blue")

legend(x = "topleft", legend = c("anchovy xmap SST", "SST xmap anchovy"), col = c("red", "blue"), 
       lwd = 1, inset = 0.02, cex = 0.8)