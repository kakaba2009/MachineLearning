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

data(sardine_anchovy_sst)

anchovy_xmap_sst <- ccm(df, E = 3, lib_column = "Close.x", target_column = "Close.y", 
                        lib_sizes = seq(800, 1000, by = 200), 
                        random_libs = FALSE, first_column_time=TRUE)

sst_xmap_anchovy <- ccm(df, E = 3, lib_column = "Close.y", target_column = "Close.x", 
                        lib_sizes = seq(800, 1000, by = 200), 
                        random_libs = FALSE, first_column_time=TRUE)

a_xmap_t_means <- ccm_means(anchovy_xmap_sst)
t_xmap_a_means <- ccm_means(sst_xmap_anchovy)

par(mar = c(4, 4, 1, 1), mgp = c(2.5, 1, 0))

plot(a_xmap_t_means$lib_size, pmax(0, a_xmap_t_means$rho), type = "l", col = "red", 
     xlab = "Library Size", ylab = "Cross Map Skill (rho)", ylim = c(0, 0.4))

lines(t_xmap_a_means$lib_size, pmax(0, t_xmap_a_means$rho), col = "blue")

legend(x = "topleft", legend = c("anchovy xmap SST", "SST xmap anchovy"), col = c("red", "blue"), 
       lwd = 1, inset = 0.02, cex = 0.8)
