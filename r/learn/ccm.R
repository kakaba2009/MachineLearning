library(rEDM)
library(data.table)
library(dplyr)

source('./mylib/mcalc.R')
source('./mylib/mtool.R')

start = "2000-01-01"

df1 <- getQuandl("CURRFX/USDJPY", "daily", start, "raw")
#df1 <- loadSymbol('JPY=X')
df1 <- head(df1, 1000)

df2 <- getQuandl("CHRIS/CME_GC1", "daily", start, "raw")
#df2 <- loadSymbol('EUR=X')
df2 <- head(df2, 1000)

setDT(df1)
setDT(df2)

df <- inner_join(df1, df2, by=c("Date"))

data(sardine_anchovy_sst)

anchovy_xmap_sst <- ccm(df, E = 7, lib_column = "Rate", target_column = "Last", 
                        lib_sizes = seq(800, 1000, by = 100), 
                        random_libs = FALSE, first_column_time=TRUE)

sst_xmap_anchovy <- ccm(df, E = 7, lib_column = "Last", target_column = "Rate", 
                        lib_sizes = seq(800, 1000, by = 100), 
                        random_libs = FALSE, first_column_time=TRUE)

a_xmap_t_means <- ccm_means(anchovy_xmap_sst)
t_xmap_a_means <- ccm_means(sst_xmap_anchovy)

par(mar = c(4, 4, 1, 1), mgp = c(2.5, 1, 0))

plot(a_xmap_t_means$lib_size, pmax(0, a_xmap_t_means$rho), type = "l", col = "red", 
     xlab = "Library Size", ylab = "Cross Map Skill (rho)", ylim = c(0, 1.0))

lines(t_xmap_a_means$lib_size, pmax(0, t_xmap_a_means$rho), col = "blue")

legend(x = "topleft", legend = c("anchovy xmap SST", "SST xmap anchovy"), col = c("red", "blue"), 
       lwd = 1, inset = 0.02, cex = 0.8)