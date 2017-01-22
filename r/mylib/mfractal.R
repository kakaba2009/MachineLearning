library(fractaldim)

BoxCountDim <- function(ts) {
    bcd <- fd.estim.boxcount (ts, nlags = "all", plot.loglog = TRUE,
                       plot.allpoints = TRUE)
    
    return(bcd)
}