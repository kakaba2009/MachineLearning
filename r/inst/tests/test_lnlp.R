library(rEDM)
library(data.table)

source("./mylib/mtool.R")

#data("two_species_model")
two_species_model <- loadTimeSeries('JPY=X', 'High')

ts <- tail(two_species_model, n=1000)

# univariate simplex projection using E = 1:10, and leave-one-out cross-validation
x <- simplex(ts, stats_only = FALSE)

# univariate simplex projection using E = 1:10, and first half to predict second half
simplex(ts, lib = c(1, 900), pred = c(901, 1000))

# univariate s-map using E = 7
s_map(ts, E = 7)

# univariate s-map using E = 7, theta = 1, and full output with smap_coefficients
y <- s_map(ts, E = 7, theta = 1, save_smap_coefficients = TRUE)
