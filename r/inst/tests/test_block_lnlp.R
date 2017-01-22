library(rEDM)
library(data.table)

source('./mylib/mtool.R')

#data("two_species_model")
df <- loadSymbol('JPY=X')
len <- length(df$Date)
two_species_model <- df

block <- tail(two_species_model, n=1000)

# multivariate simplex projection using x and y to predict x
output <- block_lnlp(block, columns = c("Open", "High", "Low", "Close"), 
                     target_column = "Close", 
                     first_column_time = TRUE, stats_only = FALSE)

output <- output[[1]]$model_output
output$pred_err <- sqrt(output$pred_var)
t <- seq(from=1, to=1000)
plot(t, output$obs[t], type = "p", col = "green")
points(t, output$pred[t], col = "blue")
#for(i in t)
#{
#    lines(c(i,i), c(output$pred[i]-output$pred_err[i], 
#                    output$pred[i]+output$pred_err[i]), col = "blue")
#}

# cross mapping using x to predict y
#block_lnlp(block, target_column = 2, columns = c("x"), first_column_time = TRUE)
