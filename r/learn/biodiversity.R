library(rEDM)
library(data.table)
library(dplyr)

source('./mylib/mcalc.R')
source('./mylib/mtool.R')

data(e120_biodiversity)
head(e120_biodiversity)

# separate time column from data
composite_ts <- e120_biodiversity[, c(7:9, 12)]

# normalize each time series
n <- NCOL(composite_ts)
blocks <- e120_biodiversity$Plot
blocks_index <- sort(unique(blocks))
for (j in 1:n) {
    for (i in 1:length(blocks_index)) {
        subs <- which(blocks == blocks_index[i])
        composite_ts[subs, j] <- (composite_ts[subs, j] - mean(composite_ts[subs, j]))/sd(composite_ts[subs, j])
    }
}

composite_ts <- cbind(year = e120_biodiversity$Year, composite_ts)

#make composite library
segments <- NULL
startpos <- 1
for(i in 2:nrow(composite_ts)) {
    if(composite_ts$year[i] < composite_ts$year[i-1]) {
        segments <- rbind(segments, c(startpos, i))
        startpos <- i+1
    }
}
segments <- rbind(segments, c(max(segments)+1, nrow(composite_ts)))

#Choose random segments for prediction
set.seed(2312)
rndlib <- sort(sample(1:nrow(segments), round(nrow(segments)/2,0), rep=FALSE))
composite_lib <- segments[rndlib,]
composite_pred <- segments[-rndlib,]

precip_ts <- unique(e120_biodiversity[,c("Year", "SummerPrecip.mm.")])
precip_ts <- precip_ts[order(precip_ts$Year),]

par(mar = c(4, 4, 1, 1), mfrow = c(2, 2), mgp = c(2.5, 1, 0))
varlst <- colnames(composite_ts)[2:4]
simplex_output_list <- NULL

for (i in 1:length(varlst)) {
    simplex_output_list[[i]] <- simplex(composite_ts[, c("year", varlst[i])], 
                                        lib = composite_lib, pred = composite_pred, E = c(2:6))
    plot(simplex_output_list[[i]]$E, simplex_output_list[[i]]$rho, type = "l", 
         xlab = "Embedding Dimension (E)", ylab = "Forecast Skill (rho)", main = varlst[i])
}

simplex_output_list[[4]] <- simplex(precip_ts, lib = c(1, 7), pred = c(1, 7), 
                                    E = c(2:5), silent = TRUE)
names(simplex_output_list) <- c(varlst, "precipmm")
plot(simplex_output_list[[4]]$E, simplex_output_list[[4]]$rho, type = "l", xlab = "Embedding Dimension (E)", 
     ylab = "Forecast Skill (rho)", main = "Precip")
