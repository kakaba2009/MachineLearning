library(nonlinearTseries)
library(quantmod)
library(tsDyn)
library(rEDM)
library(zoo)

calcTakens <- function(x, dim, lag) {
    takens = buildTakens(x, embedding.dim=dim, time.lag=lag)
    
    plot(takens)
    
    return(takens)
}

calcTimeLag <- function(x) {
    lag <- timeLag(x, technique="ami",n.partitions = 20, units = "Bits")
    #lag <- timeLag(x, technique="acf")
    
    return(lag)
}

calcLyapunov <- function(ts, lag, radius=1.0) {
    ml=maxLyapunov(time.series=ts,
                   min.embedding.dim=2,
                   max.embedding.dim=8,
                   time.lag=lag,
                   radius=1.0,theiler.window=4,
                   min.neighs=2,min.ref.points=500,
                   max.time.steps=200,do.plot=FALSE)
    
    plot(ml)
    
    ml.estimation = estimate(ml,regression.range = c(0,15),
                             use.embeddings=4:8,
                             do.plot = TRUE)
    #max Lyapunov exponent
    cat("max Lyapunov exponent: ", ml.estimation, "\n")
    return (ml.estimation)
}

calcBestDimension <- function(x, lag=1, t=0.90) {
    dimension = estimateEmbeddingDim(x, time.lag=lag, max.embedding.dim=10,
                                     threshold=t, do.plot=TRUE)
    
    cat("Best dimension:", dimension, "\n")
    
    return(dimension)
}

calcRecurrPlot <- function(tk, ts, d, lag=1, r=1.0) {
    recurrencePlot(takens = tk, time.series=ts, embedding.dim=d, time.lag=lag, radius=r)    
}

calcInfoDimension <- function(ts, d, lag, r=0.001) {
    i <- infDim(time.series=ts, min.embedding.dim=d,
                max.embedding.dim=d, time.lag=lag, min.fixed.mass=0.01,
                max.fixed.mass=0.9, number.fixed.mass.points=100, radius=r,
                increasing.radius.factor=sqrt(2), number.boxes=100,
                number.reference.vectors=100, theiler.window=2,
                kMax=100, do.plot=FALSE)
    
    r <- sapply(i[2], mean)
    
    cat('info dimension logRadius:', r, '\n')
    
    r <- 10 ** r
    
    cat('info dimension Radius:', r, '\n')
    
    names(r) <- "radius"
    
    return(r)
}

calcRQA <- function(tk, ts, d, r) {
    rqa.analysis=rqa(takens = tk, time.series = ts, embedding.dim=d, time.lag=1,
                     radius=r,lmin=2,do.plot=FALSE,distanceToBorder=2)
    plot(rqa.analysis)

    cat("Percentage of recurrence points in a Recurrence Plot:",     rqa.analysis$REC,  "\n")
    cat("Percentage of recurrence points that form diagonal lines:", rqa.analysis$DET,  "\n")
    cat("Percentage of recurrent points that form vertical lines:",  rqa.analysis$LAM,  "\n")
    cat("Length of the longest diagonal line:",                      rqa.analysis$Lmax, "\n")
    cat("Length of the longest vertical line:",                      rqa.analysis$Vmax, "\n")
    cat("Shannon entropy of the diagonal line length distribution:", rqa.analysis$ENTR, "\n")
    
    #plot(rqa.analysis$diagonalHistogram)
    
    return(rqa.analysis)
}

AdditiveNonlinearAutoregressive <- function(x, d) {
    #fit an AAR model:
    mod <- aar(x, m=d)
    #Summary informations:
    summary(mod)
    #Diagnostic plots:
    #plot(mod)    
}

SpaceTimePlot <- function(tak, d, steps=100) {
    #tak = buildTakens(sin(2*pi*0.005*(0:5000)),2,1)
    stp = spaceTimePlot(takens=t,embedding.dim=d,number.time.steps=steps,do.plot=TRUE)
    
    return(stp)
}

NeuralNetworkNonlinearAutoregressiveModel <- function(x, d, lag, step, num) {
    mod.nnet <- nnetTs(x, m=d, d=lag, steps=step, size=num)
    return(mod.nnet)
}

FindDominantFrequency <- function(x) {
    f <- findfrequency(x)
    
    return(f)
}

CorrDim <- function(ts, d1=1, d2=9, r1=0.05, r2=1, lag=1) {
    cd=corrDim(time.series=ts,
               min.embedding.dim=d1,max.embedding.dim=d2,
               corr.order=2,time.lag=lag,
               min.radius=r1,max.radius=r2,
               n.points.radius=100,
               theiler.window=20,
               do.plot=TRUE)
    
    return(cd)
}

SampleEntropy <- function(cd, d1, d2, r1, r2) {
    use.col = c("#999999", "#E69F00", "#56B4E9", "#009E73",
                "#F0E442", "#0072B2", "#D55E00", "#CC79A7")
    se=sampleEntropy(cd,do.plot=TRUE,col=use.col,
                     type="l",xlim=c(0.1,1),
                     add.legend=T)
    se.est = estimate(se,
                      regression.range = c(r1, r2),
                      use.embeddings = seq(d1, d2), col=use.col, type="b")
    print(se.est)
    
    cat("Estimated = ", mean(se.est),"\n")
}

BestDimEDM <- function(df, lib = c(1, NROW(df)), pred = lib, E = 2:12, nn = "e+1") {
    simplex_output <- simplex(df, lib, pred, E = E, num_neighbors = nn)
    
    bestE <- simplex_output$E[which.max(simplex_output$rho)]
    
    par(mar = c(4, 4, 1, 1), mgp = c(2.5, 1, 0))
    
    plot(simplex_output$E, simplex_output$rho, type = "l", xlab = "Embedding Dimension (E)", ylab = "Forecast Skill (rho)")    
    
    cat("BestE: ", bestE, "\n")
    
    return(bestE)    
}

BestThetaEDM <- function(df, lib, pred, BestE) {
    smap_output <- s_map(df, lib, pred, E=BestE, stats_only=TRUE)
    
    m <- which.max(smap_output$rho)
    cat("Best rho run:", m, "\n")
    
    bestTheta <- smap_output$theta[m]
    cat("Best theta:", bestTheta, "\n")
    
    par(mar = c(4, 4, 1, 1), mgp = c(2.5, 1, 0))
    plot(smap_output$theta, smap_output$rho, type = "l", xlab = "Nonlinearity (theta)", ylab = "Forecast Skill (rho)")
    
    return(bestTheta)
}

MinDistance <- function(df, k=2) {
    nearest <- nn2(df,df, k=k)
    m <- mean(nearest$nn.dists[,k])
    cat("min distance:", m, "\n")
}

MaxDistance <- function(df, k=NROW(df)) {
    nearest <- nn2(df,df, k=k)
    m <- mean(nearest$nn.dists[,k])
    cat("max distance:", m, "\n")
}

ComputeVol <- function(ts, width=20) {
    vol <- rollapply(ts, width, sd)
    
    return(vol)
}

ComputeChange <- function(ts1, ts2=NULL, k=1, type="arithmetic") {
    chg <- Delt(x1 = ts1, x2 = ts2, k = k, type = type)
    
    #ts <- log(ts[-1]/ts[-n])
    
    chg <- na.omit(chg)
    
    return(chg)
}

ComputeDifference <- function(ts, lag=1) {
    delta <- diff(ts, lag=lag) #From base package
    
    return(delta)
}

ComputeSign <- function(delts) {
    pattern <- sign(delts) #From base package
    
    return(pattern)
}

ComputeTimeDiff1 <- function(df) {
    t <- df[,1] #First is time
    
    n <- NROW(t)
    
    d <- difftime(t[-1], t[-n], units="days")
    
    return(d)
}

ComputeTimeDiff2 <- function(df) {
    t <- df[,1] #First is time
    
    n <- NROW(t)
    
    d <- difftime(t[1], t, units="days")
    
    d <- abs(d)
    
    return(d)
}