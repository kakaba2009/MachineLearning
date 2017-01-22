library(DBI)
library("RSQLite")

loadSymbol <- function(symbol) {
    #setwd("D:/workspace/rEDM")
    # connect to the sqlite file
    con = dbConnect(RSQLite::SQLite(), dbname="./db/forex.db")
    # get a list of all tables
    alltables <- dbListTables(con)
    #query <- paste('SELECT DISTINCT Symbol FROM FX',sep='')
    query <- paste('SELECT Date,Open,High,Low,Close FROM FX WHERE Symbol = ',"'",symbol,"'",sep='')
    # get the populationtable as a data.frame
    p <- dbGetQuery( con, query )
    
    return(p)
}

loadAllNames <- function() {
    # connect to the sqlite file
    con = dbConnect(RSQLite::SQLite(), dbname="./db/forex.db")
    # get a list of all tables
    alltables <- dbListTables(con)
    
    query <- paste('SELECT DISTINCT Symbol FROM FX',sep='')
    # get the populationtable as a data.frame
    p <- dbGetQuery( con, query )
    
    return(p)    
}

saveRData <- function() {
    forex <- loadSymbol('JPY=X')
    forex <- subset(forex, select = c(Date,High,Close))
    forex$Date<-seq.int(nrow(forex))
    forex <- setnames(forex, "Date" , "time")
    forex <- setnames(forex, "High" , "x")
    forex <- setnames(forex, "Close", "y")
    save(forex, file="./data/forex.rda")
}

loadTimeSeries <- function(symbol, field) {
    df <- loadSymbol(symbol)
    df <- subset(df, select = c(field))
    ts <- as.ts(df)
    
    return(ts)
}

saveCSV <- function(symbol) {
    df <- loadSymbol(symbol)
    
    write.csv(df, file = paste("./db/", symbol, ".csv", sep=''), row.names = FALSE)
}