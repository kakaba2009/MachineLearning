library(timeSeries)
library(lubridate)
library(quantmod)
library(Quandl)
library(DBI)
library("RSQLite")

mtool.api = 'M-pbVU9jbrCEWxxTDKhc'

loadSymbol <- function(symbol) {
    #setwd("D:/workspace/rEDM")
    # connect to the sqlite file
    con = dbConnect(RSQLite::SQLite(), dbname="./db/forex.db")
    # get a list of all tables
    alltables <- dbListTables(con)
    #query <- paste('SELECT DISTINCT Symbol FROM FX',sep='')
    query <- paste('SELECT Date,Close,Open,High,Low FROM FX WHERE Symbol = ',"'",symbol,"'"
                   ,' ORDER BY Date ASC'
                   ,sep='')
    print(query)
    # get the populationtable as a data.frame
    p <- dbGetQuery( con, query )
    
    if(nrow(p)==0) {
        stop("invalid data frame: nrow is 0")
    }
    
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

getQuantmod <- function(symbol) {
    df <- getSymbols(symbol, src="yahoo", env=NULL)
    
    return(df)    
}

getQuandl <- function(symbol, feq="daily", start="1980-01-01", end="2032-01-01", t="raw", o="asc") {
    if(nchar(mtool.api) >0 ) {
        Quandl.api_key(mtool.api)
    }
    
    df <- Quandl(symbol, collapse=feq, start_date=start, end_date=end, type=t, order=o)
    
    class(df)
    
    return(df)
}

search <- function(symbol="Oil", db="YAHOO") {
    if(nchar(mtool.api) >0 ) {
        Quandl.api_key(mtool.api)
    }
    
    ret <- Quandl.search(symbol, database_code = db, per_page = 10)
    
    return(ret)
}

download <- function(db="NSE") {
    if(nchar(mtool.api) >0 ) {
        Quandl.api_key(mtool.api)
    }
    
    Quandl.database.bulk_download_to_file(db, paste("./db/", db,".zip", sep=""))
}