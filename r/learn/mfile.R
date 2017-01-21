library(DBI)
library("RSQLite")

#setwd("D:/workspace/rEDM")
# connect to the sqlite file
con = dbConnect(RSQLite::SQLite(), dbname="./db/forex.db")
# get a list of all tables
alltables <- dbListTables(con)
symbol <- "JPY=X"
#query <- paste('SELECT DISTINCT Symbol FROM FX',sep='')
query <- paste('SELECT Date,Open,High,Low,Close FROM FX WHERE Symbol = ',"'",symbol,"'",sep='')
# get the populationtable as a data.frame
p1 <- dbGetQuery( con, query )
# count the areas in the SQLite table
#p2 = dbGetQuery( con,'select count(*) from areastable' )
# find entries of the DB from the last week
#p3 = dbGetQuery(con, "SELECT population WHERE DATE(timeStamp) < DATE('now', 'weekday 0', '-7 days')")
#Clear the results of the last query
#dbClearResult(p3)
#Select population with managerial type of job
#p0 = dbGetQuery(con, "select * from populationtable where jobdescription like '%manager%'")