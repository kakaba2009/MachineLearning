library(data.table)

source('./mylib/mcalc.R')
source('./mylib/mtool.R')

options(max.print=5.5E5)

start = as.Date("2016-01-01")
end   = as.Date("2017-12-31")

df <- getQuandl("CUR/JPY", "daily", start, end, "raw")
df <- df[, c("RATE")]
df <- diff(df, lag = 1, diff = 1)
yy <- df > 0
yy <- as.numeric(yy)
l1 <- shift(yy, n = 1, type = "lead")
l2 <- shift(yy, n = 2, type = "lead")

zz <- yy * 4 + l1 * 2 + l2

aa <- shift(zz, n = 1, type = "lead")
bb <- shift(zz, n = 2, type = "lead")
cc <- shift(zz, n = 3, type = "lead")

ic <- zz * 8 + aa * 0 + bb * 0 + cc * 1

plot(ic)
