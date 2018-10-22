library(plyr)
library(ggplot2)

workingDir <- "/home/alec/extraversion simulation/antiHomophily/"

i <- 0

p <- ggplot( NULL )
for( i in 1:15 ) {
  d <- read.csv(paste(workingDir, i, ".anti.unhap.csv", sep=""))
  p <- p + geom_line( data=d, mapping=aes(t, avgUnhap, color="Anti"), alpha=0.5 )
  
  d <- read.csv(paste(workingDir, i, ".homo.unhap.csv", sep=""))
  p <- p + geom_line( data=d, mapping=aes(t, avgUnhap, color="Homo"), alpha=0.5 )
}
p

p.rsq <- ggplot( NULL )
p.alph <- ggplot( NULL )
for( i in 1:15 ) {
  d <- read.csv(paste(workingDir, i, ".homo.alpha.csv", sep=""))
  measureHomo.homo <- ldply( seq(0,490,10), function(t) {
    limitedT = d[d$t > t & d$t < t+10, ]
    rg <- lm( friendAvgAlpha~alpha, limitedT )
    c( summary(rg)$r.squared, coef(rg)['alpha'], t )
  } )
  
  d <- read.csv(paste(workingDir, i, ".anti.alpha.csv", sep=""))
  measureHomo.anti <- ldply( seq(0,490,10), function(t) {
    limitedT = d[d$t > t & d$t < t+10, ]
    rg <- lm( friendAvgAlpha~alpha, limitedT )
    c( summary(rg)$r.squared, coef(rg)['alpha'], t )
  } )
  
  names(measureHomo.anti) <- c("RSquared", "alpha", "t")
  names(measureHomo.homo) <- c("RSquared", "alpha", "t")

  
  p.rsq <- p.rsq + 
    geom_line( data=measureHomo.homo, mapping=aes(t, RSquared, color="Homo"), alpha=0.5 ) +
    geom_line( data=measureHomo.anti, mapping=aes(t, RSquared, color="Anti"), alpha=0.5 )
  p.alph <- p.alph +
    geom_line( data=measureHomo.homo, mapping=aes(t, alpha, color="Homo"), alpha=0.5 ) +
    geom_line( data=measureHomo.anti, mapping=aes(t, alpha, color="Anti"), alpha=0.5 )
}

p
