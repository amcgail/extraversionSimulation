library(ggplot2)

dF <- read.csv("/home/alec/extraversion simulation/degreeFreq200.csv")
dF <- read.csv("/home/alec/extraversion simulation/degreeFreq200WithNewcomers.csv")
dF <- read.csv("/home/alec/extraversion simulation/degreeFreq200WithDisaster.csv")
dF <- read.csv("/home/alec/extraversion simulation/AlphaDiscrete3degreeFreq200WithDisaster.csv")
dF <- read.csv("/home/alec/extraversion simulation/AlphaJustExtrovertsFreq200WithDisaster.csv")
dF <- read.csv("/home/alec/extraversion simulation/normalLargeVarianceWithDisaster.deg.csv")
dF <- read.csv("/home/alec/extraversion simulation/longerUniform80Friends.dist.csv")
dF <- read.csv("/home/alec/extraversion simulation/sHomoUniform.alpha.csv")
dF.homo <- read.csv("/home/alec/extraversion simulation/sHomoUniform.alpha.csv")
dF.homo.deg <- read.csv("/home/alec/extraversion simulation/sHomoUniform.degDist.csv")
dF$logFreq = log(dF$freq)

dF[dF$t<102.75,"sumThisT"] = 50
dF[dF$t>102.75,"sumThisT"] = 25
dF$percent = 100 * dF$freq / dF$sumThisT
g <- ggplot(dF.homo.deg, aes(t, freq, group=degree)) + geom_area(aes(fill=degree))
g + coord_cartesian(ylim=c(0,100))

ggplot(dF, aes(alpha)) + geom_histogram(bins=15, aes(col="black")) + facet_wrap(~degree)

dF <- read.csv("/home/alec/extraversion simulation/longerUniform80Friends.csv")
ggplot(dF[dF$alpha <5,], aes(t, degree, group=alpha)) + geom_line(aes(color=alpha))


dF <- read.csv("/home/alec/extraversion simulation/antihomophilyOnExtroversion.hapEvol.csv")


library(plyr)
t <- 100

coefByTime.homo <- ldply( seq(0,150,10), function(t) {
  limitedT = dF.homo[dF.homo$t > t & dF.homo$t < t+10, ]
  rg <- lm( friendAvgAlpha~alpha, limitedT )
  c( summary(rg)$r.squared, coef(rg)['alpha'], t )
} )

coefByTime.anti <- ldply( seq(0,150,10), function(t) {
  limitedT = dF.anti[dF.anti$t > t & dF.anti$t < t+10, ]
  rg <- lm( friendAvgAlpha~alpha, limitedT )
  c( summary(rg)$r.squared, coef(rg)['alpha'], t )
} )
