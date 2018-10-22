dF <- read.csv("/home/alec/extraversion simulation/utilitySpreadForGivenAlpha.csv")
ggplot(dF, aes(sInvSumInv)) + geom_histogram(bins=100, color="black")
ggplot(dF, aes(sSum)) + geom_histogram(bins=100, color="black")

dF <- read.csv("/home/alec/extraversion simulation/alphaSpreadAffectsUtility.csv")
ggplot(dF, aes(spread,sSum)) + geom_errorbar(aes(ymin=sSum-sSumStd,ymax=sSum+sSumStd, width=0.1))
