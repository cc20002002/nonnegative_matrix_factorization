require('ggplot2')
library(ggthemes)
theme_set(theme_bw())  # from ggthemes
require(data.table)
x=t(t(c(-20:20)))
y1=t(t(dnorm(x,mean=0,sd=sqrt(40))))
y2=t(t(dpois(x+40,lambda = 40)))
df=cbind(rbind(x,x),rbind(y1,y2))
df=data.table(df)
df=cbind(df,'Gaussian')
df[42:82,3]='Poisson'
names(df)=c('Noise','Density','Distributions')
ggplot(df, aes(x=Noise, y=Density,color=Distributions)) + 
  geom_line() 

ggsave('noise.pdf',width = 8, units = "cm")
