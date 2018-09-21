require('data.table')
require('boot')
setwd('C:\\Users\\chenc\\Documents\\GitHub\\nmf\\2018-09-18-20-42-orl')
df=fread('raw_result_acc.csv')
df1=melt(df, id.vars = "V1")
df1$value=as.numeric(df1$value)
df2=dcast(df1, variable ~ V1)
df2=df2[,-1]
df3=data.matrix(df2)
n=1000
ci=list()
ndit=4
mean.fun <- function(dat, idx) mean(dat[idx], na.rm = TRUE)
for (i in 1:8){
	boot.out=boot(df3[,i],mean.fun,n)
	a=boot.ci(boot.out, conf = 0.95,type='perc')
	ci=c(ci,list(round(a$percent[4:5],ndit)))
}
ci_acc=ci

df=fread('raw_result_nmi.csv')
df1=melt(df, id.vars = "V1")
df1$value=as.numeric(df1$value)
df2=dcast(df1, variable ~ V1)
df2=df2[,-1]
df3=data.matrix(df2)
n=1000
ci=list()
mean.fun <- function(dat, idx) mean(dat[idx], na.rm = TRUE)
for (i in 1:8){
	boot.out=boot(df3[,i],mean.fun,n)
	a=boot.ci(boot.out, conf = 0.95,type='perc')
	ci=c(ci,list(round(a$percent[4:5],ndit)))
}
ci_nmi=ci

df=fread('raw_result_rre.csv')
df1=melt(df, id.vars = "V1")
df1$value=as.numeric(df1$value)
df11=df1
names(df11)=c('Algorithm','Noise','Relative residual error')
df11$Algorithm=as.character(df11$Algorithm)
df11$Noise=as.character(df11$Noise)
df11$Noise[df11$Algorithm==df11$Algorithm[1]]='No noise'
df11$Noise[df11$Algorithm==df11$Algorithm[2]]='No noise'
df11$Noise[df11$Algorithm==df11$Algorithm[3]]='Poisson noise'
df11$Noise[df11$Algorithm==df11$Algorithm[4]]='Poisson noise'
df11$Noise[df11$Algorithm==df11$Algorithm[5]]='Gaussian noise'
df11$Noise[df11$Algorithm==df11$Algorithm[6]]='Gaussian noise'
df11$Noise[df11$Algorithm==df11$Algorithm[7]]='Salt and Pepper noise'
df11$Noise[df11$Algorithm==df11$Algorithm[8]]='Salt and Pepper noise'
df11$Algorithm[df11$Algorithm==df11$Algorithm[1]]='NMF'
df11$Algorithm[df11$Algorithm==df11$Algorithm[2]]='KLNMF'
df11$Algorithm[df11$Algorithm==df11$Algorithm[3]]='NMF'
df11$Algorithm[df11$Algorithm==df11$Algorithm[4]]='KLNMF'
df11$Algorithm[df11$Algorithm==df11$Algorithm[5]]='NMF'
df11$Algorithm[df11$Algorithm==df11$Algorithm[6]]='KLNMF'
df11$Algorithm[df11$Algorithm==df11$Algorithm[7]]='NMF'
df11$Algorithm[df11$Algorithm==df11$Algorithm[8]]='KLNMF'
noises=unique(df11$Noise)
plots=list()
for (nn in noises){
  g=ggplot(df11[Noise==nn], aes(x=`Relative residual error`, fill=Algorithm)) +
geom_histogram(alpha=0.2, position="identity",bins=60)+
    ylab("Count")
  plots=c(plots,list(g+theme(legend.position="none")))
  
}
legend <- get_legend(g+theme(legend.position="top"))
plot_grid(plotlist = plots, labels=noises,hjust =c(-1,-0.55,-0.5,-0.35)) +
  theme(plot.margin=unit(c(1,0,0,0),"cm"))+
draw_grob(legend, .45, .53, .3/3.3, 1)
ggsave(filename = 'histo.pdf',width = 7, height = 7, units = "in")
df2=dcast(df1, variable ~ V1)
df2=df2[,-1]
df3=data.matrix(df2)
n=1000
ci=list()
mean.fun <- function(dat, idx) mean(dat[idx], na.rm = TRUE)
for (i in 1:8){
	boot.out=boot(df3[,i],mean.fun,n)
	a=boot.ci(boot.out, conf = 0.95,type='perc')
	ci=c(ci,list(round(a$percent[4:5],ndit)))
}
ci_rre=ci

df=fread('statistics_large.csv')
df2=as.matrix(df)
for (i in 1:8) df2[i,2]=paste0(round(df[i,2],ndit),' ',substr(ci_rre[i],start = 2,stop=nchar(ci_rre[i])))
for (i in 1:8) df2[i,3]=paste0(round(df[i,3],ndit),' ',substr(ci_acc[i],start = 2,stop=nchar(ci_acc[i])))
for (i in 1:8) df2[i,4]=paste0(round(df[i,4],ndit),' ',substr(ci_nmi[i],start = 2,stop=nchar(ci_nmi[i])))
df2=data.table(df2)
names(df2)=c('\\textsc{orl} dataset','\\textsc{rre}','\\textsc{acc}','\\textsc{nmi}')
df2[,1]=c('\\textsc{nmf} no noise','\\textsc{nmf} Gaussian noise','\\textsc{nmf} Poisson noise','\\textsc{nmf} Salt and Pepper noise'
	    ,'\\textsc{klnmf} no noise','\\textsc{klnmf} Gaussian noise','\\textsc{klnmf} Poisson noise','\\textsc{klnmf} Salt and Pepper noise')
fwrite(df2,'statistics_large_ci.csv')

df=fread('raw_result_rre.csv')
df1=melt(df, id.vars = "V1")
df1$value=as.numeric(df1$value)
df2=dcast(df1, variable ~ V1)
df2=df2[,-1]
df3=data.matrix(df2)
tes=list()
for (i in 1:4)	tes=c(tes,list(ks.test(df3[,i],df3[,i+4],alternative ='less')))
tes2=list()
for (i in 1:4)	tes2=c(tes2,list(ks.test(df3[,i],df3[,i+4],alternative ='greater')))
tes3=list()
for (i in 1:4)	tes3=c(tes3,list(ks.test(df3[,i],df3[,i+4])))
