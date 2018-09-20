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
for (i in 1:4)	tes=c(tes,list(ks.test(df3[,i],df3[,i+4])))
