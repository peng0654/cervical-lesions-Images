# InitalDataScreeing.R
# Initial Screeting cervial data
# Auther Peng Ren
# Mar 22nd 2020


# setwd to /code folder
#setwd("C:/Peng's file/github/CervialImagesProject/cervical-lesions-Images/Yan_excel/code")

suppressMessages({
  library(openxlsx)
  library(dplyr)
  library(tidyr)
  library(magrittr)
  library(ggplot2)
})

##### read data######3

datin<-read.xlsx("C:/Peng's file/yan_dat/cervial_excel_2017dat_postprocessing.xlsx")%>%
  select(ID_AND_TIME, ID, Age.y, pregnant, birth, cervial_cell, HPV, diagnostic)%>%
  rename(Age=Age.y)%>% filter(!is.na(diagnostic))

####### by age#########
Age_bin<-datin%>%
  mutate(Agebucket =cut(Age,breaks = c(-Inf,seq(25,50,5),Inf)))%>%
  group_by(Agebucket)%>%summarise(nobs=n(),diag=sum(diagnostic))%>%
  mutate(pastiveRate=diag/nobs*1000,
         Upperlimt= (1000/nobs)*(diag+sqrt(diag)*1.965),
         lowerlimt= (1000/nobs)*(diag-sqrt(diag)*1.965)
         )%>%select(Agebucket,nobs,pastiveRate,Upperlimt,lowerlimt)%>%
         mutate(scale=max(.$pastiveRate)/max(.$nobs),
                vol=nobs*scale)%>%select(-nobs)%>%
  tidyr::gather(type, value, matches('pastiveRate|Upperlimt|lowerlimt'))

p_age<-Age_bin%>%ggplot(aes(x=Agebucket,y=value, group=type,color=type,shape=type))+
              geom_point()+
              geom_line()+
  geom_col(aes(y=vol, color=NULL, fill = "# of Total Obs (Right Yaxis)"), position='dodge', alpha=0.1)+
  scale_y_continuous(sec.axis = sec_axis(~./Age_bin$scale[1]))+
  xlab("Age Buckets")+ylab("# of Passtive / 1000 ")+
  ggtitle("Passtive Rate and 95% CI by Age Buckets")+
  theme(text = element_text(size=20),
        axis.text.x = element_text(size=15))

dev.off()
pdf("../media/By_Age.pdf", width=28, height=18,onefile = T, paper='A4r')
p_age
dev.off()

########## by birthHist #############

brith_bin<-datin%>%select(pregnant, birth,diagnostic)%>%
                 drop_na()%>%
                mutate(Preg_Hist = cut(pregnant, breaks = c(-Inf,0,1,2,Inf), include.lowest = F),
                       Birth_Hist =cut(birth, breaks = c(-Inf,0,1,2,Inf), include.lowest = F))

brith_bin1<-brith_bin%>%
  group_by(Preg_Hist)%>%summarise(nobs=n(),diag=sum(diagnostic))%>%
  mutate(pastiveRate=diag/nobs*1000,
         Upperlimt= (1000/nobs)*(diag+sqrt(diag)*1.965),
         lowerlimt= (1000/nobs)*(diag-sqrt(diag)*1.965)
  )%>%select(Preg_Hist,nobs,pastiveRate,Upperlimt,lowerlimt)%>%
  mutate(scale=max(.$pastiveRate)/max(.$nobs),
         vol=nobs*scale)%>%select(-nobs)%>%
  tidyr::gather(type, value, matches('pastiveRate|Upperlimt|lowerlimt'))

p_brith_bin1<-brith_bin1%>%ggplot(aes(x=Preg_Hist,y=value, group=type,color=type,shape=type))+
  geom_point()+
  geom_line()+
  geom_col(aes(y=vol, color=NULL, fill = "# of Total Obs (Right Yaxis)"), position='dodge', alpha=0.1)+
  scale_y_continuous(sec.axis = sec_axis(~./brith_bin1$scale[1]))+
  xlab("Pregnancy Buckets")+ylab("# of Passtive / 1000 ")+
  ggtitle("Passtive Rate and 95% CI by Pregnancy History Buckets")+
  theme(text = element_text(size=20),
        axis.text.x = element_text(size=15))
dev.off()
pdf("../media/By_Pregnancy.pdf", width=28, height=18,onefile = T, paper='A4r')
p_brith_bin1
dev.off()



brith_bin2<-brith_bin%>%
  group_by(Birth_Hist)%>%summarise(nobs=n(),diag=sum(diagnostic))%>%
  mutate(pastiveRate=diag/nobs*1000,
         Upperlimt= (1000/nobs)*(diag+sqrt(diag)*1.965),
         lowerlimt= (1000/nobs)*(diag-sqrt(diag)*1.965)
  )%>%select(Birth_Hist,nobs,pastiveRate,Upperlimt,lowerlimt)%>%
  mutate(scale=max(.$pastiveRate)/max(.$nobs),
         vol=nobs*scale)%>%select(-nobs)%>%
  tidyr::gather(type, value, matches('pastiveRate|Upperlimt|lowerlimt'))

p_brith_bin2<-brith_bin2%>%ggplot(aes(x=Birth_Hist,y=value, group=type,color=type,shape=type))+
  geom_point()+
  geom_line()+
  geom_col(aes(y=vol, color=NULL, fill = "# of Total Obs (Right Yaxis)"), position='dodge', alpha=0.1)+
  scale_y_continuous(sec.axis = sec_axis(~./brith_bin2$scale[1]))+
  xlab("BrithGiven Buckets")+ylab("# of Passtive / 1000 ")+
  ggtitle("Passtive Rate and 95% CI by BirthGiven History Buckets")+
  theme(text = element_text(size=20),
        axis.text.x = element_text(size=15))


dev.off()
pdf("../media/By_Birth.pdf", width=28, height=18,onefile = T, paper='A4r')
p_brith_bin2
dev.off()



brith_bin3<-brith_bin%>%
  group_by(Birth_Hist,Preg_Hist)%>%summarise(nobs=n(),diag=sum(diagnostic))%>%
  mutate(pastiveRate=diag/nobs*1000,
         Upperlimt= (1000/nobs)*(diag+sqrt(diag)*1.965),
         lowerlimt= (1000/nobs)*(diag-sqrt(diag)*1.965)
  )%>%select(Birth_Hist,Preg_Hist,nobs,pastiveRate,Upperlimt,lowerlimt)%>%
  tidyr::gather(type, value, matches('pastiveRate|Upperlimt|lowerlimt'))

p_brith_bin3<-brith_bin3%>%ggplot(aes(x=Birth_Hist,y=value, group=type,color=type,shape=type))+
  geom_point()+
  geom_line()+
  facet_wrap(~Preg_Hist, ncol=1, scales = "fixed")+
  # geom_col(aes(y=vol, color=NULL, fill = "# of Total Obs (Right Yaxis)"), position='dodge', alpha=0.1)+
  # scale_y_continuous(sec.axis = sec_axis(~./brith_bin2$scale[1]))+
  xlab("BrithGiven Buckets")+ylab("# of Passtive / 1000 ")+
  theme(text = element_text(size=20),
        plot.title = element_text(size = 15, face = "bold"),
        axis.text.x = element_text(size=15))+
  ggtitle("Passtive Rate and 95% CI by BirthGiven History Buckets based on different Pregance History")


dev.off()
pdf("../media/By_Birth_and_Pregance.pdf", width=28, height=18,onefile = T, paper='A4r')
p_brith_bin3
dev.off()




