# RawexcelProcessing.R
# Processing cervial information
# Auther Peng Ren
# Mar 21st 2020


#install.packages('openxlsx')
#install.packages('tidyverse')

# setwd to /code folder
#setwd("C:/Peng's file/github/CervialImagesProject/cervical-lesions-Images/Yan_excel/code")

suppressMessages({
  library(openxlsx)
  library(dplyr)
  library(tidyr)
  #library(tidyverse)
  library(magrittr)
  library(ggplot2)
})

# load data
datin<-read.xlsx("C:/Peng's file/yan_dat/cervial_excel_2017dat.xlsx",sheet = 1)
datin2<-read.xlsx("C:/Peng's file/yan_dat/cervial_excel_2017dat.xlsx",sheet = 2)
Sys.setlocale(category= "LC_ALL", locale = "chinese")
datin2%<>%select("????????????","?????????")%>%
          rename(Examtime=????????????,
                 ID = ?????????)%>%
          mutate(Examtime=(as.Date(Examtime, '1899-12-30')),
                                     ID_AND_TIME=paste(as.character(ID),as.character(Examtime), sep = "_"),
                 diagnostic = 1 )


datin_use<-datin%>%select("????????????","?????????","??????","????????????")

colnames(datin_use)<-c('Examtime', 'ID', 'Age', 'Info')

datin_use%<>%mutate(Examtime=(as.Date(Examtime, '1899-12-30')),
                              ID_AND_TIME=paste(as.character(ID),as.character(Examtime), sep = "_"))


datin_use%<>%left_join(datin2, by = c("Examtime", "ID", "ID_AND_TIME"))

# processing......
datin_use_new<-datin_use%>%
         mutate(diagnostic=ifelse(is.na(diagnostic), 0, diagnostic),
                new_col=strsplit(Info, "_x000D_"),
                new_col1=lapply(new_col, grep, pattern="?????????|???????????????|HPV", value=T))%>%
        select(-Info,-new_col )%>%
  unnest(new_col1)%>%group_by(ID_AND_TIME)%>%mutate(key=row_number())%>%spread(key,new_col1)%>%
  select(Examtime,ID, Age,ID_AND_TIME,diagnostic,`1`,`2`,`3` )%>%
  mutate(birthHis= gsub("?????????:", "", `1`),
         cervial_cell= gsub("[\n]???????????????:", "", `2`),
         HPV = gsub("[\n]HPV??????:", "", `3`) )%>%select(-`1`,-`2`,-`3`)%>%
  mutate(pregnant=as.numeric(gsub('[[:space:]]',"",gsub('^???(.+)???.*', "\\1", birthHis))),
         birth = as.numeric(gsub('[[:space:]]',"",gsub('.*???', "\\1", birthHis))),
         Age = as.numeric(gsub("???", "", Age)))%>%
  select(Examtime,ID,Age,ID_AND_TIME,pregnant,birth,cervial_cell,HPV,diagnostic)

#check<-datin_use%>%anti_join(datin_use_new, by="ID_AND_TIME")


datin_use%>%select(-diagnostic)%>%
  left_join(datin_use_new, by = c('ID_AND_TIME', 'Examtime', 'ID' ))%>%
  write.xlsx("C:/Peng's file/yan_dat/cervial_excel_2017dat_postprocessing.xlsx")

