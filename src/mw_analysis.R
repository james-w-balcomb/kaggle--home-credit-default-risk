

setwd('/Users/matt.winkler/Documents/repos/kaggle--home-credit-default-risk/src')

### Load requirements:
if (!require("pacman")) install.packages("pacman")
pacman::p_load(tidyverse, skimr, GGally, plotly, viridis, 
               caret, DT, data.table, lightgbm)

### Load dataset:
train <-fread('../data/application_train.csv', stringsAsFactors = FALSE, showProgress=F,
              data.table = F, na.strings=c("NA","NaN","?", ""))
test <-fread('../data/application_test.csv', stringsAsFactors = FALSE, showProgress=F,
             data.table = F, na.strings=c("NA","NaN","?", ""))
bureau <-fread('../data/bureau.csv', stringsAsFactors = FALSE, showProgress=F,
               data.table = F, na.strings=c("NA","NaN","?", ""))
prev <-fread('../data/previous_application.csv', stringsAsFactors = FALSE, showProgress=F,
             data.table = F, na.strings=c("NA","NaN","?", "")) 

# inspect training data:
train[sample(1:nrow(train), size = 1000),] %>% 
  datatable(filter = 'top', options = list(
    pageLength = 15, autoWidth = F
  ))

# summarize train dataset:
train %>% skim() %>% kable()

# Look at target variable:
train %>% count(TARGET) %>% kable()

# mean of target:
mean(train$TARGET)

# mean values of variables in tabular format:
dt <- data.table(train)
dt[, .(target_avg = mean(train$TARGET))]
options(scipen = 999) # turn off scientific notation
dt[, .N / nrow(dt), by = CODE_GENDER]
dt[, .N / nrow(dt), by = NAME_CONTRACT_TYPE]
dt[, .N / nrow(dt), by = FLAG_OWN_CAR]
dt[, .N / nrow(dt), by = FLAG_OWN_REALTY]
dt[, .N / nrow(dt), by = CNT_CHILDREN]
dt[, .N / nrow(dt), by = NAME_TYPE_SUITE]
dt[, .N / nrow(dt), by = NAME_EDUCATION_TYPE]
 # etc . . .

# same idea but with pretty plot:
train %>% 
  mutate(TARGET=as.factor(TARGET)) %>% 
  count(NAME_EDUCATION_TYPE, TARGET) %>% 
  plot_ly(x = ~NAME_EDUCATION_TYPE , y = ~n, color = ~TARGET,type = "bar") %>%
  # add_trace(y = ~LA_Zoo, name = 'LA Zoo')   %>%
  layout(title = "ORGANIZATION_TYPE Type Group" , 
         barmode = 'stack',
         xaxis = list(title = ""),
         yaxis = list(title = ""))


# 

graph <- list()

for (i in 1:21){
  
  graph[[i]] <- train[, sapply(train, is.numeric)] %>% na.omit() %>%
    select(TARGET,((i-1)*5+1):((i-1)*5+5)) %>%
    mutate(TARGET = factor(TARGET)) %>%
    ggpairs(aes(col = TARGET, alpha=.4))
  
  print(graph[[i]])
}

# Notes:

# Only 8% of cases in the train data are positive.  Think about oversampling from the minority class.
# Potentially eliminate some of the _AVG / _MEDI / _MODE columns because they'll be redundant
# Lots of NULLs - Consider their relationship to the target before dropping; may still be predictive
# TODO: Look for collinearities among input features






