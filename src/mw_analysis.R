

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
names <- colnames(dt)

class.values <- sapply(dt, class)
unique(class.values)
ints <- as.vector(class.values == "integer")
nums <- as.vector(class.values == "numeric")
chars <- as.vector(class.values == "character")

sapply(dt[, ..ints], n_unique)

# these columns are probably better treated as numerics:
dt$DAYS_BIRTH <- as.numeric(dt$DAYS_BIRTH)
dt$DAYS_BIRTH <- as.numeric(dt$DAYS_EMPLOYED)
dt$DAYS_BIRTH <- as.numeric(dt$DAYS_ID_PUBLISH)




for (n in names) {
  # finds the mean value of the target and proportion of the total dataset
  # represented by level of the factors in the data
  #result <- dt[, list(mean.val = sum(TARGET) / .N, # avg. value of the target variable by level 
  #                    prop.records = .N / nrow(dt)), # proportion of records in the data by level
  #             by = n]
  
  print(result)
}

######## numeric visualizations ########

one_plot <- function(d, col) {
  plot_ly(d, x = d[, col], type="histogram") %>%
    add_annotations(
      ~unique(TARGET), x = 0.5, y = 1, 
      xref = "paper", yref = "paper", showarrow = FALSE
    )
}



class.values

x.axis <- list(
  title = c,
  titlefont = list(
    family = "Courier New, monospace",
    size = 18,
    color = "#7f7f7f")
)

for (i in 1:length(class.values)) {
  cv = class.values[[i]]
  c = names[i]
  if (cv == "numeric") {
    
    plt <- train %>%
      split(.$TARGET) %>%
      lapply(one_plot, col=c) %>% 
      subplot(nrows = 2, shareX = TRUE, titleX = TRUE) %>%
      layout(xaxis = x.axis) %>%
      hide_legend()
    
    print(plt)
  }
}



######## categorical visualizations ######## 

# proportions of total data by attribute and target variable
train %>% 
  mutate(TARGET=as.factor(TARGET)) %>% 
  count(NAME_EDUCATION_TYPE, TARGET) %>% 
  mutate(prop = n / nrow(train)) %>%
  plot_ly(x = ~NAME_EDUCATION_TYPE , y = ~prop, color = ~TARGET,type = "bar") %>%
  # add_trace(y = ~LA_Zoo, name = 'LA Zoo')   %>%
  layout(title = "ORGANIZATION_TYPE Type Group" , 
         barmode = 'stack',
         xaxis = list(title = ""),
         yaxis = list(title = ""))

# proportion of attribute values by target variable:
dt$TARGET = as.factor(dt$TARGET)
my.cc <- count(dt, NAME_EDUCATION_TYPE, TARGET)
my.cc2 <- left_join(my.cc, count(my.cc, NAME_EDUCATION_TYPE, wt = n))

my.cc2 %>%
  mutate(prop = n / nn) %>%
  plot_ly(x = ~NAME_EDUCATION_TYPE, y = ~prop, color = ~TARGET) %>%
  add_bars() %>%
  layout(barmode = "stack")






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






