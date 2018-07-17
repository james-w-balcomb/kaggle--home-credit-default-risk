

train.X$y <- y
super.sym <- trellis.par.get("superpose.symbol")

splom(train.X[c(1:120)], groups=y, data=train.X,
      panel=panel.superpose, 
      key=list(title="Default / Not Default",
               columns=2,
               points=list(pch=super.sym$pch[1:2],
                           col=super.sym$col[1:2]),
               text=list(c("Not Default","Default"))))


setwd('/Users/matt.winkler/Documents/repos/kaggle--home-credit-default-risk/src')

### Load requirements:
#if (!require("pacman")) install.packages("pacman")
pacman::p_load(tidyverse, skimr, GGally, plotly, viridis, 
               caret, DT, data.table, lightgbm)

install.packages("lightgbm")

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
# mean of target:
mean(train$TARGET)

# mean values of variables in tabular format:
dt <- data.table(train)
options(scipen = 999) # turn off scientific notation
class.values <- sapply(dt, class)
unique(class.values)

ints <- as.vector(class.values == "integer")
nums <- as.vector(class.values == "numeric")
chars <- as.vector(class.values == "character")

# look at number of unique values for each set of columns by type:
sapply(dt[, ..ints], n_unique)
#sapply(dt[, ..chars], n_unique)

# these columns are probably better treated as numerics:
dt$DAYS_BIRTH <- as.numeric(dt$DAYS_BIRTH)
dt$DAYS_BIRTH <- as.numeric(dt$DAYS_EMPLOYED)
dt$DAYS_BIRTH <- as.numeric(dt$DAYS_ID_PUBLISH)

# assign class values again:
class.values <- sapply(dt, class)

# core plotting function:
one_plot <- function(d, colname) {
  plot_ly(d, x = d[, colname], type="histogram") %>%
    add_annotations(
      ~unique(TARGET), x = 0.5, y = 1, 
      xref = "paper", yref = "paper", showarrow = FALSE
    )
}

# wrapper to call one_plot
plot.numeric <- function(data, colname) {
  #defines x axis
  x.axis <- list(
    title = colname,
    titlefont = list(
      family = "Courier New, monospace",
      size = 18,
      color = "#7f7f7f")
  )
  
  # split dataset by target values and call one_plot
  plt <- train %>%
    split(.$TARGET) %>%
    lapply(one_plot, colname=colname) %>% 
    subplot(nrows = 2, shareX = TRUE, titleX = TRUE) %>%
    layout(xaxis = x.axis) %>%
    hide_legend()

  plt
}

#pdf(file="core_dataset_plots.pdf")
# Loop through dataset and make plots:
for (i in 1:length(class.values)) {
  obj = class.values[i]
  cn = names(obj)
  cv = obj[[1]]
  # PDF setup:
  plotname = paste(cn, "_plot.pdf", sep="")
  if (cv == "numeric") {
    plt = plot.numeric(dt, cn)
    print(plt)
  } else {
    cat.plt.1 <- plot.cat.1(dt, cn)
    print(cat.plt.1)
    cat.plt.2 <- plot.cat.2(dt, cn)
    print(cat.plt.2)
  }
}





######## categorical / integer visualizations ######## 

plot.cat.1 <- function(data, colname) {
  data %>%
    mutate(TARGET=as.factor(TARGET)) %>%
    count(UQ(as.name(colname)), TARGET) %>%
    mutate(proportion = n / nrow(data)) %>%
    plot_ly(x = unique(select(data, colname)), y = ~proportion, color = ~TARGET, type="bar") %>%
     layout(barmode = "stack", 
            title = colname)
  
}

# proportion of attribute values by target variable:

plot.cat.2 <- function(data, colname) {
  data$TARGET = as.factor(data$TARGET)
  ct = count(data, UQ(as.name(colname)), TARGET)
  ct.2 <- left_join(ct, count(ct, UQ(as.name(colname)), wt = n))
  ct.2 %>%
    mutate(prop = n / nn) %>%
    plot_ly(x = unique(select(data, colname)), y = ~prop, color = ~TARGET) %>%
    add_bars() %>%
    layout(barmode = "stack", 
           title = colname)
}

plot.cat.1(dt, "NAME_EDUCATION_TYPE")
plot.cat.2(dt, "NAME_EDUCATION_TYPE")




"
graph <- list()

for (i in 1:21){
  
  graph[[i]] <- train[, sapply(train, is.numeric)] %>% na.omit() %>%
    select(TARGET,((i-1)*5+1):((i-1)*5+5)) %>%
    mutate(TARGET = factor(TARGET)) %>%
    ggpairs(aes(col = TARGET, alpha=.4))
  
  print(graph[[i]])
}
"

# Notes:

# Only 8% of cases in the train data are positive.  Think about oversampling from the minority class.
# Potentially eliminate some of the _AVG / _MEDI / _MODE columns because they'll be redundant
# Lots of NULLs - Consider their relationship to the target before dropping; may still be predictive
# TODO: Look for collinearities among input features






