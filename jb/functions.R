# https://github.com/benvliet/DSUR/blob/master/20150101_DSUR_h04.R
outlierSummary <- function(variable, digits = 2){

  zvariable <- (variable - mean(variable, na.rm = TRUE)) / sd(variable, na.rm = TRUE)

  outlier95 <- abs(zvariable) >= 1.96
  outlier99 <- abs(zvariable) >= 2.58
  outlier999 <- abs(zvariable) >= 3.29

  ncases <- length(na.omit(zvariable))

  percent95 <- round(100 * length(subset(outlier95, outlier95 == TRUE)) / ncases, digits)
  percent99 <- round(100 * length(subset(outlier99, outlier99 == TRUE)) / ncases, digits)
  percent999 <- round(100 * length(subset(outlier999, outlier999 == TRUE)) / ncases, digits)

  cat("Absolute z-score greater than 1.96 = ", percent95, "%", "\n")
  cat("Absolute z-score greater than 2.58 = ",  percent99, "%", "\n")
  cat("Absolute z-score greater than 3.29 = ",  percent999, "%", "\n")
}


missing_val_var <- function(data, variable, new_var_name) {
  data$new_var_name <- ifelse(is.na(variable), 1, 0)
  return(data$new_var_name)
}


numeric_impute <- function(data, variable) {
  mean1 <- mean(data$variable)
  data$variable <- ifelse(is.na(data$variable), mean1, data$variable)
  #return(new_var_name)
}


character_impute <- function(data,variable) {
  data$variable <- ifelse(is.na(data$variable), "MISSING", data$variable)
  #return(new_var_name)
}


create_model <- function(trainData,target) {
  set.seed(120)
  myglm <- glm(target ~ ., data = trainData, family = "binomial")
  return(myglm)
  }



flag_missing_values <- function(data,variable_name) {
  variable_name_new <- paste(variable_name,"MISSING", sep = "_")
  data$variable_name_new <- ifelse(is.na(variable_name), 1, 0)
  return(data$variable_name_new)
}


quantitative_impute <- function(data,variable) {
  mean1 <- mean(data$variable)
  data$variable <- ifelse(is.na(data$variable), mean1, data$variable)
  return(new_var_name)
}


qualitative_impute <- function(data,variable) {
  data$variable <- ifelse(is.na(data$variable), "MISSING", data$variable)
  return(new_var_name)
}

