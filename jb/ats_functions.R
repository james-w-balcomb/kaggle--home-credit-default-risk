
#### ats_padding_string ####
ats_padding_string <- function(character_count,padding_character=" ") {
  padding_output <- character() # initialize the character vector
  for (itr in 1:character_count) {
    padding_output <- paste(padding_output, padding_character, sep="")
  }
  padding_output
}
#ats_padding_string(78)

#### ats_mode ####
ats_mode <- function(ats_vector, na.rm = FALSE) {
  if (na.rm) {
    ats_vector <- ats_vector[!is.na(ats_vector)] # rebuild the vector without the missing values from the input vector
  }
  ats_vector_unique <- unique(ats_vector) # build a new vector of distinct values from the input vector
  return(ats_vector_unique[which.max(tabulate(match(ats_vector, ats_vector_unique)))])
  #https://www.tutorialspoint.com/r/r_mean_median_mode.htm
  #http://stackoverflow.com/questions/2547402/is-there-a-built-in-function-for-finding-the-mode
}
#ats_mode(c(1,2,3,5,6,7,8,9,0,0))

#### ats_standard_error ####
## Calculate the Standard Error of the Mean (SE) for a vector.
ats_standard_error <- function(ats_vector) {
  ats_vector_narm <- ats_vector[!is.na(ats_vector)] # rebuild the vector without the missing values from the input vector
  ats_count <- length(ats_vector_narm)
  ats_stardard_deviation <- sd(ats_vector_narm)
  ats_count_squareroot <- sqrt(ats_count)
  ats_stderr <- ats_stardard_deviation/ats_count_squareroot
  return(ats_stderr)
  #http://rcompanion.org/rcompanion/c_03.html
  #
}
#ats_standard_error(c(1,2,3,5,6,7,8,9,0,0))

#### ats_output_vector ####
ats_output_vector <- function(ats_vector) {
  if (length(ats_vector) == 0) {
    max_char_len <- 0
  } else {
    max_char_len <- max(nchar(ats_vector))
  }
  column_count <- floor(80/max_char_len)
  chars_per_column <- (max_char_len + 1)
  column_counter <- 1
  #cat("max_char_len: ", max_char_len, "\n")
  #cat("column_count: ", column_count, "\n")
  #cat("chars_per_column: ", chars_per_column, "\n")
  for (row in ats_vector) {
    padding_length <- (chars_per_column - nchar(row))
    #cat(itr, "\n")
    #cat("padding_length: ", padding_length, "\n")
    cat(row, ats_padding_string(padding_length))
    if (column_counter < column_count) {
      column_counter <- (column_counter + 1)
    } else {
      column_counter <- 1
      cat("\n")
    }
  }
  cat("\n")
}

#### ats_classlist_dataframe ####
ats_classlist_dataframe <- function(ats_dataframe) {
  list_factors<- character() # initialize the character vector
  list_integers <- character() # initialize the character vector
  list_numerics <- character() # initialize the character vector
  list_logicals <- character() # initialize the character vector
  list_unidentified <- character() # initialize the character vector
  for(name in names(ats_dataframe)){
    #if (is.factor(ats_dataframe[[name]])) {
    if (is.factor(ats_dataframe[,name])) {
      #cat("factor: ", name, "\n") # outpout "factory: " and the name of the data-frame column
      list_factors <- c(list_factors, name)
    } else if (is.integer(ats_dataframe[,name])) {
      #cat("integer: ", name, "\n") # outpout "integer: " and the name of the data-frame column
      list_integers <- c(list_integers, name)
    } else if (is.numeric(ats_dataframe[,name])) {
      #cat("numeric: ", name, "\n") # outpout "numeric: " and the name of the data-frame column
      list_numerics <- c(list_numerics, name)
    } else if (is.logical(ats_dataframe[,name])) {
      #cat("logical: ", name, "\n") # outpout "logical: " and the name of the data-frame column
      list_logicals <- c(list_logicals, name)
    } else {
      #cat(class(name),": ", name, "\n") # output the class and the name of the unidentified data-frame column
      list_unidentified <- c(list_unidentified, name)
    }
  }
  #print(list_factors, row.names = FALSE)
  #cat(list_factors)
  #print.data.frame(list_factors, row.names = FALSE)
  cat("\n")
  cat("Factor Class Variables:","\n")
  ats_output_vector(list_factors)
  cat("\n")
  cat("Integer Class Variables:","\n")
  ats_output_vector(list_integers)
  cat("\n")
  cat("Numeric Class Variables:","\n")
  ats_output_vector(list_numerics)
  cat("\n")
  cat("Logical Class Variables:","\n")
  ats_output_vector(list_logicals)
  cat("\n")
  cat("Unidentified Class Variables:","\n")
  ats_output_vector(list_unidentified)
  cat("\n")
}

#### ats_describe_dataframe ####
ats_describe_dataframe <- function(ats_dataframe) {
  #cat("ats_dataframe: ", ) # output the name of the data-frame
  for(name in names(ats_dataframe)) {
    #cat("Column_Name: ", name, "\n") # DEBUG
    if (is.integer(ats_dataframe[,name])) {
      #cat(ats_dataframe[,name]) # DEBUG
      #cat(name) # DEBUG
      ats_count <- length(ats_dataframe[,name])
      ats_min <- min(ats_dataframe[,name], na.rm = TRUE)
      ats_max <- max(ats_dataframe[,name], na.rm = TRUE)
      ats_range <- (max(ats_dataframe[,name], na.rm = TRUE) - min(ats_dataframe[,name], na.rm = TRUE))
      ats_mode <- ats_mode(ats_dataframe[,name])
      ats_mean <- mean(ats_dataframe[,name], na.rm = TRUE)
      #TBD: geometric mean
      #TBD: harmonic mean
      ats_mean_trimmed <- mean(ats_dataframe[,name], na.rm = TRUE, trim = 0.2)
      ats_var <- var(ats_dataframe[,name], na.rm = TRUE)
      ats_stddev <- sd(ats_dataframe[,name], na.rm = TRUE)
      ats_stderr <- ats_standard_error(ats_dataframe[,name])
      #ats_stderr <- ats_stddev/sqrt(ats_count)
      #ats_stderr <- sd(x)/sqrt(length(x))
      #ats_stderr <- sqrt(var(x)/length(x))
      ats_median <- median(ats_dataframe[,name], na.rm = TRUE)
      ats_mad <- mad(ats_dataframe[,name], na.rm = TRUE)
      ats_iqr <- IQR(ats_dataframe[,name], na.rm = TRUE)
      
      cat("Column_Name: ", name, "\n")
      cat("Count(n) =", ats_count, "\n")
      cat("Minimum =", ats_min, "\n")
      cat("Maximum =", ats_max, "\n")
      cat("Range =", ats_range, "\n")
      cat("Mode =", ats_mode, "\n")
      cat("Mean =", ats_mean, "\n")
      cat("Trimmed Mean (20%) =", ats_mean_trimmed, "\n")
      cat("Variance = ", ats_var, "\n")
      cat("Standard Deviation (SD) =", ats_stddev, "\n")
      cat("Standard Error (SE) = ", ats_stderr, "\n")
      cat("Median =", ats_median, "\n")
      cat("Median Absolute Deviation (MAD) =", ats_mad, "\n")
      cat("Inner-Quartile Range (IQR) =", ats_iqr, "\n")
      cat("\n")
    }
  }
}

#### ats_plot_dataframe ####
ats_plot_dataframe <- function(ats_dataframe) {
  #ln <- length(names(ats_dataframe))
  for(name in names(ats_dataframe)) {
    #mname <- substitute(ats_dataframe[,name])
    if (is.integer(ats_dataframe[,name])) {
      hist(ats_dataframe[,name],main=name)
    } else if (is.numeric(ats_dataframe[,name])) {
      hist(ats_dataframe[,name],main=name)
    } else if (is.factor(ats_dataframe[,name])) {
      plot(ats_dataframe[,name],main=name)
    } else if (is.logical(ats_dataframe[,name])) {
      plot(ats_dataframe[,name],main=name)
    } else {
      cat("Unhandled Class: ", class(name),": ", name, "\n") # output the class and the name of the unidentified data-frame column
    }
  }
  #http://stackoverflow.com/questions/4877357/how-to-plot-all-the-columns-of-a-data-frame-in-r
  #...this function prints a histogram for numeric variables and a bar chart for factor variables...
  #par(mfrow=c(3,3),mar=c(2,1,1,1)) #my example has 9 columns
  #dfplot(students.inquiries.df)
}

#### ats_plot_dependent_Variable ####
ats_plot_dependent_Variable <- function(ats_dependent,ats_dataframe) {
  
  
}
