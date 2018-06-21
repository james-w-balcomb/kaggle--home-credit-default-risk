
####################### #
#### Prepare Session ####
####################### #------------------------------------------------------#
#  #
#------------------------------------------------------------------------------#

# Clear the Environment
rm(list = ls())

#
graphics.off()

# Set our preferred Graphical Parameters
# "ask","fig","fin","lheight","mai","mar","mex","mfcol","mfrow","mfg","new","oma","omd","omi","pin","plt","ps","pty","usr","xlog","ylog","ylbias"
graphicalParametersDefaults <- par(no.readonly = TRUE)
graphicalParametersDefaults
graphicalParameters <- graphicalParametersDefaults
graphicalParameters
graphicalParameters["mar"]
graphicalParameters["mar"] <- list(mar = c(5, 4, 4, 2) + 0.1)
graphicalParameters["mar"]
graphicalParameters["mfrow"]
graphicalParameters["mfrow"] <- list(mfrow = c(1, 1))
graphicalParameters["mfrow"]
par(graphicalParameters)

# Set the working directory, so that R knows where to find our files
#setwd("C:/Development/kaggle--home-credit-default-risk/jb")
##data_file_path <- Sys.getenv("HCDR_WORKING_DIRECTORY")


###################### #
#### Load Libraries ####
###################### #------------------------------------------------------#
# Load the requisite libraries, perhaps even installing them first            #
#-----------------------------------------------------------------------------#

# Problem Packages
## rattle # Requires RGtk2, which requires GTK+, an external dependency
## RGtk2  # Requires GTK+, an external dependency
## influ  # DNE? WTF?
##   install_github("trophia/influ") # https://github.com/trophia/influ
##   devtools::install_github("hoxo-m/githubinstall")
##   library(githubinstall)
##   githubinstall("influ")
##   source("https://install-github.me/r-lib/influ")
##   remotes::install_github("trophia/influ")
## doMC # DNE
## Rmpi # Requires msmpi.dll from Microsoft MPI
## > Error in Rmpi::mpi.comm.spawn(slave = mpitask, slavearg = args, nslaves = count,  :
## >   Internal MPI error!, error stack:
## > MPI_Comm_spawn(cmd="C:/PROGRA~1/R/R-34~1.3/bin/x64/Rscript.exe", argv=0x0000000063AE0ED0, maxprocs=8, MPI_INFO_NULL, root=0, MPI_COMM_SELF, intercomm=0x000000001943DCC0, errors=0x0000000061BFFFA0) failed
## > Internal MPI error!  FAILspawn not supported without process manager
## snow # Don't remember what the issue was
## coda # Don't remember what the issue was
##install.packages("parallel")
#library(parallel)
##install.packages("snow")
#library(snow)
##install.packages("Rmpi")
#library(Rmpi)

# Packages I do not need and/or want
##janitor

packageList <- scan("00_PackageList.txt", what = "", sep = "\n")
#packageList

for (element in packageList) {
    #print(element)
    library(element,
            lib.loc = NULL,
            character.only = TRUE,
            logical.return = FALSE,
            warn.conflicts = FALSE,
            quietly = TRUE,
            verbose = FALSE
    )
}



########################## #
#### Set Up Parallelism ####
########################## #---------------------------------------------------#
# Detect, Make, and Set the Clusters                                           #
#------------------------------------------------------------------------------#

#detectedCoreCount <- detectCores()

#parallelClusters <- makeCluster(detectedCoreCount)

#setDefaultCluster(parallelClusters)

##stopCluster(parallelClusters)



######################## #
#### Import Functions ####
######################## #-----------------------------------------------------#
# Import ATS, helper, and other functions                                      #
#------------------------------------------------------------------------------#

# Analyst's Tool-Share
source(file = "ats_functions.R")

# Miscellaneous Functions, copied and pasted, willy-nilly, from the Inter-Webs
source(file = "functions.R")


######################### #
#### Load the Data-Set ####
######################### #----------------------------------------------------#
# Load the complete data set, while replacing any missing values with NA       #
#------------------------------------------------------------------------------#

#data_file_path <- "C:/Development/kaggle--home-credit-default-risk/data"
data_file_path <- Sys.getenv("HCDR_DATA_FILE_PATH")

data_file_name <- "application_train.csv"

#filename <- "C:/Development/kaggle--home-credit-default-risk/data/application_train.csv"
#filename = paste0(data_file_path, "/", data_file_name)
#filename = sprintf("%s/%s", data_file_path, data_file_name)

df <- read.table(
    sprintf("%s/%s", data_file_path, data_file_name),
    header = TRUE,
    sep = ",",
    na.strings = c(""),
    stringsAsFactors = FALSE
)


########################## #
#### Check the Data-Set ####
########################## #---------------------------------------------------#
#  #
#------------------------------------------------------------------------------#

# Examine the structure of the data-set
#names(df)
#dim(df)
#str(df)
#glimpse(df)
#summary(df)



######################## #
#### Fix the Data-Set ####
######################## #-----------------------------------------------------#
# Fix any issues with the initial/original/raw data-set                        #
#------------------------------------------------------------------------------#
# TODO(JamesBalcomb): Decide on the scope of coverage for this section
