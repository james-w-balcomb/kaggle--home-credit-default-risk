library(readr)

Sys.getenv("DATA_FILE_PATH")

Sys.getenv("PATH")

Sys.setenv(R_TEST = "testit")
Sys.getenv("R_TEST")


application_test <- read_csv("data/application_test.csv")

#application_train <- read_csv("data/application_train.csv")
#bureau <- read_csv("data/bureau.csv")
#bureau_balance <- read_csv("data/bureau_balance.csv")
#credit_card_balance <- read_csv("data/credit_card_balance.csv")
#installments_payments <- read_csv("data/installments_payments.csv")
#POS_CASH_balance <- read_csv("data/POS_CASH_balance.csv")
#previous_application <- read_csv("data/previous_application.csv")

##sample_submission <- read_csv("data/sample_submission.csv")

