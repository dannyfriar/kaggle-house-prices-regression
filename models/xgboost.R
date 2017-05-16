library(plyr)
library(dplyr)
library(ggplot2)
library(data.table)
library(glmnet)
library(elasticnet)
library(lars)
library(Amelia)
library(rpart)
library(randomForest)
library(xgboost)
library(caret)

source('~/Desktop/CSML/applied_ml/Applied_ML/house_prices_regression/clean_data.R')
y_full_train <- y_train

# Change type of a few variables (better off as factors)
df$MSSubClass <- factor(df$MSSubClass)
df$MoSold <- factor(df$MoSold)
df$YrSold <- factor(df$YrSold)

# Vector of the numeric columns
num_cols <- which(sapply(df, is.numeric))
num_cols <- num_cols[2:length(num_cols)]  # remove ID

# Log transform skewed numeric variables (all numeric variables >= 0 for this dataset)
for (col in num_cols) {
  if (skewness(df[, col]) > 0.75) {
    df[, col] <- log(1+df[, col])
  }
}

y_full_train <- log(y_full_train)

# Standardise numeric variables
df[, num_cols[]] <- scale(df[, num_cols])

## Split into train, test and validation and then X and y
train <- df[1:length(y_train), ]
train <- subset(train, select=-Id)
test <- df[(length(y_train)+1):nrow(df), ]
test_id <- test$Id
test <- subset(test, select=-Id)

smp_size <- floor(0.75 * nrow(train))
train_ind <- sample(seq_len(nrow(train)), size = smp_size)
train_set <- train[train_ind, ]
val_set <- train[-train_ind, ]
y_val <- y_full_train[-train_ind]
y_train <- y_full_train[train_ind]

##---- XGBoost model - xgboost parameter descritpions - https://github.com/dmlc/xgboost/blob/master/doc/parameter.md
train_set[] <- lapply(train_set, as.numeric)
val_set[]<-lapply(val_set, as.numeric)
test[]<-lapply(test, as.numeric)

# # Choosing hyperparameters - http://www.openjems.com/grid-search-cross-validation-xgboost-r/
# xgbGrid <- expand.grid(
#   nrounds = 200,
#   max_depth = c(12, 15, 20),
#   eta = c(0.01, 0.02, 0.05),
#   gamma = c(0, 0.5, 1, 1.5),
#   colsample_bytree = c(0.7, 0.8, 0.9, 1),
#   min_child_weight = c(8, 12, 16, 20)
# )
# 
# xgbTrControl <- trainControl(
#   method = "repeatedcv",
#   number = 2,
#   repeats = 1,
#   verboseIter = TRUE,
#   returnData = FALSE,
#   allowParallel = TRUE
# )
# 
# xgbTrain <- train(
#   x = as.matrix(train_set),
#   y = y_train,
#   objective = "reg:linear",
#   trControl = xgbTrControl,
#   tuneGrid = xgbGrid,
#   method = "xgbTree"
# )
# 
# # Get the top model and its results
# head(xgbTrain$results[with(xgbTrain$results, order(RMSE)), ], 1)

# # Optimizing over other parameters - alpha and lambda
# alpha_vals <- c(0.8, 0.9, 1, 1.1, 1.2)
# lambda_vals <- c(3, 3.2, 3.5, 3.7, 4)
# 
# xg_eval_mae <- function (yhat, dtrain) {
#   y = getinfo(dtrain, "label")
#   err= mae(exp(y),exp(yhat) )
#   return (list(metric = "error", value = err))
# }
# 
# dtrain = xgb.DMatrix(as.matrix(train_set),label= y_train)
# dval = xgb.DMatrix(as.matrix(val_set))
# dtest = xgb.DMatrix(as.matrix(test))
# 
# i = 0
# results <- numeric()
# alphas <- numeric()
# lambdas <- numeric()
# 
# for (my_alpha in alpha_vals) {
#   for (my_lambda in lambda_vals) {
#     i <- i + 1
#     i_vals <- c(i, i_vals)
#     alphas <- c(alphas, my_alpha)
#     lambdas <- c(lambdas, my_lambda)
#     print_string <- paste0('Running for iteration ', i)
#     print(print_string)
#     xgb_params = list(
#       seed = 0,
#       colsample_bytree = 1,
#       subsample = 0.5,
#       eta = 0.05,
#       objective = 'reg:linear',
#       max_depth = 12,
#       alpha = my_alpha,
#       lambda = my_lambda,
#       gamma = 0,
#       min_child_weight = 10
#     )
#     gb_dt = xgb.train(xgb_params, dtrain, nrounds = 500)
#     val_pred = predict(gb_dt, dval)
#     results <- c(results, sqrt(sum((val_pred-y_val)^2)/length(y_val)))
#   }
# }
# 
# results_frame <- data.frame(alphas, lambdas, results)
# View(results_frame)
# which(results_frame$results == min(results_frame$results))

dtrain = xgb.DMatrix(as.matrix(train_set),label= y_train)
dval = xgb.DMatrix(as.matrix(val_set))
dtest = xgb.DMatrix(as.matrix(test))

#xgboost parameters
xgb_params = list(
  seed = 0,
  colsample_bytree = 1,
  subsample = 0.5,
  eta = 0.05,
  objective = 'reg:linear',
  max_depth = 12,
  alpha = 1.1,
  lambda = 4,
  gamma = 0,
  min_child_weight = 10
)


xg_eval_mae <- function (yhat, dtrain) {
  y = getinfo(dtrain, "label")
  err= mae(exp(y),exp(yhat) )
  return (list(metric = "error", value = err))
}

best_n_rounds = 1000 # try more rounds

#train data
gb_dt = xgb.train(xgb_params, dtrain, nrounds = as.integer(best_n_rounds))

train_pred = predict(gb_dt, dtrain)
sqrt(sum((train_pred-y_train)^2)/length(y_train))

val_pred = predict(gb_dt, dval)
sqrt(sum((val_pred-y_val)^2)/length(y_val))

# # Retrain on full training set and predict on test set
# train[] <- lapply(train, as.numeric)
# dtrain = xgb.DMatrix(as.matrix(train), label= c(y_train, y_full_train))
# gb_dt = xgb.train(xgb_params, dtrain, nrounds = as.integer(best_n_rounds))
# 
# test_pred <- predict(gb_dt, dtest)
# submission <- data.table(cbind(test_id, exp(test_pred)))
# colnames(submission) <- c('Id', 'SalePrice')
# write.csv(submission, '~/Desktop/CSML/applied_ml/Applied_ML/house_prices_regression/xgboost_submission.csv', row.names=FALSE)
