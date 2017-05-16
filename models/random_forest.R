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

# # Plot all numeric variables vs SalesPrice
# for (col in num_cols) {
#   my_frame <- data.frame(x=df[1:length(y_full_train), col], y=y_full_train)
#   print(ggplot(data=my_frame, aes(x=x, y=y)) + geom_point() + labs(x=names(df)[col], y='SalePrice'))
# }

# # Try others as factors
# df$YearBuilt <- factor(df$YearBuilt)
# df$YearRemodAdd <- factor(df$YearRemodAdd)
# df$OverallCond <- factor(df$OverallCond)
# df$OverallQual <- factor(df$OverallQual)
# df$BsmtFullBath <- factor(df$BsmtFullBath)
# df$BsmtHalfBath <- factor(df$BsmtHalfBath)
# df$FullBath <- factor(df$FullBath)
# df$HalfBath <- factor(df$HalfBath)
# df$GarageCars <- factor(df$GarageCars)
# df$Fireplaces <- factor(df$Fireplaces)
# df$TotRmsAbvGrd <- factor(df$TotRmsAbvGrd)
# df$KitchenAbvGr <- factor(df$KitchenAbvGr)
# df$BedroomAbvGr <- factor(df$BedroomAbvGr)

# # Try feature engineering
# df$GarageCarsSq <- df$GarageCars^2
# df$FullBath <- df$FullBath^2
# df$FullBath <- df$FullBath^2

# Make garage year numeric if using random forest
df$GarageYrBlt <- as.numeric(df$GarageYrBlt)

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


##---- Random forest
train_set$SalePrice <- y_train
val_set$SalePrice <- y_val

to_try <- ncol(train_set) / 20
rf_model <- randomForest(SalePrice ~ ., data=train_set, ntree=1500, nodesize = 5, mtry=20)

train_pred <- predict(rf_model)
sqrt(sum((train_pred-y_train)^2)/length(y_train))

val_pred <- predict(rf_model, newdata=val_set)
sqrt(sum((val_pred-y_val)^2)/length(y_val))
# # gives roughly 0.1426292
