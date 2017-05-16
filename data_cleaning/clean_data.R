library(Amelia)
library(rpart)
library(ggplot2)
library(mice)
library(data.table)

source('~/Desktop/CSML/applied_ml/Applied_ML/house_prices_regression/clean_data_functions.R')

##---- Part 1 - Read data
train <- read.csv('~/Desktop/CSML/applied_ml/Applied_ML/house_prices_regression/data/train.csv', stringsAsFactors = FALSE)
y_train <- train$SalePrice
train <- subset(train, select=-SalePrice)
test <- read.csv('~/Desktop/CSML/applied_ml/Applied_ML/house_prices_regression/data/test.csv', stringsAsFactors = FALSE)
df <- rbind(train, test)  # data to be cleaned


##---- Part 2 - Deal with missing data

#### Deal with missing Pool values
# Houses that have no pool
df$PoolQC <- as.character(df$PoolQC)
df[df$PoolArea == 0, ]$PoolQC <- 'None'

# Houses that have a pool but are missing data for pool quality - set to average ('TA')
df[df$PoolArea > 0 & is.na(df$PoolQC), ]$PoolQC <- 'TA'

# Predict other missing pool values with rpart
col.pred <- c("YearBuilt","YearRemodAdd", "PoolQC", "PoolArea","WoodDeckSF","OpenPorchSF","EnclosedPorch", "X3SsnPorch",
              "ScreenPorch","ExterQual","ExterCond", "YrSold","SaleType","SaleCondition")

qlty.rpart <- rpart(as.factor(PoolQC) ~ ., data = df[!is.na(df$PoolQC),col.pred], method = "class", na.action=na.omit)
df$PoolQC[is.na(df$PoolQC)] <- predict(qlty.rpart, df[is.na(df$PoolQC),col.pred], type="class")


### Deal with other miscellaneous missing feature values - set them to 'None' if missing
misc_vars <- c('Alley', 'Fence', 'FireplaceQu', 'MiscFeature')
df <- replace_miss_with_string(df, misc_vars, "None")


### Deal with missing Basement values
col.bsmt <- c("TotalBsmtSF", "BsmtExposure", "BsmtCond", "BsmtQual","BsmtFinType1", "BsmtFinType2", "BsmtFinSF1","BsmtFinSF2", "BsmtUnfSF")
df$TotalBsmtSF[is.na(df$BsmtExposure) & is.na(df$TotalBsmtSF)] <- 0  # NA for all basement data

# Rows with zero basement area and NAs - set to 'None'
col.bsmt <- c("BsmtExposure", "BsmtCond", "BsmtQual","BsmtFinType1", "BsmtFinType2")
df[df$TotalBsmtSF == 0 & is.na(df$BsmtExposure), col.bsmt] <- 
  apply(df[df$TotalBsmtSF == 0 & is.na(df$BsmtExposure), col.bsmt], 2, function(x) x <- rep("None", nrow(df[df$TotalBsmtSF == 0 & is.na(df$BsmtExposure), ])))

# Predict other missing basement values using rpart (with basement variables and year house was built as predictor variables)
col.pred <- c("BsmtExposure", "BsmtCond", "BsmtQual","BsmtFinType1", "BsmtFinType2","TotalBsmtSF","YearBuilt")

BsmtFinType2.rpart <- rpart(as.factor(BsmtFinType2) ~ ., data = df[!is.na(df$BsmtFinType2),col.pred], method = "class", na.action=na.omit)
df$BsmtFinType2[is.na(df$BsmtFinType2)] <- as.character(predict(BsmtFinType2.rpart, df[is.na(df$BsmtFinType2),col.pred], type="class"))

BsmtQual.rpart <- rpart(as.factor(BsmtQual) ~ ., data = df[!is.na(df$BsmtQual),col.pred], method = "class", na.action=na.omit)
df$BsmtQual[is.na(df$BsmtQual)] <- as.character(predict(BsmtQual.rpart, df[is.na(df$BsmtQual),col.pred], type="class"))

BsmtCond.rpart <- rpart(as.factor(BsmtCond) ~ ., data = df[!is.na(df$BsmtCond),col.pred], method = "class", na.action=na.omit)
df$BsmtCond[is.na(df$BsmtCond)] <- as.character(predict(BsmtCond.rpart, df[is.na(df$BsmtCond),col.pred], type="class"))

BsmtExposure.rpart <- rpart(as.factor(BsmtExposure) ~ ., data = df[!is.na(df$BsmtExposure),col.pred], method = "class", na.action=na.omit)
df$BsmtExposure[is.na(df$BsmtExposure)] <- as.character(predict(BsmtExposure.rpart, df[is.na(df$BsmtExposure),col.pred], type="class"))

# Fix final basement NAs (no basement here as TotalBsmtSF = 0)
df$BsmtFinSF1[is.na(df$BsmtFinSF1)|is.na(df$BsmtFinSF2)|is.na(df$BsmtUnfSF)] <- 0
df$BsmtFinSF2[is.na(df$BsmtFinSF1)|is.na(df$BsmtFinSF2)|is.na(df$BsmtUnfSF)] <- 0
df$BsmtUnfSF[is.na(df$BsmtFinSF1)|is.na(df$BsmtFinSF2)|is.na(df$BsmtUnfSF)] <- 0
df$BsmtFullBath[df$TotalBsmtSF == 0 & is.na(df$BsmtFullBath)] <- rep(0,2)
df$BsmtHalfBath[df$TotalBsmtSF == 0 & is.na(df$BsmtHalfBath)] <- rep(0,2)
  

### Deal with missing Garage values
col.Garage <- c("GarageType", "GarageYrBlt", "GarageFinish", "GarageQual","GarageCond")

# Houses with GarageArea = 0 and no Garage Type mentioned - set all other garage relevant columns to 'None'
df[df$GarageArea == 0 & df$GarageCars==0 & is.na(df$GarageType), col.Garage] <- apply(df[df$GarageArea == 0 & df$GarageCars==0 & is.na(df$GarageType), col.Garage], 2, function(x) x <- rep("None", nrow(df[df$GarageArea == 0 & df$GarageCars==0 & is.na(df$GarageType), ])))

# Predict other missing garage variables with rpart (using existing garage variables as predictors)
col.pred <- c("GarageType", "GarageYrBlt", "GarageFinish", "GarageQual","GarageCond","YearBuilt", "GarageCars", "GarageArea")

area.rpart <- rpart(GarageArea ~ ., data = df[!is.na(df$GarageArea),col.pred], method = "anova", na.action=na.omit)
df$GarageArea[is.na(df$GarageArea)] <- round(predict(area.rpart, df[is.na(df$GarageArea),col.pred]))

cars.rpart <- rpart(GarageCars ~ ., data = df[!is.na(df$GarageCars),col.pred], method = "anova", na.action=na.omit)
df$GarageCars[is.na(df$GarageCars)] <- round(predict(cars.rpart, df[is.na(df$GarageCars),col.pred]))

blt.rpart <- rpart(as.factor(GarageYrBlt) ~ ., data = df[!is.na(df$GarageYrBlt),col.pred], method = "class", na.action=na.omit)
df$GarageYrBlt[is.na(df$GarageYrBlt)] <- as.numeric(as.character(predict(blt.rpart, df[is.na(df$GarageYrBlt),col.pred], type = "class")))

# Garages built in this year were unfinished and had average quality
df$GarageFinish[df$GarageType == "Detchd" &  df$GarageYrBlt == 1950 & is.na(df$GarageFinish)] <- "Unf"
df$GarageQual[df$GarageType == "Detchd" &  df$GarageYrBlt == 1950 & is.na(df$GarageQual)] <- "TA"
df$GarageCond[df$GarageType == "Detchd" &  df$GarageYrBlt == 1950 & is.na(df$GarageCond)] <- "TA"


### Set functional, type of sale, electrical, MSZoning, Kitchen quality, Utilities to most common if unknown
df[is.na(df$Functional), ]$Functional <- 'Typ'
df[is.na(df$SaleType), ]$SaleType <- 'WD'
df[is.na(df$Electrical), ]$Electrical <- 'SBrkr'
df[is.na(df$KitchenQual), ]$KitchenQual <- 'TA'
df[is.na(df$Utilities), ]$Utilities <- 'AllPub'


### Predicting missing MSZoning values using rpart
col.pred <- c("Neighborhood", "Condition1", "Condition2", "MSZoning")
msz.rpart <- rpart(as.factor(MSZoning) ~ ., data = df[!is.na(df$MSZoning),col.pred], method = "class", na.action=na.omit)
df$MSZoning[is.na(df$MSZoning)] <- as.character(predict(msz.rpart, df[is.na(df$MSZoning),col.pred], type = "class"))


### Clean Masonry Veneer data
# Fix mistakes in MasVnrArea
df$MasVnrArea <- ifelse(df$MasVnrArea == 1,0,df$MasVnrArea)

# Assign rows with areas > 0 but having Type as None to NA - will be fixed later
df$MasVnrType[df$MasVnrArea > 0 & df$MasVnrType == "None" & !is.na(df$MasVnrType)] <- rep(NA, nrow(df[df$MasVnrArea > 0 & df$MasVnrType == "None" & !is.na(df$MasVnrType), ]))

# Set NA areas to zero and none to type
df$MasVnrArea[is.na(df$MasVnrArea)] <-rep(0, nrow(df[is.na(df$MasVnrArea), ]))
df$MasVnrType[is.na(df$MasVnrType) & df$MasVnrArea == 0] <- rep("None", nrow(df[is.na(df$MasVnrType) & df$MasVnrArea == 0, ]))

# Assign observations with Area = 0  to Type = None
df$MasVnrType[df$MasVnrType == "BrkFace" & df$MasVnrArea == 0] <- rep("None", nrow(df[df$MasVnrType == "BrkFace" & df$MasVnrArea == 0, ]))
df$MasVnrType[df$MasVnrType == "Stone" & df$MasVnrArea == 0] <- rep("None", nrow(df[df$MasVnrType == "Stone" & df$MasVnrArea == 0, ]))

# Predict MasVnrType 
Type.rpart <- rpart(as.factor(MasVnrType) ~ MasVnrArea, data = df[!is.na(df$MasVnrType),c("MasVnrType","MasVnrArea")],method = "class", na.action=na.omit)
df$MasVnrType[is.na(df$MasVnrType)] <- as.character(predict(Type.rpart, df[is.na(df$MasVnrType),c("MasVnrType","MasVnrArea")], type="class"))


### Predict missing values of LotFrontage
col.pred <- c("MSSubClass", "MSZoning", "LotFrontage", "LotArea", "Street", "Alley", "LotShape", "LandContour", "LotConfig", "LandSlope", "BldgType", "HouseStyle", "YrSold", "SaleType", "SaleCondition")
frntage.rpart <- rpart(LotFrontage ~ ., data = df[!is.na(df$LotFrontage),col.pred], method = "anova", na.action=na.omit)
df$LotFrontage[is.na(df$LotFrontage)] <- ceiling(predict(frntage.rpart, df[is.na(df$LotFrontage),col.pred]))


### Predict missing exterior values
col.pred <- c("BldgType", "HouseStyle", "OverallQual", "OverallCond", "YearBuilt", "YearRemodAdd", "RoofStyle", "RoofMatl", "Exterior1st", "Exterior2nd", "MasVnrType", "MasVnrArea", "ExterQual", "ExterCond")

ext1.rpart <- rpart(as.factor(Exterior1st) ~ ., data = df[!is.na(df$Exterior1st),col.pred], method = "class", na.action=na.omit)
df$Exterior1st[is.na(df$Exterior1st)] <- as.character(predict(ext1.rpart, df[is.na(df$Exterior1st),col.pred], type = "class"))

ext2.rpart <- rpart(as.factor(Exterior2nd) ~ ., data = df[!is.na(df$Exterior2nd),col.pred], method = "class", na.action=na.omit)
df$Exterior2nd[is.na(df$Exterior2nd)] <- as.character(predict(ext2.rpart, df[is.na(df$Exterior2nd),col.pred], type = "class"))

# Final plot of missing data
# missmap(df, main='Missing data map', col=c('yellow', 'black'))


##---- Part 3 - clean up variable types

# Convert all character variables to string
df <- as.data.frame(unclass(df))

# Save clean data
train_clean <- df[1:length(y_train), ]
test_clean <- df[(length(y_train)+1):nrow(df), ]

##---- Part 4 - Outliers

# Outliers in area vs sale price that buck linear trend
# sale_area <- data.frame(x=df$GrLivArea[1:1460], y=y_train)
# ggplot(data=sale_area, aes(x=x, y=y)) + geom_point()
# train_frame <- df[1:1460, ]
# View(train_frame[with(train_frame, order(-df$GrLivArea)), ])

# Try removing outliers
# df <- df[-c(524, 1299), ]

train_clean$SalePrice <- y_train
write.csv(train_clean, '~/Desktop/CSML/applied_ml/Applied_ML/house_prices_regression/train_clean.csv', row.names=FALSE)
write.csv(test_clean, '~/Desktop/CSML/applied_ml/Applied_ML/house_prices_regression/test_clean.csv', row.names=FALSE)