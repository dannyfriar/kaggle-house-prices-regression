##---- Functions for cleaning data and dealing with missing values

## Function to get column names with missing values - takes data frame
na_cols <- function(df) {
  miss_list <- colnames(train)[unlist(lapply(df, function(x) any(is.na(x))))]
  return(sort(miss_list))
}

## Function to replace missing variables in a data frame with a string - takes data frame
# adds an extra level to the categorical variable represnting NA
replace_miss_with_string <- function(df, replace_vars, replace_string) {
  for (var in replace_vars) {
    df[, var] <- as.character(df[, var])
    df[is.na(df[, var]), var] <- replace_string
    df[, var] <- factor(df[, var])
  }
  return(df)
}