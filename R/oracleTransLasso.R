#' Oracle Transfer Learning Lasso Estimator
#'
#' @description
#' Fits a two-stage Lasso model where information is transferred from a source dataset to a target dataset. This method estimates an initial model using the combined datasets, computes an offset based on the source data,
#' and then fits a second Lasso model on the target dataset adjusted for this offset.
#'
#' @param X0 A numeric matrix of predictors from the source domain.
#' @param y0 A response vector corresponding to `X0`.
#' @param X1 A numeric matrix of predictors from the target domain.
#' @param y1 A response vector corresponding to `X1`.
#' @param nfolds Integer. Number of cross-validation folds. Default is 5.
#' @param family Character string specifying the model family.
#' Supported values are `"gaussian"`, `"binomial"`, `"poisson"`, or `"multinomial"`.
#' Default is `"gaussian"`.
#' @param ... Additional arguments passed to [glmnet()] or [cv.glmnet()].
#'
#' @details
#' The function first fits a Lasso model on the combined dataset `(X0, y0)` and `(X1, y1)`
#' using cross-validation to select the best `lambda`. Then, it computes predicted offsets based on the estimated coefficients from the source data `X0`.
#' Finally, it fits a second Lasso model to the source data adjusted by the offset, producing a final estimate of the coefficients.
#'
#' Stratified folds are used for cross-validation when `family` is `"binomial"` or `"multinomial"`.
#'
#'
#' @returns
#' - If `family = "multinomial"`, a list of coefficient matrices, one for each class.
#' - Otherwise, a matrix of estimated coefficients.
#' @export
#'
#' @examples
#' p <- 1000
#' n0 <- 100
#' n <- 500
#'
#' X0 = matrix(rnorm(n0*p), nrow = n0, ncol = p)
#' X1 = matrix(rnorm(n*p), nrow = n, ncol = p)
#'
#' beta = runif(p, -2, 2) * sample(c(0,1), p, replace = TRUE, prob = c(0.9, 0.1))
#' beta0 = beta + runif(p, -1, 1) * sample(c(0,1), p, replace = TRUE, prob = c(0.95, 0.05))
#'
#' y0 = X0%*%beta0 + rnorm(n0)
#' y1 = X1%*%beta + rnorm(n)
#'
#' oracle_beta = oracleTransLasso(X0,y0,X1,y1)

oracleTransLasso <- function(X0, y0, X1, y1, nfolds = 5, family = "gaussian",...){
  X <- rbind(X0,X1)
  y <- c(y0,y1)
  foldid <- NULL

  if (family == "binomial") {
    # Create stratified folds
    pos_indices <- which(y == 1)
    neg_indices <- which(y == 0)
    pos_folds <- sample(rep(1:nfolds, length.out = length(pos_indices)))
    neg_folds <- sample(rep(1:nfolds, length.out = length(neg_indices)))
    foldid <- numeric(length(y))
    foldid[pos_indices] <- pos_folds
    foldid[neg_indices] <- neg_folds
  }

  if (family == "multinomial") {
    # Create stratified folds for multinomial
    unique_classes <- sort(unique(y))
    foldid <- numeric(length(y))
    for (class in unique_classes) {
      class_indices <- which(y == class)
      class_folds <- sample(rep(1:nfolds, length.out = length(class_indices)))
      foldid[class_indices] <- class_folds
    }
  }

  cv_model1 <- cv.glmnet(X,y, alpha = 1, foldid = foldid, family = family,...)
  best_lambda1 <- cv_model1$lambda.min
  best_model1 <- glmnet(X,y, alpha = 1, lambda = best_lambda1, family = family,...)

  if (family == "multinomial"){
    w <- coef(best_model1)
    n_classes <- length(w)
    w_list <- lapply(1:n_classes, function(i) as.matrix(w[[i]]))
    offset <- matrix(0, nrow = nrow(X0), ncol = n_classes)
    for (i in 1:(n_classes-1)) {
      offset[,i] <- cbind(1, X0) %*% w_list[[i]]
    }
  } else {
    w <- as.matrix(coef(best_model1))
    offset <- cbind(1,X0) %*% w
  }

  foldid <- NULL
  if (family == "binomial") {
    pos_indices <- which(y0 == 1)
    neg_indices <- which(y0 == 0)
    pos_folds <- sample(rep(1:nfolds, length.out = length(pos_indices)))
    neg_folds <- sample(rep(1:nfolds, length.out = length(neg_indices)))
    foldid <- numeric(length(y0))
    foldid[pos_indices] <- pos_folds
    foldid[neg_indices] <- neg_folds
  }

  if (family == "multinomial") {
    # Create stratified folds for multinomial
    unique_classes <- sort(unique(y0))
    foldid <- numeric(length(y0))
    for (class in unique_classes) {
      class_indices <- which(y0 == class)
      class_folds <- sample(rep(1:nfolds, length.out = length(class_indices)))
      foldid[class_indices] <- class_folds
    }
  }

  if (family %in% c("binomial","poisson","multinomial")){
    cv_model2 <- cv.glmnet(X0, y0, alpha = 1, offset = offset, foldid = foldid, family = family,...)
    best_lambda2 <- cv_model2$lambda.min
    best_model2 <- glmnet(X0,y0,alpha = 1,lambda = best_lambda2,offset = offset,family = family,...)
  } else {
    cv_model2 <- cv.glmnet(X0,y0 - offset, alpha = 1, family = family,...)
    best_lambda2 <- cv_model2$lambda.min
    best_model2 <- glmnet(X0,y0 - offset,alpha = 1,lambda = best_lambda2,family = family,...)
  }

  if (family == "multinomial"){
    delta <- coef(best_model2)
    # Convert deltas to dense matrices and store in a list
    delta_list <- lapply(1:length(delta), function(i) as.matrix(delta[[i]]))
    beta <- list()
    for (i in 1:length(w)) {
      beta[[i]] <- w_list[[i]] + delta_list[[i]]
    }
  } else {
    delta <- as.matrix(coef(best_model2))
    beta <- w + delta
  }
  return(beta)
}
