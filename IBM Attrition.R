# Set-up
set.seed(123)
library(stargazer)
library(MLmetrics)
library(glmnet)
library(glmnetUtils)
library(tree)
library(randomForest)
library(gbm)
library(ROCR)
library(pROC)
setwd("C:/Users/Bret/OneDrive/R Projects/IBM Attrition/")

# Importing Data:
data.hr <- read.csv("./WA_Fn-UseC_-HR-Employee-Attrition.csv", stringsAsFactors = TRUE, 
                    fileEncoding="UTF-8-BOM")
data.hr <- within(data.hr,
                  rm(EmployeeCount, EmployeeNumber, Over18, StandardHours))
summary(data.hr)


# Data Cleaning and Set up
test <- sample(1:1470, 470, replace = FALSE)
train <- (1:1470)[-test]
y.test <- data.hr$Attrition[test]
n.obs <- nrow(data.hr)
n.var <- ncol(data.hr)

# LASSO MODEL:
mod.lasso <- glmnet(Attrition ~ ., data=data.hr[train,], alpha=1, family='binomial')
plot(mod.lasso, "lambda")

mod.lasso.cv <- cv.glmnet(Attrition ~ ., data=data.hr[train,], alpha=1, family='binomial',
                          nfolds=10)
plot(mod.lasso.cv)

lambda.best <- mod.lasso.cv$lambda.1se
lasso.best <- glmnet(Attrition ~., data=data.hr[train,], alpha=1, family='binomial',
                     lambda=lambda.best)

pred.lasso <- predict(lasso.best, newdata=data.hr[test,], type='response')
plot(pred.lasso)


# RandomForest:
mod.forest <- randomForest(Attrition ~ ., data=data.hr, subset=train, ntree=1000)
plot(mod.forest)

n.tree.max <- 1000
mat.err <- matrix(NA, nrow=n.tree.max, ncol=n.var)
for(i in 1:n.var){
  mod <- randomForest(Attrition ~., data=data.hr, subset=train, ntree=n.tree.max, 
                      mtry=i)
  mat.err[,i] <- mod$err.rate[,1]
}
ts.plot(mat.err, col=1:n.var, xlab='# of Trees', ylab='OOB Error') 
pos.best <- which(mat.err == min(mat.err), arr.ind=TRUE) 
# Several possible mins. Best mtry is 16 (for all), 
# and best ntree is between 500 and 517 (using 500 for parsimony).
ntree.best <- pos.best[1,1]
mtry.best <- pos.best[1,2]

forest.best <- randomForest(Attrition ~., data=data.hr, subset=train, ntree=ntree.best,
                            mtry=mtry.best)

pred.forest <- predict(forest.best, newdata=data.hr[test,], type='prob')
plot(pred.forest[,2]) 



# Boosting:
data.hr$Attrition <- as.numeric(data.hr$Attrition)
data.hr$Attrition <- data.hr$Attrition-1
mod.boost <- gbm(Attrition ~., distribution = "bernoulli", data=data.hr[train,], n.trees = 10000,
                 interaction.depth = 2, bag.fraction = 1, train.fraction = 1, cv.folds=0)


n.tree.max <- 10000
oob.boost <- rep(NA, n.var)
trees.best <- rep(NA, n.var)
  for(i in 1:n.var){
   mod <- gbm(Attrition ~., distribution='bernoulli', data=data.hr[train,],
              n.trees=n.tree.max, interaction.depth = i, shrinkage=0.001,
              verbose=FALSE)
   trees.best[i] <- gbm.perf(mod, plot.it=TRUE, oobag.curve = FALSE, method='OOB')
   oob.boost[i] <- mod$train.error[trees.best[i]]
}
d.best <- which.min(oob.boost)
n.tree.best <- trees.best[d.best]

d.best <- 23
n.tree.best <- 2288

boost.best <- gbm(Attrition ~., distribution = 'bernoulli', data=data.hr[train,],
                  n.trees=n.tree.best, interaction.depth = d.best, shrinkage=0.001)
pred.boost <- predict(boost.best, newdata=data.hr[test,], type='response')
plot(pred.boost)

# Comparing Models:
y.test <- as.data.frame(y.test)
y.test$y.test <- as.numeric(y.test$y.test)
y.test$lasso <- as.numeric(pred.lasso)
y.test$rf <- as.numeric(pred.forest[,2])
y.test$boost <- as.numeric(pred.boost)

ROC.lasso <- roc(y.test, y.test, lasso)
ROC.rf <- roc(y.test, y.test, rf)
ROC.boost <- roc(y.test, y.test, boost)

ggroc(list(ROC.lasso, ROC.rf, ROC.boost))
auc <- matrix(c(ROC.lasso$auc, ROC.rf$auc, ROC.boost$auc), nrow=1, ncol=3)
colnames(auc) <- c("Lasso", "Random Forest", "Boost")
auc


# Most important predictors?
lasso <- as.data.frame(lasso.best$beta[1:51,1])
lasso <- lasso[order(lasso$`lasso.best$beta[1:51, 1]`), , drop=FALSE]
head(lasso)
varImpPlot(forest.best, cex=0.5)
summary.gbm(boost.best, plotit=FALSE)[1:5,1]