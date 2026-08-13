
library(catboost) 
library(extraTrees) 
library(DMwR)
library(pbapply)
library(rlang)
library(tidyverse)
library(reshape2)
library(openxlsx)
library(DALEX)
library(readr)
library(gbm) 
library(kknn)
library(dplyr)
library(caret)
library(ggplot2)
library(pROC)
library(rms)
library(rmda)
library(dcurves)
library(Hmisc)
library(ResourceSelection)
library(DynNom)
library(survey)
library(caret)
library(foreign)
library(plotROC)
library(survival)
library(shapper)
library(iml)
library(e1071)
library(ROCR)
library(corrplot)
library(lattice)
library(Formula)
library(SparseM)
library(survival)
library(riskRegression)
library(pheatmap) 
library(fastshap)
library(naivebayes)
library(ingredients)
library(mlr3)
library(table1)
library(tableone)
library(adabag)
library(RColorBrewer)
library(VIM)
library(mice)
library(autoReg)
library(cvms)
library(tibble)
library(plotROC)
library(pROC)
library(ggplot2)
library(cvms)
library(tibble)
library(corrplot)
library(data.table)
library(pheatmap)
library(ComplexHeatmap)
library(RColorBrewer)
library(circlize)
library(ROSE)
library(scales)
library(lightgbm) 
library(plotROC)
library(pROC)
library(ggplot2)
library(kernelshap)  
library(shapviz) 
-----------------------------------------------------------------------------

setwd("D:\\SEER")

data=read.csv("data.csv",header = T,encoding = "GBK")

data$Result = factor(data$Result,levels = c(0,1),labels = c('No','Yes'))  

data$Sex = factor(data$Sex,levels = c(1,2),
                     labels = c('Female','Male'))
data$Race = factor(data$Race,levels = c(1,2,3),
                   labels = c('Black','Other','White'))
data$Marital = factor(data$Marital,levels = c(1,2),
                      labels = c('Married','Unmarried'))
data$Site = factor(data$Site,levels = c(1,2,3,4),
                         labels = c('Body','Head','Other','Tail'))
data$Grade = factor(data$Grade,levels = c(1,2,3),
                     labels = c('Well','Moderately','Poorly'))
data$Income = factor(data$Income,levels = c(1,2,3),
                             labels = c('Low','Middle','High'))
data$Residence = factor(data$Residence,levels = c(1,2),
                             labels = c('Rural','Urban'))
data$Year = factor(data$Year,levels = c(1,2,3),
                    labels = c('2000-2007','2008-2015','2016-2023'))

data$Histology = factor(data$Histology,levels = c(1,2,3,4),
                   labels = c('Atypical carcinoid tumor','Carcinoid tumor',
                              'Neuroendocrine carcinoma','Other'))
----------------------------------------------------------------------------
set.seed(20260509)
inTrain <- createDataPartition(data$Result, p = 0.7, list = FALSE)

train <- data[inTrain, ]
test  <- data[-inTrain, ]

ini <- mice(train, maxit = 0, printFlag = FALSE)

meth <- ini$method
pred <- ini$predictorMatrix

meth["Result"] <- ""

imp_train <- mice(train,
                  m = 5,
                  method = meth,
                  predictorMatrix = pred,
                  seed = 20260509,
                  printFlag = FALSE)

train_imp <- complete(imp_train, 1)

imp_test <- mice.mids(imp_train,newdata = test)

test_imp <- complete(imp_test, 1)

write.csv(train_imp, "dev.csv", row.names = FALSE)
write.csv(test_imp,"vad.csv", row.names = FALSE)

-----------------------------------------------------------------------
x = train_imp
x1 = colnames(x[,11:12])
x2 = colnames(x[,2:10])

CreateTableOne(data=x)

myVars = colnames(x[,2:ncol(x)])

catVars = colnames(x[,2:10])

# VIF
library(car)
temp_model <- glm(Result ~ Residence+Grade+Histology+
                    Marital+Income+Age+Year+Race+Sex+Site+Size,
                  data = x, family = binomial)

vif_result <- vif(temp_model)
print(vif_result)


colnames(x)

var=c("Result",
      "Year","Race","Site","Grade","Histology","Size",
      "Marital","Sex","Income","Age","Residence")

dev = train_imp
vad = test_imp

dev = dev[,var]
vad = vad[,var]
dev$Result = factor(as.character(dev$Result))

models = c("glm","svmRadial","gbm","nnet","extraTrees",
           "xgbTree","kknn","AdaBoost.M1")

models_names = list(Logistic="glm",SVM="svmRadial",GBM="gbm",
                    NeuralNetwork="nnet",RandomForest="extraTrees",
                    Xgboost="xgbTree",KNN="kknn",Adaboost="AdaBoost.M1")#

glm.tune.grid = NULL
svm.tune.grid = expand.grid(sigma = c(0.0001, 0.001, 0.01, 0.1), 
                            C = seq(0.05, 0.5, 0.05))

gbm.tune.grid = expand.grid(n.trees = 100, 
                            interaction.depth = c(2, 3, 5),  
                            shrinkage = c(0.01, 0.1), 
                            n.minobsinnode = 5)

nnet.tune.grid = expand.grid(size = c(3, 5, 7, 10, 15),  
                             decay = c(0.1, 0.6, 1.0))   

rf.tune.grid = expand.grid(mtry = c(2, 5, 8, 11, 15, 20),  
                           numRandomCuts = 3)

xgb.tune.grid = expand.grid(nrounds = 10, 
                            max_depth = c(3, 5, 7),  
                            eta = c(0.3, 0.1, 0.01, 0.001, 0.0001),  
                            gamma = 0.5,
                            colsample_bytree = 0.5,
                            min_child_weight = 1,
                            subsample = 0.6)

knn.tune.grid = expand.grid(kmax = c(3:15), 
                            distance = 1,
                            kernel = "optimal")

ada.tune.grid = expand.grid(mfinal = 2,
                            maxdepth = c(1, 2, 3, 5),  
                            coeflearn = "Zhu")

Tune_table = list(glm = glm.tune.grid,
                  svmRadial = svm.tune.grid,
                  gbm = gbm.tune.grid,
                  nnet = nnet.tune.grid,
                  extraTrees = rf.tune.grid,
                  xgbTree = xgb.tune.grid,
                  kknn = knn.tune.grid,
                  AdaBoost.M1 = ada.tune.grid)

train_probe = data.frame(Result = dev$Result)
test_probe = data.frame(Result = vad$Result)

importance = list()

ML_calss_model = list()

set.seed(520)
train.control <- trainControl(method = 'repeatedcv',
                              number = 10, 
                              repeats = 5, 
                              classProbs = TRUE, 
                              summaryFunction = twoClassSummary)
pb = txtProgressBar(min = 0, max = length(models), style = 3)
for (i in seq_along(models)) {
  model <- models[i]
  model_name <- names(models_names)[which(models_names == model)]  
  set.seed(52)
  fit = train(Result~.,
              data = dev,
              tuneGrid = Tune_table[[model]],
              metric='ROC',
              method= model,
              trControl=train.control)
  
  train_Pro = predict(fit, newdata = dev, type = 'prob')
  test_Pro = predict(fit, newdata = vad, type = 'prob')
  
  train_probe[[model_name]] <- train_Pro$Yes
  test_probe[[model_name]] <- test_Pro$Yes
  
  ML_calss_model[[model_name]] = fit  
  importance[[model_name]] = varImp(fit, scale = TRUE)  
  
  setTxtProgressBar(pb, i)
}
close(pb)  

----------------------------------------------------------------------------
#9.LightGBM
train = dev
train$Result = ifelse(train$Result == "Yes", 1, 0)
test = vad[, var]
test$Result = ifelse(test$Result == "Yes", 1, 0)

dtrain = lgb.Dataset(as.matrix(train[, 2:ncol(train)]), label = train$Result)

lgb.tune.grid = expand.grid(
  learning_rate = c(0.01, 0.1, 0.5, 1.0),   
  min_data      = c(1, 5, 10),               
  num_threads   = 2L,                        
  stringsAsFactors = FALSE)

set.seed(123)
cv_results = data.frame()

for (i in 1:nrow(lgb.tune.grid)) {
  params_i = list(
    objective = "binary",
    metric = "auc",
    learning_rate = lgb.tune.grid$learning_rate[i],
    min_data = lgb.tune.grid$min_data[i],
    num_threads = lgb.tune.grid$num_threads[i],
    force_col_wise = TRUE)
  
  cv_fit = lgb.cv(
    params = params_i,
    data = dtrain,
    nrounds = 20,                 
    nfold = 10,
    stratified = TRUE,
    early_stopping_rounds = 3L,   
    verbose = -1)
  
  cv_results = rbind(cv_results, data.frame(
    learning_rate = params_i$learning_rate,
    min_data      = params_i$min_data,
    best_iter     = cv_fit$best_iter,
    auc           = cv_fit$best_score
  ))
}

best_params = cv_results[which.max(cv_results$auc), ]
print(best_params)

final_params = list(
  objective = "binary",
  metric = "auc",
  learning_rate = best_params$learning_rate,
  min_data = best_params$min_data,
  num_threads = 2L,
  force_col_wise = TRUE)

dtest = lgb.Dataset.create.valid(dtrain, as.matrix(test[, 2:ncol(test)]), 
                                 label = test$Result)
valids = list(test = dtest)

lightgbm_model = lgb.train(
  params = final_params,
  data = dtrain,
  nrounds = best_params$best_iter,   
  valids = valids,
  early_stopping_rounds = 3L)

train_probe$LightGBM = predict(lightgbm_model, data = as.matrix(dev[, 2:ncol(dev)]))
test_probe$LightGBM  = predict(lightgbm_model, data = as.matrix(vad[, 2:ncol(vad)]))

----------------------------------------------------------------------------
#10.CatBoost
train = dev
train$Result = ifelse(train$Result == "Yes", 1, 0)
train = as.data.frame(lapply(train, function(x) {
  if (is.integer(x)) return(as.numeric(x))
  return(x)}))

test = vad[, var]
test$Result = ifelse(test$Result == "Yes", 1, 0)
test = as.data.frame(lapply(test, function(x) {
  if (is.integer(x)) return(as.numeric(x))
  return(x)}))

train_pool = catboost.load_pool(as.matrix(train[, 2:ncol(train)]), label = train$Result)
test_pool  = catboost.load_pool(as.matrix(test[, 2:ncol(test)]),  label = test$Result)

cb.tune.grid = expand.grid(
  depth         = c(3, 5, 7),
  learning_rate = c(0.01, 0.03, 0.1),
  border_count  = c(32, 64, 128),
  stringsAsFactors = FALSE)

set.seed(123)
cv_results = data.frame()

for (i in 1:nrow(cb.tune.grid)) {
  params_i = list(
    loss_function = "Logloss",
    eval_metric = "AUC",
    iterations = 100,
    depth = cb.tune.grid$depth[i],
    learning_rate = cb.tune.grid$learning_rate[i],
    border_count = cb.tune.grid$border_count[i],
    ignored_features = c(4, 9),
    random_seed = 123)
  
  cv_fit = catboost.cv(
    pool = train_pool,
    params = params_i,
    fold_count = 10,
    type = "Classical",
    partition_random_seed = 123)
  
  best_auc = max(cv_fit$test.AUC.mean, na.rm = TRUE)
  best_iter = which.max(cv_fit$test.AUC.mean)
  
  cv_results = rbind(cv_results, data.frame(
    depth = params_i$depth,
    learning_rate = params_i$learning_rate,
    border_count = params_i$border_count,
    best_iter = best_iter,
    auc = best_auc
  ))
}

best_params = cv_results[which.max(cv_results$auc), ]
print(best_params)

final_fit_params = list(
  iterations = 100,
  use_best_model = TRUE,
  eval_metric = 'AUC',
  ignored_features = c(4, 9),
  border_count = best_params$border_count,
  depth = best_params$depth,
  learning_rate = best_params$learning_rate,
  random_seed = 123)

Catboost_model = catboost.train(train_pool, test_pool, final_fit_params)
Catboost_model

train_probe$CatBoost = catboost.predict(Catboost_model, train_pool,
                                        prediction_type = 'Probability')
test_probe$CatBoost = catboost.predict(Catboost_model, test_pool,
                                       prediction_type = 'Probability')

-----------------------------------------------------------------------------
# PR
library(PRROC)
library(ggplot2)
library(dplyr)


model_cols = setdiff(colnames(test_probe), "Result")  

pr_list = list()
auprc_table = data.frame(Model = character(), AUPRC = numeric())

for (model in model_cols) {
  scores_pos = test_probe[[model]][test_probe$Result == "Yes"]  
  scores_neg = test_probe[[model]][test_probe$Result == "No"]   
  
  pr = pr.curve(scores.class0 = scores_pos,
                scores.class1 = scores_neg,
                curve = TRUE)
  
  pr_df = as.data.frame(pr$curve)
  colnames(pr_df) = c("Recall", "Precision", "Threshold")
  pr_df$Model = model
  
  pr_list[[model]] = pr_df
  auprc_table = rbind(auprc_table,
                      data.frame(Model = model, AUPRC = pr$auc.integral))
}

pr_all = do.call(rbind, pr_list)

baseline = mean(test_probe$Result == "Yes")

auprc_table$Label = sprintf("%s (AUPRC = %.3f)", auprc_table$Model, auprc_table$AUPRC)
pr_all = merge(pr_all, auprc_table[, c("Model", "Label")], by = "Model")

p = ggplot(pr_all, aes(x = Recall, y = Precision, color = Label)) +
  geom_line(linewidth = 0.8) +
  geom_hline(yintercept = baseline, linetype = "dashed", color = "grey40") +
  annotate("text", x = 0.85, y = baseline + 0.02,
           label = paste0("Baseline = ", round(baseline, 3)),
           size = 3, color = "grey40") +
  labs(x = "Recall", y = "Precision", color = "Model",
       title = "Precision-Recall Curves (Test Set)") +
  theme_bw() +
  theme(legend.position = "right",
        legend.text = element_text(size = 8))

print(p)
ggsave("PR_curve_test.pdf", plot = p, width = 8, height = 6)

write.csv(auprc_table[, c("Model", "AUPRC")], "AUPRC_test.csv", row.names = FALSE)
print(auprc_table[, c("Model", "AUPRC")])



# PR
model_cols = setdiff(colnames(train_probe), "Result")  # 排除Result列，只留各模型概率

pr_list = list()
auprc_table = data.frame(Model = character(), AUPRC = numeric())

for (model in model_cols) {
  scores_pos = train_probe[[model]][train_probe$Result == "Yes"]  # 阳性组预测概率
  scores_neg = train_probe[[model]][train_probe$Result == "No"]   # 阴性组预测概率
  
  pr = pr.curve(scores.class0 = scores_pos,
                scores.class1 = scores_neg,
                curve = TRUE)
  
  pr_df = as.data.frame(pr$curve)
  colnames(pr_df) = c("Recall", "Precision", "Threshold")
  pr_df$Model = model
  
  pr_list[[model]] = pr_df
  auprc_table = rbind(auprc_table,
                      data.frame(Model = model, AUPRC = pr$auc.integral))
}

pr_all = do.call(rbind, pr_list)

baseline = mean(train_probe$Result == "Yes")

auprc_table$Label = sprintf("%s (AUPRC = %.3f)", auprc_table$Model, auprc_table$AUPRC)
pr_all = merge(pr_all, auprc_table[, c("Model", "Label")], by = "Model")

p = ggplot(pr_all, aes(x = Recall, y = Precision, color = Label)) +
  geom_line(linewidth = 0.8) +
  geom_hline(yintercept = baseline, linetype = "dashed", color = "grey40") +
  annotate("text", x = 0.85, y = baseline + 0.02,
           label = paste0("Baseline = ", round(baseline, 3)),
           size = 3, color = "grey40") +
  labs(x = "Recall", y = "Precision", color = "Model",
       title = "Precision-Recall Curves (Train Set)") +
  theme_bw() +
  theme(legend.position = "right",
        legend.text = element_text(size = 8))

print(p)
ggsave("PR_curve_train.pdf", plot = p, width = 8, height = 6)

write.csv(auprc_table[, c("Model", "AUPRC")], "AUPRC_train.csv", row.names = FALSE)
print(auprc_table[, c("Model", "AUPRC")])

------------------------------------------------------------------------------
# DCA、ROC、校准曲线
models_names = list(Logistic="glm", SVM="svmRadial", GBM="gbm",
                    NeuralNetwork="nnet", RandomForest="extraTrees",
                    Xgboost="xgbTree", KNN="kknn", Adaboost="AdaBoost.M1",
                    LightGBM = "LightGBM", CatBoost = "CatBoost")

Train = train_probe
Test = test_probe

cutpoint = 10  

datalist = list(Train = train_probe, Test = test_probe)

model_colors <- c("Logistic" = "#1f77b4", "SVM" = "#ff7f0e", "GBM" = "#2ca02c",
                  "NeuralNetwork" = "#d62728", "RandomForest" = "#9467bd",
                  "Xgboost" = "#8c564b", "KNN" = "#e377c2", "Adaboost" = "#7f7f7f",
                  "LightGBM" = "#bcbd22", "CatBoost" = "#17becf")

for (newdata_tt in names(datalist)) {
  
  newdata = datalist[[newdata_tt]]
  
# ========================================
  cat("\n绘制", newdata_tt, "校准曲线...\n")
  
  available_models <- intersect(names(models_names), colnames(newdata))
  if(length(available_models) == 0) {
    cat("警告：没有找到任何模型预测列！\n")
    next
  }
  
  caldata_list <- list()
  
  for(model_name in available_models) {
    preds <- as.numeric(newdata[, model_name])
    obs <- ifelse(newdata$Result == "Yes", 1, 0)
    
    breaks <- seq(0, 1, length.out = cutpoint + 1)
    
    preds_bin <- cut(preds, breaks = breaks, include.lowest = TRUE, 
                     labels = FALSE)
    
    for(i in 1:(length(breaks)-1)) {
      idx <- which(preds_bin == i)
      if(length(idx) >= 5) {  
        caldata_list[[length(caldata_list) + 1]] <- data.frame(
          calibModelVar = model_name,
          midpoint = mean(preds[idx], na.rm = TRUE),
          Percent = mean(obs[idx], na.rm = TRUE),
          n = length(idx),
          stringsAsFactors = FALSE
        )
      }
    }
  }
  
  caldata <- do.call(rbind, caldata_list)
  
  caldata$Percent <- pmax(pmin(caldata$Percent, 1), 0)
  caldata$midpoint <- pmax(pmin(caldata$midpoint, 1), 0)
  
  cat("校准数据点数:", nrow(caldata), "\n")
  if(nrow(caldata) > 0) {
    cat("midpoint分布:\n")
    print(summary(caldata$midpoint))
    cat("\nPercent分布:\n")
    print(summary(caldata$Percent))
    
    Calibrat_plot <- ggplot(caldata, aes(x = midpoint, y = Percent, 
                                         colour = calibModelVar,
                                         group = calibModelVar)) +
      geom_point(aes(size = n), alpha = 0.7) +  
      geom_smooth(method = "loess", se = FALSE,  
                  linewidth = 1, span = 0.9) +   
      geom_abline(slope = 1, intercept = 0, linetype = "dashed", 
                  colour = "grey40", linewidth = 0.8) +
      scale_x_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.2), 
                         labels = scales::number_format(accuracy = 0.1),
                         expand = c(0.02, 0.02)) +
      scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.2), 
                         labels = scales::number_format(accuracy = 0.1),
                         expand = c(0.02, 0.02)) +
      scale_size_continuous(range = c(2, 6), guide = "none") +  
      scale_color_manual(values = model_colors, 
                         breaks = intersect(names(model_colors), 
                                            unique(caldata$calibModelVar))) +
      labs(x = "Predicted Probability", 
           y = "Observed Probability",
           color = NULL, 
           title = paste(newdata_tt, "Calibration Curves")) +
      theme_bw(base_size = 12) +
      theme(
        plot.title = element_text(hjust = 0.5, size = 15, face = "bold"),
        panel.grid = element_line(color = "grey95", linewidth = 0.3),
        panel.border = element_rect(colour = "black", linewidth = 0.8),
        axis.text = element_text(size = 11, face = "bold", color = "black"),
        axis.title = element_text(size = 13, face = "bold", color = "black"),
        legend.position = c(0.25, 0.85),  
        legend.background = element_rect(fill = "white", color = "black", 
                                         linewidth = 0.5),
        legend.text = element_text(size = 8),
        legend.title = element_blank(),
        legend.key.size = unit(0.8, "lines"),
        legend.spacing.y = unit(0.1, "lines"),
        legend.margin = margin(5, 5, 5, 5),
        legend.box.background = element_rect(fill = "white", color = NA),
        plot.margin = margin(15, 15, 15, 15)
      ) +
      coord_fixed(ratio = 1) +
      guides(color = guide_legend(ncol = 1, byrow = TRUE))  
    
    ggsave(paste0(newdata_tt, "_Calibration.pdf"), Calibrat_plot, 
           width = 8, height = 8, device = "pdf", family = "serif")
    
    Calibrat_facet <- ggplot(caldata, aes(x = midpoint, y = Percent)) +
      geom_point(aes(size = n), alpha = 0.7, color = "#2c3e50") +
      geom_smooth(method = "loess", se = TRUE, alpha = 0.2, 
                  linewidth = 1, color = "#e74c3c") +
      geom_abline(slope = 1, intercept = 0, linetype = "dashed", 
                  colour = "grey40", linewidth = 0.6) +
      facet_wrap(~ calibModelVar, scales = "free", ncol = 4) +
      scale_x_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.5)) +
      scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.5)) +
      scale_size_continuous(range = c(1, 4), guide = "none") +
      labs(x = "Predicted Probability", 
           y = "Observed Probability",
           title = paste(newdata_tt, "Individual Calibration Curves")) +
      theme_bw(base_size = 10) +
      theme(
        plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
        strip.text = element_text(size = 10, face = "bold"),
        strip.background = element_rect(fill = "lightblue"),
        panel.grid = element_line(color = "grey95", linewidth = 0.2)
      )
    
    ggsave(paste0(newdata_tt, "_Calibration_Facet.pdf"), Calibrat_facet, 
           width = 12, height = 10, device = "pdf", family = "serif")
    
    cat("校准曲线已保存\n")
  } else {
    cat("警告：没有有效的校准数据！\n")
  }
  
  
  cat("计算", newdata_tt, "模型评估指标...\n")
  
  ROC_list = list()
  ROC_label = list()
  Evaluation_metrics = data.frame()
  
  for (model_name in available_models) {
    
    if(length(unique(newdata[, model_name])) < 2) {
      cat("警告：", model_name, "预测值没有变化，跳过！\n")
      next
    }
    
    tryCatch({
      ROC = roc(response = newdata$Result, 
                predictor = newdata[, model_name],
                levels = c("No", "Yes"),
                direction = "<")
      AUC = round(auc(ROC), 3)
      CI = ci.auc(ROC)
      
      label = paste0(model_name, " (AUC=", sprintf("%0.3f", AUC), 
                     ", 95%CI: ", sprintf("%0.3f", CI[1]), 
                     "-", sprintf("%0.3f", CI[3]), ")")
      
      bestp = ROC$thresholds[which.max(ROC$sensitivities + ROC$specificities - 1)]
      
      obs = ifelse(newdata$Result == "Yes", 1, 0)
      pred_prob = as.numeric(newdata[, model_name])
      
      epsilon <- 1e-6
      pred_prob[pred_prob < epsilon] <- epsilon
      pred_prob[pred_prob > (1 - epsilon)] <- (1 - epsilon)
      
      Brier = mean((pred_prob - obs)^2, na.rm = TRUE)
      
      tryCatch({
        set.seed(123)
        noise <- rnorm(length(pred_prob), 0, 1e-6)
        pred_prob_noise <- pred_prob + noise
        pred_prob_noise <- pmax(pmin(pred_prob_noise, 1-epsilon), epsilon)
        
        cal_model = glm(obs ~ qlogis(pred_prob_noise), 
                        family = binomial(link = "logit"))
        Intercept = coef(cal_model)[1]
        Slope = coef(cal_model)[2]
      }, error = function(e) {
        Intercept <- NA
        Slope <- NA
      })
      
      predlab = as.factor(ifelse(newdata[, model_name] > bestp, "Yes", "No"))
      
      tryCatch({
        index_table = confusionMatrix(data = predlab,
                                      reference = newdata$Result,
                                      positive = "Yes",
                                      mode = "everything")
        
        Evaluation_metrics = rbind(Evaluation_metrics, 
                                   data.frame(
                                     Model = model_name,
                                     Threshold = round(bestp, 4),
                                     Accuracy = sprintf("%0.3f", 
                                                        index_table[["overall"]][["Accuracy"]]),
                                     Sensitivity = sprintf("%0.3f", 
                                                           index_table[["byClass"]][["Sensitivity"]]),
                                     Specificity = sprintf("%0.3f", 
                                                           index_table[["byClass"]][["Specificity"]]),
                                     Precision = sprintf("%0.3f", 
                                                         index_table[["byClass"]][["Precision"]]),
                                     F1 = sprintf("%0.3f", ifelse(is.na(index_table[["byClass"]][["F1"]]), 0, 
                                                                  index_table[["byClass"]][["F1"]])),
                                     Brier = sprintf("%.3f", Brier),
                                     Intercept = sprintf("%.3f", ifelse(is.na(Intercept), 0, Intercept)),
                                     Slope = sprintf("%.3f", ifelse(is.na(Slope), 1, Slope)),
                                     stringsAsFactors = FALSE
                                   ))
      }, error = function(e) {
        cat("混淆矩阵计算失败：", model_name, " - ", e$message, "\n")
      })
      
      ROC_label[[model_name]] = label
      ROC_list[[model_name]] = ROC
      
    }, error = function(e) {
      cat("模型评估失败：", model_name, " - ", e$message, "\n")
    })
  }
  
  cat("执行DeLong检验...\n")
  mods = names(ROC_list)
  DeLong = data.frame()
  
  if(length(mods) >= 2) {
    for(i in 1:(length(mods)-1)) {
      for(j in (i+1):length(mods)) {
        tryCatch({
          tmp = roc.test(ROC_list[[mods[i]]], 
                         ROC_list[[mods[j]]], 
                         method = "delong")
          DeLong = rbind(DeLong,
                         data.frame(
                           Model1 = mods[i],
                           Model2 = mods[j],
                           AUC1 = round(auc(ROC_list[[mods[i]]]), 3),
                           AUC2 = round(auc(ROC_list[[mods[j]]]), 3),
                           Z = round(tmp$statistic, 3),
                           P = round(tmp$p.value, 4),
                           stringsAsFactors = FALSE
                         ))
        }, error = function(e) {
          cat("DeLong检验失败：", mods[i], "vs", mods[j], "\n")
        })
      }
    }
    
    if(nrow(DeLong) > 0) {
      DeLong$FDR = round(p.adjust(DeLong$P, method = "BH"), 4)
      DeLong$Significance = ifelse(DeLong$FDR < 0.001, "***",
                                   ifelse(DeLong$FDR < 0.01, "**",
                                          ifelse(DeLong$FDR < 0.05, "*", "ns")))
      
      write.csv(DeLong, paste0(newdata_tt, "_DeLong.csv"), row.names = FALSE)
    }
  }
  
  if(nrow(Evaluation_metrics) > 0) {
    write.csv(Evaluation_metrics, paste0(newdata_tt, "_Evaluation_metrics.csv"), 
              row.names = FALSE)
  }
  
  if(length(ROC_list) > 0) {
    cat("绘制", newdata_tt, "ROC曲线...\n")
    
    ROC_plot = pROC::ggroc(ROC_list, size = 0.9, legacy.axes = TRUE) +
      theme_bw(base_size = 12) +
      labs(title = paste(newdata_tt, "ROC Curves"), 
           x = "1 - Specificity", 
           y = "Sensitivity") +
      theme(
        plot.title = element_text(hjust = 0.5, size = 15, 
                                  face = "bold", color = "black"),
        axis.text = element_text(size = 11, face = "bold", color = "black"),
        axis.title = element_text(size = 13, face = "bold", color = "black"),
        legend.title = element_blank(),
        legend.text = element_text(size = 9, face = "bold"),
        legend.position = c(0.7, 0.25),
        legend.background = element_rect(fill = "white", color = "grey80"),
        panel.border = element_rect(color = "black", linewidth = 0.8),
        panel.grid = element_line(color = "grey90", linewidth = 0.3),
        plot.margin = margin(15, 15, 15, 15)
      ) +
      geom_segment(aes(x = 0, y = 0, xend = 1, yend = 1), 
                   colour = 'grey50', linetype = 'dashed', linewidth = 0.6) +
      scale_color_manual(values = model_colors[names(ROC_list)],
                         breaks = names(ROC_list),
                         labels = unlist(ROC_label[names(ROC_list)])) +
      coord_equal() +
      scale_x_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.2), 
                         expand = c(0.01, 0.01)) +
      scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.2), 
                         expand = c(0.01, 0.01))
    
    ggsave(paste0(newdata_tt, "_ROC.pdf"), ROC_plot, 
           width = 8, height = 7, device = "pdf", family = "serif")
  }
  
  if(exists("DeLong") && nrow(DeLong) > 0) {
    cat("绘制DeLong检验热图...\n")
    
    all_models = unique(c(DeLong$Model1, DeLong$Model2))
    n_models = length(all_models)
    p_matrix = matrix(NA, nrow = n_models, ncol = n_models)
    rownames(p_matrix) = all_models
    colnames(p_matrix) = all_models
    diag(p_matrix) = 1 
    
    for(k in 1:nrow(DeLong)) {
      p_matrix[DeLong$Model1[k], DeLong$Model2[k]] = DeLong$FDR[k]
      p_matrix[DeLong$Model2[k], DeLong$Model1[k]] = DeLong$FDR[k]
    }
    
    p_melt = reshape2::melt(p_matrix, na.rm = TRUE)
    colnames(p_melt) = c("Model1", "Model2", "FDR")
    
    p_melt$label = ifelse(is.na(p_melt$FDR), "", 
                          ifelse(p_melt$FDR < 0.001, "***",
                                 ifelse(p_melt$FDR < 0.01, "**",
                                        ifelse(p_melt$FDR < 0.05, "*", 
                                               sprintf("%.3f", p_melt$FDR)))))
    
    DeLong_heatmap = ggplot(p_melt, aes(x = Model1, y = Model2, fill = FDR)) +
      geom_tile(color = "white", linewidth = 0.5) +
      scale_fill_gradientn(
        colors = c("#67001f", "#b2182b", "#d6604d", "#f4a582", 
                   "#fddbc7", "#f7f7f7", "#d1e5f0", "#92c5de", 
                   "#4393c3", "#2166ac", "#053061"),
        limits = c(0, 1),
        name = "FDR p-value",
        na.value = "grey90"
      ) +
      geom_text(aes(label = label),
                size = 3.5, fontface = "bold") +
      theme_minimal(base_size = 12) +
      labs(title = paste(newdata_tt, "DeLong Test Results"),
           x = "", y = "") +
      theme(
        plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
        axis.text.x = element_text(angle = 45, hjust = 1, size = 9, face = "bold"),
        axis.text.y = element_text(size = 9, face = "bold"),
        legend.position = "right",
        legend.title = element_text(size = 10, face = "bold"),
        panel.grid = element_blank()
      ) +
      coord_fixed()
    
    ggsave(paste0(newdata_tt, "_DeLong_Heatmap.pdf"), DeLong_heatmap, 
           width = 10, height = 8, device = "pdf", family = "serif")
  }
  
  # ==================== DCA ====================
  cat("绘制", newdata_tt, "DCA曲线...\n")
  
  dca_data = newdata
  dca_data$Result = ifelse(dca_data$Result == "Yes", 1, 0)
  
  DCA_list = list()
  for (model_name in available_models) {
    dca_formula = as.formula(paste("Result ~", model_name))
    tryCatch({
      set.seed(123)
      dca_curvers = decision_curve(dca_formula,
                                   data = dca_data,
                                   study.design = "cohort",
                                   bootstraps = 500)
      DCA_list[[model_name]] = dca_curvers
    }, error = function(e) {
      cat("DCA计算失败：", model_name, " - ", e$message, "\n")
    })
  }
  
  if(length(DCA_list) > 0) {
    dca = setNames(DCA_list, names(DCA_list))
    
    pdf(paste0(newdata_tt, "_DCA.pdf"), width = 10, height = 8, family = "serif")
    
    par(mar = c(5, 5, 4, 12), 
        mgp = c(3, 1, 0),
        xaxs = "i",
        yaxs = "i")
    
    plot_decision_curve(
      dca,
      curve.names = names(DCA_list),
      cost.benefit.axis = FALSE,
      confidence.intervals = "none",
      lwd = 2.5,
      xlim = c(0, 1),
      ylim = c(0, 1),
      legend.position = "topright",
      col = unlist(model_colors[names(DCA_list)])
    )
    
    grid(col = "grey90", lty = "dashed", lwd = 0.5)
    
    dev.off()
  }
  
  write.csv(dca_data, paste0(newdata_tt, "_PRplot.csv"), row.names = FALSE)
  
  cat(newdata_tt, "所有分析和图形已完成！\n\n")
}

cat("\n✅ 所有分析完成！生成的文件：\n")
cat("- Calibration plots: *_Calibration.pdf (整体) 和 *_Calibration_Facet.pdf (分面)\n")
cat("- ROC curves: *_ROC.pdf\n")
cat("- DeLong test results: *_DeLong.csv\n")
cat("- DeLong heatmaps: *_DeLong_Heatmap.pdf\n")
cat("- Evaluation metrics: *_Evaluation_metrics.csv\n")
cat("- DCA curves: *_DCA.pdf\n")

-----------------------------------------------------------------------------
# SHAP
ML_calss_model$LightGBM = lightgbm_model
ML_calss_model$CatBoost = Catboost_model


n_train = nrow(dev)  
n_test = nrow(vad)  

names(models_names)

best_Model = "GBM" 

saveRDS(ML_calss_model[["GBM"]], "GBM_model.rds")

explain_kernel = kernelshap(ML_calss_model[[best_Model]], 
                            dev[1:n_train,-1], bg_X = vad[1:n_test,-1])  

set.seed(42)
km <- kmeans(vad[1:n_test, -1], centers = 50)
bg_X_small <- as.data.frame(km$centers)

explain_kernel <- kernelshap(
  ML_calss_model[[best_Model]], 
  dev[1:n_train, -1], 
  bg_X = bg_X_small)   

shap_value = shapviz(explain_kernel,X_pred = dev[1:n_train,-1],
                     interactions = TRUE) 

sv_force(shap_value$Yes, row_id = 2, size = 9) +
  labs(title = best_Model) +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"))


pdf(paste0("SHAP_", best_Model, "_importance_beeswarm.pdf"), 7, 5)
sv_importance(shap_value$Yes, kind = "beeswarm",
              viridis_args = list(begin = 0.25, end = 0.85, option = "B"),
              show_numbers = FALSE) +
  theme_bw(base_size = 9) +
  ggtitle(label = paste0("", best_Model)) +
  theme(
    plot.title         = element_text(hjust = 0.5, face = "bold", 
                                      color = "black", size = 10),
    axis.title         = element_text(size = 9),
    axis.text          = element_text(size = 8),
    legend.title       = element_text(size = 8),
    legend.text        = element_text(size = 7),
    panel.border       = element_rect(color = "black", linewidth = 0.8),
    panel.grid.minor   = element_blank())
dev.off()

pdf(paste0("SHAP_", best_Model, "_importance_bar.pdf"), 7, 5)
sv_importance(shap_value$Yes, kind = "bar",
              show_numbers = TRUE,
              fill = "#2E86AB") +
  theme_bw(base_size = 9) +
  labs(title = paste0(best_Model, " - SHAP Feature Importance"),
       x = "Mean |SHAP value|",
       y = NULL) +
  theme(
    plot.title         = element_text(hjust = 0.5, face = "bold", 
                                      color = "black", size = 10),
    axis.text.y        = element_text(face = "plain", color = "black", 
                                      size = 8),
    axis.text.x        = element_text(color = "black", size = 7.5),
    axis.title.x       = element_text(size = 8.5),
    panel.grid.major.y = element_blank(),
    panel.grid.minor   = element_blank(),
    panel.border       = element_rect(color = "black", linewidth = 0.8))
dev.off()

sv_dependence(shap_value$Yes,v = "Size",
              color = "#3b528b",
              color_var = "Age",
)+
  theme_bw()+
  ggtitle(label = paste0("",best_Model))+
  theme(plot.title = element_text(hjust = 0.5,face = "bold",color = "black"))


library(patchwork)
shap_vars <- c("Size", "Age","Site","Grade","Histology","Year")

plot_list <- lapply(shap_vars, function(v) {
  sv_dependence(shap_value$Yes, v = v, color_var = "auto") +
    theme_bw(base_size = 9) +
    labs(title = v, x = v, y = "SHAP value") +
    theme(
      plot.title        = element_text(hjust = 0.5, face = "bold",
                                       color = "black", size = 10),
      axis.text         = element_text(color = "black", size = 8),
      axis.title        = element_text(size = 9),
      legend.text       = element_text(size = 7),
      legend.title      = element_text(size = 8),
      legend.key.size   = unit(0.4, "cm"),
      panel.grid.minor  = element_blank(),
      panel.border      = element_rect(color = "black", linewidth = 0.8)
    )
})
combined_plot <- wrap_plots(plot_list, ncol = 3) +
  plot_layout(guides = "collect") &   
  theme(legend.position = "right")
pdf(paste0("SHAP_", best_Model, "_dependence.pdf"), width = 10, height = 7)
print(combined_plot)
dev.off()

#瀑布图
pdf(paste0("SHAP_", best_Model, "_waterfall.pdf"), width = 7, height = 5)
sv_waterfall(shap_value$Yes, row_id = 2,
             fill_colors = c("#f7d13d", "#a52c60")) +
  theme_bw(base_size = 9) +
  labs(title = paste0(best_Model, " - SHAP Waterfall")) +
  theme(
    plot.title   = element_text(hjust = 0.5, face = "bold",
                                color = "black", size = 10),
    axis.text    = element_text(color = "black", size = 8),
    axis.title   = element_text(size = 9),
    panel.grid.minor = element_blank(),
    panel.border = element_rect(color = "black", linewidth = 0.8))
dev.off()
------------------------------------------------------------------------------
# GBM模型风险分层（高风险 vs 低风险）
------------------------------------------------------------------------------
gbm_prob_test <- test_probe$GBM
gbm_prob_train <- train_probe$GBM

roc_gbm_train <- roc(response = train_probe$Result, predictor = gbm_prob_train)
best_idx_train <- which.max(roc_gbm_train$sensitivities + roc_gbm_train$specificities - 1)
cutoff <- roc_gbm_train$thresholds[best_idx_train]
best_sens   <- roc_gbm_train$sensitivities[best_idx_train]
best_spec   <- roc_gbm_train$specificities[best_idx_train]

cat("最优截断值（Youden，训练集确定）:", round(cutoff, 4), "\n")

vad$Risk_Group <- ifelse(gbm_prob_test >= cutoff, "High-risk", "Low-risk")
vad$Risk_Group <- factor(vad$Risk_Group, levels = c("Low-risk", "High-risk"))
vad$GBM_prob   <- gbm_prob_test

test_roc <- roc(response = vad$Result, predictor = gbm_prob_test)
sens_at_cutoff <- coords(test_roc, x = cutoff, input = "threshold", ret = "sensitivity")
spec_at_cutoff <- coords(test_roc, x = cutoff, input = "threshold", ret = "specificity")

group_table <- vad %>%
  group_by(Risk_Group) %>%
  summarise(
    N = n(),
    LNM_positive = sum(Result == "Yes"),
    LNM_rate = round(mean(Result == "Yes") * 100, 1))
print(group_table)

write.csv(group_table, "GBM_risk_group_summary.csv", row.names = FALSE)


chi_test <- chisq.test(table(vad$Risk_Group, vad$Result))
cat("卡方检验 p值:", chi_test$p.value, "\n")


p_dist <- ggplot(vad, aes(x = GBM_prob, fill = Risk_Group)) +
  geom_histogram(binwidth = 0.02, color = "white", alpha = 0.8) +
  geom_vline(xintercept = cutoff, linetype = "dashed",
             color = "black", linewidth = 0.8) +
  annotate("text", x = cutoff + 0.03, y = Inf, vjust = 1.5,
           label = paste0("Cutoff = ", round(cutoff, 3)), size = 3.5) +
  scale_fill_manual(values = c("Low-risk" = "#4575b4", "High-risk" = "#d73027")) +
  labs(title = "GBM Predicted Probability Distribution",
       x = "Predicted Probability of LNM", y = "Count", fill = "Risk Group") +
  theme_bw(base_size = 11) +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"),
        legend.position = c(0.85, 0.8),
        legend.background = element_blank(),
        panel.border = element_rect(color = "black", linewidth = 0.8))

pdf("GBM_risk_distribution.pdf", 7, 5, family = "serif")
print(p_dist)
dev.off()


colnames(testdata)
colnames(vad)


library(survminer)

vad$time   <- testdata$time
vad$status <- testdata$status
vad$CSD    <- testdata$CSD

surv_os <- survfit(Surv(time, status) ~ Risk_Group, data = vad)
p_km_os <- ggsurvplot(surv_os,
                      data = vad,
                      pval = TRUE,
                      pval.method = TRUE,
                      conf.int = TRUE,
                      risk.table = TRUE,
                      palette = c("#4575b4", "#d73027"),
                      legend.labs = c("Low-risk", "High-risk"),
                      title = "Overall Survival by GBM Risk Group",
                      xlab = "Time (months)",
                      ggtheme = theme_bw())

pdf("GBM_KM_OS.pdf", 8, 7, family = "serif")
print(p_km_os)
dev.off()


surv_css <- survfit(Surv(time, CSD) ~ Risk_Group, data = vad)
p_km_css <- ggsurvplot(surv_css, data = vad, pval = TRUE, conf.int = TRUE,
                        risk.table = TRUE, palette = c("#4575b4","#d73027"),
                        title = "Cancer-Specific Survival by GBM Risk Group",
                        xlab = "Time (months)", ggtheme = theme_bw())
pdf("GBM_KM_CSS.pdf", 8, 7, family = "serif")
print(p_km_css)
dev.off()


library(cmprsk)
library(ggplot2)

fg_model <- crr(ftime   = vad$time,
                fstatus = vad$CSD,
                cov1    = as.numeric(vad$Risk_Group) - 1,  # Low=0, High=1
                failcode = 1,
                cencode  = 0)
summary(fg_model)


cif <- cuminc(ftime   = vad$time,
              fstatus = vad$CSD,
              group   = vad$Risk_Group,
              cencode = 0)


cif_df <- do.call(rbind, lapply(names(cif)[names(cif) != "Tests"], function(nm) {
  data.frame(
    time       = cif[[nm]]$time,
    est        = cif[[nm]]$est,
    group      = sub(" \\d$", "", nm),
    event_code = sub(".* ", "", nm)
  )
}))
cif_df1 <- cif_df[cif_df$event_code == "1", ]
cif_df1$group <- factor(cif_df1$group, levels = c("Low-risk", "High-risk"))


gray_p   <- cif$Tests[1, "pv"]
p_label  <- ifelse(gray_p < 0.001, "p < 0.001",
                   paste0("p = ", round(gray_p, 3)))


timepoints <- c(0, 100, 200, 300) 

risk_table <- do.call(rbind, lapply(levels(vad$Risk_Group), function(grp) {
  grp_time <- vad$time[vad$Risk_Group == grp]
  grp_csd  <- vad$CSD[vad$Risk_Group == grp]
  n_risk   <- sapply(timepoints, function(t) sum(grp_time >= t))
  data.frame(time = timepoints, n_risk = n_risk, group = grp)
}))
risk_table$group <- factor(risk_table$group, levels = c("Low-risk", "High-risk"))


p_main <- ggplot(cif_df1, aes(x = time, y = est, color = group)) +
  geom_step(linewidth = 0.8) +
  scale_color_manual(values = c("Low-risk" = "#4575b4", "High-risk" = "#d73027")) +
  annotate("text", x = max(cif_df1$time) * 0.05, y = 0.92,
           label = p_label, hjust = 0, size = 4) +
  labs(title = "Cumulative Incidence of Cancer-Specific Death",
       x = NULL, y = "Cumulative Incidence", color = "Risk Group") +
  scale_y_continuous(limits = c(0, 1)) +
  scale_x_continuous(limits = c(0, max(timepoints))) +
  theme_bw(base_size = 11) +
  theme(plot.title       = element_text(hjust = 0.5, face = "bold"),
        legend.position  = c(0.15, 0.85),
        legend.background = element_blank(),
        panel.border     = element_rect(color = "black", linewidth = 0.8),
        panel.grid.minor = element_blank(),
        axis.text.x      = element_blank(),
        axis.ticks.x     = element_blank())

p_risk <- ggplot(risk_table, aes(x = time, y = group, 
                                 label = n_risk, color = group)) +
  geom_text(size = 3.5, fontface = "bold") +
  scale_color_manual(values = c("Low-risk" = "#4575b4", "High-risk" = "#d73027")) +
  scale_x_continuous(limits = c(0, max(timepoints)),
                     breaks = timepoints) +
  labs(x = "Time (months)", y = NULL, title = "Number at risk") +
  theme_bw(base_size = 11) +
  theme(plot.title       = element_text(size = 10, face = "bold"),
        legend.position  = "none",
        panel.border     = element_rect(color = "black", linewidth = 0.8),
        panel.grid       = element_blank(),
        axis.text.y      = element_text(color = c("#4575b4", "#d73027"),
                                        face = "bold"))

library(patchwork)
p_combined <- p_main / p_risk + plot_layout(heights = c(4, 1))

pdf("GBM_CIF_CSS.pdf", 8, 7, family = "serif")
print(p_combined)
dev.off()


write.csv(vad[, c("Result", "GBM_prob", "Risk_Group")],
          "GBM_risk_stratification.csv", row.names = FALSE)
cat("风险分层完成，结果已保存。\n")
------------------------------------------------------------------------------



  

  
  
  
  