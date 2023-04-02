# Import libraries & dataset ----
library(tidyverse)
library(data.table)
library(rstudioapi)
library(skimr)
library(car)
library(h2o)
library(rlang)
library(glue)
library(highcharter)
library(lime)


raw <- fread("C:/Users/USER_1/Desktop/R/week (6)/day 1/mushrooms.csv")

names(raw) <- names(raw) %>% 
  str_replace_all(' ','_') %>% 
  str_replace_all('-','_') %>% 
  str_replace_all('/','_')

raw <- raw %>% select(class,everything())

raw %>% skim()

raw[] <- lapply(raw, function(x) gsub("'", "", x))
raw <- data.frame(lapply(raw, as.factor))

raw$class <- raw$class %>% recode(" 'e'=1 ; 'p'=0 ") %>% as_factor()


raw$class %>% table() %>% prop.table()

# --------------------------------- Modeling ----------------------------------
h2o.init()

h2o_data <- raw %>% as.h2o()


# Splitting the data ----
h2o_data <- h2o_data %>% h2o.splitFrame(ratios = 0.8, seed = 123)
train <- h2o_data[[1]]
test <- h2o_data[[2]]

target <- 'class'
features <- raw %>% select(-class) %>% names()


# Fitting h2o model ----
model <- h2o.automl(
  max_models = 1000,
  x = features, y = target,
  training_frame = train,
  validation_frame = test,
  leaderboard_frame = test,
  stopping_metric = "AUC",
  nfolds = 10, seed = 123,
  max_runtime_secs = 480)
names(raw)
model@leaderboard %>% as.data.frame() %>% view()
model@leader 

# Predicting the Test set results ----
pred <- model@leader %>% h2o.predict(test) %>% as.data.frame()

# cross validation
model@leader %>% 
  h2o.performance(test) %>% 
  h2o.find_threshold_by_max_metric('f1') -> treshold

# Confusion Matrix----

model@leader %>% 
  h2o.confusionMatrix(test) %>% 
  as_tibble() %>% 
  select("0","1") %>% 
  .[1:2,] %>% t() %>% 
  fourfoldplot(conf.level = 0, color = c("red", "darkgreen"),
               main = paste("Accuracy = ",
                            round(sum(diag(.))/sum(.)*100,1),"%"))

model@leader %>% 
  h2o.performance(test) %>% 
  h2o.metric() %>% 
  select(threshold,precision,recall,tpr,fpr) %>% 
  add_column(tpr_r=runif(nrow(.),min=0.001,max=1)) %>% 
  mutate(fpr_r=tpr_r) %>% 
  arrange(tpr_r,fpr_r) -> deep_metrics

model@leader %>% 
  h2o.performance(test) %>% 
  h2o.auc() %>% round(2) -> auc

highchart() %>% 
  hc_add_series(deep_metrics, "scatter", hcaes(y=tpr,x=fpr), color='green', name='TPR') %>%
  hc_add_series(deep_metrics, "line", hcaes(y=tpr_r,x=fpr_r), color='red', name='Random Guess') %>% 
  hc_add_annotation(
    labels = list(
      point = list(xAxis=0,yAxis=0,x=0.3,y=0.6),
      text = glue('AUC = {enexpr(auc)}'))
  ) %>%
  hc_title(text = "ROC Curve") %>% 
  hc_subtitle(text = "Model is performing much better than random guessing") 

