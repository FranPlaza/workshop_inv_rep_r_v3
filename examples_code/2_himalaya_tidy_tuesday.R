# Paquetes ----
# Carga de las bibliotecas necesarias para el análisis de datos y modelado.
library(tidytuesdayR)
library(skimr)
library(themis)
library(tidyverse)
library(tidymodels)

# Datos ----

# Tenemos dos opciones:
# Opción 1: Carga de datos del proyecto TidyTuesday para la semana 39 del año 2020.
tt_data <- tidytuesdayR::tt_load(2020, week = 39)
# Uso de skim() para obtener un resumen rápido de los datos.
tt_data$members |> 
  skimr::skim() 

# Opción 2:
# Cargar directamente las bases de datos

members <- readr::read_csv('https://raw.githubusercontent.com/rfordatascience/tidytuesday/main/data/2020/2020-09-22/members.csv')
expeditions <- readr::read_csv('https://raw.githubusercontent.com/rfordatascience/tidytuesday/main/data/2020/2020-09-22/expeditions.csv')
peaks <- readr::read_csv('https://raw.githubusercontent.com/rfordatascience/tidytuesday/main/data/2020/2020-09-22/peaks.csv')


## Climbers DF ----
# Creación de un dataframe 'climbers_df' seleccionando y filtrando las columnas relevantes.
climbers_df <- tt_data$members |> 
  select(member_id, peak_name, season, year, sex, age, citizenship, expedition_role, hired, solo, oxygen_used, success, died) |> 
  filter((!is.na(sex) & !is.na(citizenship) & !is.na(peak_name) & !is.na(expedition_role)) == T) |> 
  mutate(across(where(~ is.character(.) | is.logical(.)), as.factor))

# Data Split ----
## Semilla ----
# Establecimiento de una semilla para la reproducibilidad.
set.seed(2024)
## Split inicial ----
# División inicial de los datos en conjuntos de entrenamiento y prueba.
climbers_split <- initial_split(climbers_df, prop = 0.8, strata = died)
## Conjunto de entrenamiento ----
# Creación del conjunto de entrenamiento.
train_set <- training(climbers_split)
## Conjunto de prueba ----
# Creación del conjunto de prueba.
test_set <- testing(climbers_split)
## CV ----
# Preparación para la validación cruzada.
climbers_fold <- train_set |> 
  vfold_cv(v = 10, repeats = 1, strata = died)

# Receta ----
# Creación de una receta para el preprocesamiento de los datos.
mod_recipe <- recipe(formula = died ~ ., data = train_set)
# Configuración de pasos en la receta: manejo de ID, imputación, normalización, agrupamiento de factores raros, y codificación one-hot.
mod_recipe <- mod_recipe |>
  update_role(member_id, new_role = "id") |> 
  step_impute_median(age) |> 
  step_normalize(all_numeric_predictors()) |>
  step_other(peak_name, citizenship, expedition_role, threshold = 0.05) |> 
  step_dummy(all_predictors(), -all_numeric(), one_hot = F) |> 
  step_upsample(died, over_ratio = 0.2, seed = 2023, skip = T) 
## Preparación ----
# Preparación de la receta.
mod_recipe_prepped <- prep(mod_recipe, retain = T)
## Bake ----
# Aplicación de la receta preparada a los datos.
bake(mod_recipe_prepped, new_data = NULL)

# Modelos ----
## Regresión logística glm ----
# Definición de un modelo de regresión logística con GLM.
log_cls <- logistic_reg() |> 
  set_engine('glm') |> 
  set_mode("classification")

## Regresión logística glmnet ----
# Definición de un modelo de regresión logística con penalización (glmnet).
reg_log_cls <- logistic_reg() |>
  set_args(penalty = tune(), mixture = tune()) |>  set_mode("classification") |> 
  set_engine("glmnet", family ="binomial")

## Workflow ----
# Creación de un flujo de trabajo combinando la receta y el modelo.
cls_wf <- workflow() |> 
  add_recipe(mod_recipe) |> 
  add_model(reg_log_cls)

## Parámetros ----
# Definición de una cuadrícula de parámetros para ajustar el modelo.
param_grid <- grid_regular(
  penalty(), mixture(),
  levels = c(10,10)
)

## Ajuste ----
# Inicio del proceso de ajuste del modelo.
start <- Sys.time()
# Ajuste del modelo usando la validación cruzada.
cls_wf_fit <- tune_grid(
  cls_wf, climbers_fold,
  grid = param_grid,
  metrics = metric_set(roc_auc, accuracy, sens, spec),
  control = control_grid(save_pred = T, verbose = T)
)
# Tiempo total tomado en el ajuste.
Sys.time() - start

## Desempeño ----
# Recopilación y visualización del desempeño del modelo.
cls_wf_fit |> collect_metrics(summarize = T)
cls_wf_fit |>  show_best(metric = "roc_auc", n = 3)
cls_wf_fit |>  select_best(metric = "roc_auc")

# Predicción y evaluación final del modelo.
## Finalización del flujo ----
# Finalización del flujo de trabajo con los mejores parámetros.
cls_wf_final <- cls_wf %>% 
  finalize_workflow(select_best(cls_wf_fit, metric = "roc_auc"))

## Predicción ----
# Ajuste final y evaluación del modelo en el conjunto de prueba.
cls_wf_last_fit <- cls_wf_final %>% 
  last_fit(split = climbers_split, metrics = metric_set(roc_auc, accuracy, sens, spec))

# Entrega de resultados ----
# Carga de bibliotecas adicionales para la evaluación del modelo.
library(broom)
library(tune)
# Extracción del modelo ajustado.
reg_log_cls_fit <- cls_wf_last_fit |> extract_fit_parsnip()

## Componentes del modelo ----
# Visualización de los componentes del modelo.
tidy(reg_log_cls_fit) |> glimpse()
## Diagnóstico del modelo
# Diagnóstico general del modelo.
glance(reg_log_cls_fit) |>  glimpse()
## Matriz de confusión ----
# Creación de una matriz de confusión.
collect_predictions(cls_wf_last_fit) |> 
  conf_mat(died, estimate = .pred_class)
## Métricas de evaluación ----
# Cálculo de diferentes métricas de evaluación.
metrics <- metric_set(accuracy, sens, spec)
collect_predictions(cls_wf_last_fit) |>  
  metrics(died, estimate = .pred_class)
## Curvas ROC ----
# Generación y visualización de las curvas ROC.
collect_predictions(cls_wf_fit) |> 
  group_by(id) |> 
  roc_curve(
    died, .pred_TRUE,
    event_level = "second"
  ) |> 
  autoplot()
