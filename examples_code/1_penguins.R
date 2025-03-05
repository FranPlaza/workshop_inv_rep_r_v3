# Carga de paquetes necesarios para modelado y visualización.
library(tidymodels)
library(palmerpenguins)
library(esquisse)

# Creación de un gráfico para explorar la relación entre la longitud del pico, la longitud de la aleta, el sexo y el peso corporal de los pingüinos, separado por especie.
penguins %>%
  filter(!is.na(sex)) %>%
  ggplot(aes(flipper_length_mm, bill_length_mm, color = sex, size = body_mass_g)) +
  geom_point(alpha = 0.5) +
  facet_wrap(~species)

# Preparación del conjunto de datos de pingüinos, excluyendo la columna 'island'.
penguins_df <- penguins %>%
  filter(!is.na(sex)) %>%
  dplyr::select(-island)

# Establecimiento de una semilla para reproducibilidad y subdivisión de los datos en conjuntos de entrenamiento y prueba.
set.seed(123)
penguin_split = initial_split(penguins_df, strata = sex)
penguin_train = training(penguin_split)
penguin_test = testing(penguin_split)

# Creación de remuestreos bootstrap del conjunto de entrenamiento.
penguin_boot = bootstraps(penguin_train)

# Definición de modelos ----
# Especificación de un modelo de regresión logística con GLM.
glm_spec <- logistic_reg() %>%
  set_engine('glm')

# Especificación de un modelo de bosque aleatorio para clasificación.
rf_spec <- rand_forest() %>%
  set_engine('ranger') %>%
  set_mode('classification')

# Definición del flujo de trabajo ----
# Creación de un flujo de trabajo con la fórmula para predecir el sexo.
penguin_wf = workflow() %>% 
  add_formula(sex ~ .)

## Definición y ajuste del modelo GLM ----
# Ajuste del modelo de regresión logística usando remuestreos bootstrap.
glm_rs <- penguin_wf %>% 
  add_model(glm_spec) %>% 
  fit_resamples( 
    resamples = penguin_boot, 
    control = control_resamples(save_pred = TRUE)
  )

## Definición y ajuste del modelo de Bosque Aleatorio ----
# Ajuste del modelo de bosque aleatorio usando remuestreos bootstrap.
rf_rs <- penguin_wf %>% 
  add_model(rf_spec) %>% 
  fit_resamples( 
    resamples = penguin_boot, 
    control = control_resamples(save_pred = TRUE)
  )

# Evaluación de modelos
# Recolección y comparación de métricas de los modelos.
collect_metrics(glm_rs)
collect_metrics(rf_rs)

# Matriz de confusión para el modelo GLM.
glm_rs %>%
  conf_mat_resampled()

# Ajuste final y evaluación del modelo GLM en el conjunto completo de datos.
penguin_final <- penguin_wf %>%
  add_model(glm_spec) %>%
  last_fit(penguin_split)
penguin_final
