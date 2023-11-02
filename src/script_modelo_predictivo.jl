
using CSV
using DataFrames
using Random
using MLJ
using DecisionTree
using PyPlot
using Statistics

# Load the data
data_brut = CSV.File("data.csv") |> DataFrame

# Calculate 'Proporcion' and filter data
data_brut.Proporcion = data_brut.Precio ./ data_brut.Avaluo_Fiscal
data_brut = filter(row -> row.Proporcion >= 1, data_brut)

data = copy(data_brut)

# Add the required columns first
data.log_Precio = log.(data.Precio)
data.raiz_Superficie_Terreno = sqrt.(data.Superficie_Terreno)
data.raiz_Superficie_Habitacional = sqrt.(data.Superficie_Habitacional)
data.log_Avaluo_Fiscal = log.(data.Avaluo_Fiscal)
data.log_Avaluo_Fiscal_Unitario = log.(data.Avaluo_Fiscal_Unitario)
data.log_Valor_Unitario_Terreno = log.(data.Valor_Unitario_Terreno)
data.log_Valor_Unit_Const_Hab = log.(data.Valor_Unitario_Construccion_Habitacional_Promedio)



features = [
    :log_Precio,
    :raiz_Superficie_Terreno,
    :raiz_Superficie_Habitacional,
    :Agno_Construccion,
    :Antiguedad_Promedio,
    :log_Avaluo_Fiscal,
    :log_Avaluo_Fiscal_Unitario,
    :Barrio,
    :Calidad_Habitacional,
    :IPV_General,
    :Latitud,
    :Longitud,
    :Tipo_Comprador_Simple,
    :Tipo_Vendedor_Simple,
    :Trimestre,
    :log_Valor_Unitario_Terreno,
    :log_Valor_Unit_Const_Hab
]
data = data[:, features]
names(data)

coerce!(data, :Precio => Continuous, :Superficie_Terreno => Continuous, :Superficie_Habitacional => Continuous)
coerce!(data, :Barrio => Multiclass, :Tipo_Comprador_Simple => Multiclass, :Tipo_Vendedor_Simple => Multiclass, :Trimestre => Multiclass)


hot = OneHotEncoder(drop_last=false)
mach = MLJ.fit!(machine(hot, data))
data_oneHot = MLJ.transform(mach, data)



y, X = unpack(data_oneHot, ==(:log_Precio))
pos_train, pos_test = partition(collect(eachindex(y)), 0.8, shuffle=true, rng=5)



RFR = @load RandomForestRegressor pkg=DecisionTree
aaa = RFR()
aaa.n_trees = 1000
aaa.min_samples_leaf = 1
tree = machine(aaa, X, y)
MLJ.fit!(tree, rows=pos_train)

y_train = exp.(y[pos_train])
y_hat_train = exp.(MLJ.predict(tree, rows=pos_train))
y_test = exp.(y[pos_test])
y_hat_test = exp.(MLJ.predict(tree, rows=pos_test))
residuos_train = y_hat_train .- y_train
residuos_test = y_hat_test - y_test

errores_train = residuos_train ./ y_train
errores_test = residuos_test ./ y_test

errores_abs_train = abs.(errores_train)
errores_abs_test = abs.(errores_test)

RECM_train = sqrt(mean(residuos_train.^2)) # nolint
RECM_test = sqrt(mean(residuos_test.^2)) # nolint

rms(y[pos_test], MLJ.predict(tree, rows=pos_test))



coerce!(X, Count => Continuous)
XGBR = @load XGBoostRegressor 
bbb = XGBR()
bbb.num_round = 100
bbb.eta = .1
xgbm = machine(bbb, X, y)
MLJ.fit!(xgbm, rows=pos_train)

y_train = exp.(y[pos_train])
y_hat_train = exp.(MLJ.predict(xgbm, rows=pos_train))
y_test = exp.(y[pos_test])
y_hat_test = exp.(MLJ.predict(xgbm, rows=pos_test))
residuos_train = y_hat_train .- y_train
residuos_test = y_hat_test - y_test

errores_train = residuos_train ./ y_train
errores_test = residuos_test ./ y_test

errores_abs_train = abs.(errores_train)
errores_abs_test = abs.(errores_test)

RECM_train = sqrt(mean(residuos_train.^2)) # nolint
RECM_test = sqrt(mean(residuos_test.^2)) # nolint




# Set seed and split data
# training = data[pos_train, :]
# test = data[pos_test, :]
# n = size(data,1)          # 2402
# ntrain = size(training,1)      # 1921
# ntest = size(test,1)           # 481
# sd = std(data.log_Precio)   # 8126



training_data = DataFrame(training)

schema(training_data)

# Define a custom model using DecisionTree's RandomForestClassifier
@load RandomForestClassifier
# custom_rf_model = RandomForestClassifier(n_trees=1000,
#                                         max_depth=0,
#                                         min_samples_split=5,
#                                         min_samples_leaf=1,
#                                         min_purity_increase=0.0,
#                                         pkg = )

# Create a pipeline with the custom model
rf_pipeline = @pipeline(Standardizer(),
                        FeatureSelector(features=features),
                        custom_rf_model)

# Create a machine with the pipeline
mach = machine(rf_pipeline, training_data, target=target)

# Fit the model to the data
fit!(mach)




# Split the data into training and test sets
Random.seed!(2022)
split = 0.8
position = randsubseq(1:size(data, 1), split)
training = data[position, :]
X_train = Array(training[:,features])
y_train = Array(training[:,target])
test = data[setdiff(1:size(data, 1), position), :]
X_test = Array(training[:,features])
y_test = Array(test[:,target])

# Define the machine learning model
RandomForestRegressor = @load RandomForestRegressor pkg=DecisionTree
model_rf = RandomForestRegressor(
    n_trees = 2000,
    max_depth = -1,
    min_samples_split = 2,
    min_samples_leaf = 1,
    min_purity_increase = 0.0,
    n_subfeatures = 7,
    rng = MersenneTwister(2022)
)

# Create a machine for the model
mach = machine(model_rf, X_train, y_train)

# Fit the model
fit!(mach, rows = position)

# Make predictions on training and test sets
preds_train = predict(mach, rows = position)
preds_test = predict(mach, rows = setdiff(1:size(data, 1), position))

# Calculate Root Mean Squared Error (RMSE)
RECM_train_rf = sqrt(mean((preds_train .- training[!, target]) .^ 2))
println("RMSE on training data: $RECM_train_rf")

RECM_test_rf = sqrt(mean((preds_test .- test[!, target]) .^ 2))
println("RMSE on test data: $RECM_test_rf")
