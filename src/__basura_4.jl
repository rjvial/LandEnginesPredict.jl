using MLJ
import DataFrames: DataFrame
using PrettyPrinting
using StableRNGs

rng = StableRNG(512)
Xraw = rand(rng, 300, 3)
y = exp.(Xraw[:,1] - Xraw[:,2] - 2Xraw[:,3] + 0.1*rand(rng, 300))
X = DataFrame(Xraw, :auto)
train, test = partition(eachindex(y), 0.7)

KNNRegressor = @load KNNRegressor
knn_model = KNNRegressor(K=10)

ensemble_model = EnsembleModel(model=knn_model, n=20)

ensemble = machine(ensemble_model, X, y)
estimates = evaluate!(ensemble, resampling=CV(), rows=train)
estimates


B_range = range(ensemble_model, :bagging_fraction,
                lower=0.5, upper=1.0)
K_range = range(ensemble_model, :(model.K),
                lower=1, upper=20)

tm = TunedModel(model=ensemble_model,
            tuning=Grid(resolution=10), # 10x10 grid
            resampling=Holdout(fraction_train=0.8, rng=StableRNG(42)),
            ranges=[B_range, K_range])

tuned_ensemble = machine(tm, X, y)

tuned_estimates = evaluate!(tuned_ensemble, resampling=CV(), rows=train)
tuned_estimates

