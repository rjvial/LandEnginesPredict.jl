using MLJ
import DataFrames: DataFrame
using PrettyPrinting
using StableRNGs

rng = StableRNG(512)
Xraw = rand(rng, 300, 3)
y = exp.(Xraw[:,1] - Xraw[:,2] - 2Xraw[:,3] + 0.1*rand(rng, 300))
X = DataFrame(Xraw, :auto)

KNNRegressor = @load KNNRegressor
knn = machine(KNNRegressor(K=10), X, y)

# Alertnativa 1
train, test = partition(eachindex(y), 0.7)
MLJ.fit!(knn, rows=train)
ŷ = predict(knn, X[test, :]) # or use rows=test
rms(ŷ, y[test])



# Alertnativa 2
evaluate!(knn, resampling=Holdout(fraction_train=0.7, rng=StableRNG(666)),
          measure=rms)
