# # Getting started with SimSpread.jl
# SimSpread is a novel approach for predicting interactions between two distinct set of
# nodes, query and target nodes, using a similarity measure vector between query nodes as
# a meta-description in combination with the network-based inference for link prediction.
#
# In this tutorial, we will skim through the basic workflow for using `SimSpread.jl` using
# as an example the prediction of dug-target interactions for the Nuclear Receptor dataset
# from Yamanishi, et al (2008).
#
# ## Preparing our problem
# First, we will download the known drug-target interaction matrix and drug-drug SIMCOMP
# similarity matrix for the 'Nuclear Receptor' dataset. The package provides a helper for
# easy download and preparation for this group of datasets (refer to `??getyamanishi`
# for more information).
using CairoMakie #hide
using NamedArrays # hide
CairoMakie.activate!() # hide
using SimSpread

DT, DD = getyamanishi("nr")
nothing #hide

# Let's visualize our data as heatmaps:
heatmaps(M::NamedArray, N::NamedArray) = begin #hide
    M′ =  M' #hide
    N′ =  N' #hide
    f = Figure(resolution=(700, 300)) #hide
    axdt, _ = heatmap(f[1, 1], M′.array; colorrange=(0, 1), colormap=:binary) #hide
    axdd, hmdd = heatmap(f[1, 2], N′.array; colorrange=(0, 1), colormap=:binary) #hide
    Colorbar(f[1, 3], hmdd; label="SIMCOMP similarity") #hide
    axdt.title = "Drug-Target\ninteractions" #hide
    axdd.title = "Drug-Drug\nsimilarity" #hide
    axdt.xlabel = "Targets" #hide
    axdt.ylabel = "Drugs" #hide
    axdd.xlabel = "Drugs" #hide
    axdd.ylabel = "Drugs" #hide
    colsize!(f.layout, 1, Aspect(1, 1.0)) #hide
    colsize!(f.layout, 2, Aspect(1, 1.0)) #hide
    return f #hide
end; #hide
heatmaps(DT, DD) #hide

# ## Data splitting
#
# Next, we will train a model using SimSpread to predict the targets for a subset of drugs
# in the dataset. For this, we will split our dataset in 2 groups: training set, which will
# correspond to 90% of the data, and testing set, which will correspond to the remaining 10%:
using Random #hide
Random.seed!(1) #hide
N = size(DT, 1)

train = rand(N) .< 0.9
test = .!train

ytrain = DT[train, :]
ytest = DT[test, :]
nothing #hide

#

println("Training set size: ", size(ytrain)) #hide
println("Testing set size:  ", size(ytest)) #hide

# As seen here, around 90% of the dataset corresponds to training and the remaining to testing
# sets. From this splitting we will proceed to construct our query network and predict the
# interactions for the testing set.

# ## Similarity-based meta-description preparation
#
# SimSpread uses a meta-description constructed from the similarity between source nodes (drugs
# in the working example). For this, a similarity threshold (denoted with α) is employed to keep
# links between source nodes that have a weight greater or equal to this threshold. 
#
# This procedure encodes the question "Is drug _i_ similar to drug _j_?", which is later used by
# the resource spreading algorithm for link prediction.
α = 0.35
Xtrain = featurize(DD[train, train], α, false)
Xtest = featurize(DD[test, train], α, false)
nothing #hide

# Let's compare the similarity matrices before and after the featurization procedure:
# - Training set:
heatmaps(M::NamedArray, N::NamedArray) = begin #hide
    f = Figure(resolution=(700, 300)) #hide
    axold, _  = heatmap(f[1, 1], M.array; colorrange=(0, 1), colormap=:binary) #hide
    axnew, hmnew = heatmap(f[1, 2], N.array; colorrange=(0, 1), colormap=:binary) #hide
    Colorbar(f[1, 3], hmnew; label="Similarity") #hide
    axold.title = "Before" #hide
    axnew.title = "After" #hide
    axold.xlabel = "Drugs" #hide
    axold.ylabel = "Drugs" #hide
    axnew.xlabel = "Drugs" #hide
    axnew.ylabel = "Drugs" #hide
    colsize!(f.layout, 1, Aspect(1, 1.0)) #hide
    colsize!(f.layout, 2, Aspect(1, 1.0)) #hide
    return f #hide
end; #hide

heatmaps(DD[train, train], Xtrain) #hide

# - Testing set:
heatmaps(DD[test, train], Xtest) #hide

# As seen here, all comparisons with a weight lower than our threshold α are eliminated (i.e.
# filled with a zero, 0) and a structure arises from this new featurized matrix.

# ## Predicting DTIs with SimSpread
#
# Now that we have all the information necessary for SimSpread, we can construct the query
# graph that is used to predict links using network-based-inference resource allocation
# algorithm.
#
# In first place, we need to construct the query network for label prediction:
G = construct(ytrain, ytest, Xtrain, Xtest)
nothing #hide

# From this, we can predict the labels as follows:
ŷtrain = predict(G, ytrain)
ŷtest = predict(G, ytest)
nothing #hide

# Finally, we assess the performance of our model using the area under ROC curve, denoted as
# AuROC:

println("AuROC training set: ", round(AuROC(Bool.(vec(ytrain)), vec(ŷtrain)); digits=3)) #hide
println("AuROC testing set:  ", round(AuROC(Bool.(vec(ytest)), vec(ŷtest)); digits=3)) #hide

# Let's visualize the predictions obtained from our model:
# - Training set:
heatmaps(M::NamedArray, N::NamedArray) = begin #hide
    M′ =  M' #./ maximum(M; dims=2))' #hide
    N′ =  N' #./ maximum(N; dims=2))' #hide
    @show maxscore = maximum([maximum(vec(M)), maximum(vec(N))]) #hide
    f = Figure(resolution=(700, 300)) #hide
    axold, _ = heatmap(f[1, 1], M′.array; colorrange=(0, maxscore), colormap=:binary) #hide
    axnew, hmnew = heatmap(f[1, 2], N′.array; colorrange=(0, maxscore), colormap=:binary) #hide
    Colorbar(f[1, 3], hmnew; label="SimSpread score") #hide
    axold.title = "Ground-truth" #hide
    axnew.title = "Predictions" #hide
    axold.xlabel = "Targets" #hide
    axold.ylabel = "Drugs" #hide
    axnew.xlabel = "Targets" #hide
    axnew.ylabel = "Drugs" #hide
    colsize!(f.layout, 1, Aspect(1, 1.0)) #hide
    colsize!(f.layout, 2, Aspect(1, 1.0)) #hide
    return f #hide
end; #hide
heatmaps(ytrain, ŷtrain) #hide

# - Testing set:
heatmaps(ytest, ŷtest) #hide

# This wraps up our tutorial. The following tutorial provided (i) a more in-depth use case of
# the core utilities of `SimSpread.jl` and (ii) how to optimize a SimSpread model and evaluate its
# predictions. Adittionally, recipes for common ML tasks are provided in the next sections, specifically,
# common corss-validation scenarios.
