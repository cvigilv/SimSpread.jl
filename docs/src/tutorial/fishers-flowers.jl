# # Fisher's flowers
#
# In this tutorial, we will go through step by step in a more in depth workflow for using
# `SimSpread.jl` having as example the classic classification problem of R.A. Fisher iris
# dataset.
#
# Let's go ahead and load the dataset we will work with, that is, the flower classes and
# features:
using NamedArrays #hide
using SimSpread

y = read_namedmatrix("data/iris.classes")
X = read_namedmatrix("data/iris.features")
setdimnames!(y, ["Flower", "Class"]) #hide
setdimnames!(X, ["Flower", "Feature"]) #hide
nothing #hide

# Classes are one-hot encoded due to how SimSpread works:
y[1:5, :]

# And features can be of any type (e.g., continuous floats describing the plants):
X[1:5, :]

# Next, we will train a model using SimSpread to predict the classes for a subset of plants
# in the Iris dataset. For this, we will split our dataset in 2 groups: training set, which
# will correspond to 90% of the data, and testing set, which will correspond to the remaining
# 10%:
using Random #hide
Random.seed!(1) #hide
nflowers = size(y, 1)

train = rand(nflowers) .< 0.9
test = .!train

ytrain = y[train, :]
ytest = y[test, :]
nothing#hide

# ## Meta-description preparation
#
# As we previously mentioned, SimSpread uses an abstracted feature set where entities are
# described by their similarity to other entities. This permits the added flexibility of
# freely choosing any type of features and similarity measurement to correctly describe
# the problems entities.
#
# To generate this meta-description features, the following steps are taken:
# 1. A similarity matrix $S$ is obtained from the calculation of a similarity metric between
#    all pairs of feature vectors of the entities on the studied dataset:
# 2. From $S$ we can construct a similarity-based feature matrix $S^\prime$ by applying the
#    similarity threshold $\alpha$ using the following equation:
#    $S^\prime_{ij}={w(i,j) \ \text{if} \ S_{ij} \ge \alpha; \ 0 \ \text{otherwise.}}$ where
#    $S$ corresponds to the entities similarity matrix, $S^\prime$ to the final feature
#    matrix, $i$ and $j$ to entities in the studied dataset, and $w(i,j)$ the weighting
#    scheme employed for feature matrix construction, which can be binary,
#    $w(i,j) = S_{ij} > 0$, or continuous, $w(i,j) = (S_{ij} > 0) \times S_{ij}$.
#
# This meta-description matrix encodes the question "Is plant _i_ similar to plant _j_?", 
# which is later used by the resource spreading algorithm for link prediction.
#
# Here, we will use the Jaccard index as our similarity measure, similarity measurement that
# is bound between 0 and 1, and will use a cutoff of $J(x,y) = 0.9$, since this will conserve
# all comparison between highly similar flowers:
using CairoMakie#hide
using Distances, NamedArrays

S = NamedArray(1 .- pairwise(Jaccard(), X, dims=1))
f = Figure(resolution=(600, 500)) #hide
axdd, hmdd = heatmap(f[1, 1], S.array'; colorrange=(0, 1), colormap=:binary) #hide
Colorbar(f[1, 2], hmdd; label="Jaccard similarity") #hide
axdd.title = "Flower similarity" #hide
axdd.xlabel = "Flower" #hide
axdd.ylabel = "Flower" #hide
colsize!(f.layout, 1, Aspect(1, 1.0)) #hide
f#hide

# From this similarity matrix, we will prepare our meta-description for both training and
# testing sets:
heatmaps(M::NamedArray, N::NamedArray) = begin #hide
    f = Figure(resolution=(700, 300)) #hide
    axold, _ = heatmap(f[1, 1], M.array'; colorrange=(0, 1), colormap=:binary) #hide
    axnew, hmnew = heatmap(f[1, 2], N.array'; colorrange=(0, 1), colormap=:binary) #hide
    Colorbar(f[1, 3], hmnew; label="Jaccard Similarity") #hide
    axold.title = "Before" #hide
    axnew.title = "After" #hide
    axold.xlabel = "Flowers" #hide
    axold.ylabel = "Flowers" #hide
    axnew.xlabel = "Flowers" #hide
    axnew.ylabel = "Flowers" #hide
    colsize!(f.layout, 1, Aspect(1, 1.0)) #hide
    colsize!(f.layout, 2, Aspect(1, 1.0)) #hide
    return f #hide
end; #hide

α = 0.9
Xtrain = featurize(S[train, train], α, true)
Xtest = featurize(S[test, train], α, true)
nothing #hide

# - Training set meta-description matrix:
heatmaps(S[train, train], Xtrain) #hide

# - Testing set meta-description matrix:
heatmaps(S[test, train], Xtest) #hide

# ## Predicting labels with SimSpread
#
# Now that we have all the information necessary for SimSpread, we can construct the query
# graph that is used to predict links using network-based-inference resource allocation
# algorithm.
#
# In first place, we need to construct the query network for label prediction:
G = construct(ytrain, ytest, Xtrain, Xtest)
nothing#hide

# From this, we can predict the labels as follows:
ŷtrain = predict(G, ytrain)
ŷtest = predict(G, ytest)
nothing #hide

# Let's visualize the predictions obtained from our model:
heatmaps(M::NamedArray, N::NamedArray) = begin #hide
    M′ = M' #./ maximum(M; dims=2))' #hide
    N′ = N' #./ maximum(N; dims=2))' #hide
    @show maxscore = maximum([maximum(vec(M)), maximum(vec(N))]) #hide
    f = Figure(resolution=(700, 300)) #hide
    axold = Axis(#hide
        f[1, 1],#hide
        title="Ground-truth", #hide
        xlabel="Class", #hide
        ylabel="Flowers", #hide
        xticks=(1:3, names(y, 2)), #hide
        xticklabelrotation=π / 4, #hide
    )
    axnew = Axis(
        f[1, 2],
        title="Predictions",
        xlabel="Class",
        ylabel="Flowers",
        xticks=(1:3, names(y, 2)), #hide
        xticklabelrotation=π / 4 #hide
    )
    heatmap!(axold, M′.array; colorrange=(0, maxscore), colormap=:binary) #hide
    hmnew = heatmap!(axnew, N′.array; colorrange=(0, maxscore), colormap=:binary) #hide

    Colorbar(f[1, 3], hmnew; label="SimSpread score") #hide
    colsize!(f.layout, 1, Aspect(1, 1.0)) #hide
    colsize!(f.layout, 2, Aspect(1, 1.0)) #hide
    return f #hide
end; #hide

# - Training set:
heatmaps(ytrain, ŷtrain) #hide

# - Testing set:
heatmaps(ytest, ŷtest) #hide

# As we can see, we predict the probability for each class of flower possible. To evaluate
# the predictive performance as a multiclass problem, we will assign the label with the
# highest score as the predicted label.
# > In the example above, the predicted labels for each row in the matrix would be
# > "Iris-virginica" (C3), "Iris-setosa" (C1) & "Iris-setosa" (C1).
# 
# To convert the problem from single-class to multi-class, we do the following:
class_mapper = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]

ŷ = hcat(
    vcat(test_idx, train_idx),
    vcat(
        [class_mapper[cidx] for (_, cidx) in Tuple.(argmax(ŷtest, dims=2))],
        [class_mapper[cidx] for (_, cidx) in Tuple.(argmax(ŷtrain, dims=2))]
    )
)

first(ŷ[:, 2], 3)

# Great! Our predicted labels match what we expected. Now let's assess how good is SimSpread
# in predicting the classes for the iris dataset.
#
# ## Assesing the predictive performance of the proposed model
# In order to have an idea of the predictive performance of the model we constructed, we
# will use two common metrics in multi-class prediction problems to evaluate the predictions
# for both the training and testing sets:
#
# 1. _Accuracy_, that indicates how close a given set of predictions are to their true
# value, and 2. _Error rate_, that indicates the inverse of accuracy.
#
# Let's start with accuracy:
using AlgebraOfGraphics, CairoMakie
set_aog_theme!()

df = (
    train=[Bool(i ∈ train_idx) for i in 1:N],
    y=y,
    yhat=ŷ[sortperm(ŷ[:, 1]), 2]
)

plt = data(df)
plt *= expectation()
plt *= mapping(
    :y => "Class",
    (:y, :yhat) => isequal => "Accuracy"
)
plt *= mapping(
    dodge=:train => renamer(true => "Training set", false => "Testing set") => "Dataset",
    color=:train => renamer(true => "Training set", false => "Testing set") => "Dataset"
)

draw(plt; axis=(width=400, height=225))

# As we can see, our proposed SimSpread model achieves high accuracy for both training and
# testing sets. Let's see the error rates for the same grouping:
plt = data(df)
plt *= expectation()
plt *= mapping(
    :y => "Class",
    (:y, :yhat) => !isequal => "Error rate"
)
plt *= mapping(
    dodge=:train => renamer(true => "Training set", false => "Testing set") => "Dataset",
    color=:train => renamer(true => "Training set", false => "Testing set") => "Dataset"
)

draw(plt; axis=(width=400, height=225))

# Here we also see goo performance, achieving low error rate for all the classes in both the
# training and testing sets. We also can appreciate that the the testing set present a higher
# mean error rate than the training set.
#
# Let's visualize where the predicted classes fall in our training and testing sets. First,
# lets see our ground truth:
df = (
    sepallength=S[!, "sepallength"],
    petallength=S[!, "petallength"],
    y=iris[!, "class"],
)
plt = data(df)
plt *= mapping(
    :sepallength => "Sepal Length (cm)",
    :petallength => "Petal Length (cm)",
    color=:y => "Class"
)
draw(plt; axis=(width=300, height=300))

# We can clearly see that _setosa_ plants are completely separated from the rest of the plants
# in the dataset. _Versicolor_ and _virginica_ present some overlap, which might respond to
# what we have see in the predictive performance.
#
# Let's visualize the prediction over this scatter plot to map where are the incorrect
# predictions:
df = (
    sepallength=S[!, "sepallength"],
    petallength=S[!, "petallength"],
    train=[Bool(i ∈ train_idx) for i in 1:N],
    y=iris[!, "class"],
    yhat=ŷ[sortperm(ŷ[:, 1]), 2]
)

plt = data(df)
plt *= mapping(
    :sepallength => "Sepal Length (cm)",
    :petallength => "Petal Length (cm)",
    row=:train => renamer(true => "Training set", false => "Testing set") => "Dataset",
    col=:y => "Class",
    color=:yhat => "Predicted class"
)

draw(plt; axis=(width=225, height=225))
