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

train = rand(nflowers) .< 0.90
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
    ) #hide
    axnew = Axis( #hide
        f[1, 2], #hide
        title="Predictions", #hide
        xlabel="Class", #hide
        ylabel="Flowers", #hide
        xticks=(1:3, names(y, 2)), #hide
        xticklabelrotation=π / 4 #hide
    ) #hide
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
class_mapper = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"] #hide
ŷ = hcat( #hide
    vcat(findall.(==(1), [test, train])...), #hide
    vcat( #hide
        [class_mapper[cidx] for (_, cidx) in Tuple.(argmax(ŷtest, dims=2))], #hide
        [class_mapper[cidx] for (_, cidx) in Tuple.(argmax(ŷtrain, dims=2))] #hide
    ) #hide
); #hide

# ## Assesing the predictive performance of the proposed model
# Firstly, let's visualize the prediction over this scatter plot to map where are the incorrect
# predictions:
using AlgebraOfGraphics, CairoMakie #hide
set_aog_theme!() #hide
df = ( #hide
    sepallength=X[:, "sepallength"], #hide
    petallength=X[:, "petallength"], #hide
    train=train, #hide
    y=vec([class_mapper[cidx] for (_, cidx) in Tuple.(argmax(y, dims=2))]), #hide
    yhat=ŷ[sortperm(ŷ[:, 1]), 2] #hide
) #hide
plt = data(df) #hide
plt *= mapping( #hide
    :sepallength => "Sepal Length (cm)", #hide
    :petallength => "Petal Length (cm)", #hide
    row=:train => renamer(true => "Training set", false => "Testing set") => "Dataset", #hide
    col=:y => "Class", #hide
    color=:yhat => "Predicted class" #hide
) #hide
draw(plt; axis=(width=225, height=225)) #hide

# As we can see, the model is capable of predicting the vast mayority of the labels correctly,
# failing in some cases where the virginica and versicolor flower overlap. Let's quantify the
# predictive performance of the model using some of `SimSpread.jl` built-in performance
# assesment functions, for example, area under the Receiver-Operating-Characteristic:
println("AuROC (training set): ", AuROC(Bool.(vec(ytrain)), vec(ŷtrain)))
println("AuROC (testing set):  ", AuROC(Bool.(vec(ytest)), vec(ŷtest)))
# and Precision-Recall curve:
println("AuPRC (training set): ", AuPRC(Bool.(vec(ytrain)), vec(ŷtrain)))
println("AuPRC (testing set):  ", AuPRC(Bool.(vec(ytest)), vec(ŷtest)))
