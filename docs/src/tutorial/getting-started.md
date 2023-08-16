```@meta
EditURL = "<unknown>/src/tutorial/getting-started.jl"
```

# Getting started with SimSpread.jl

TODO: Small introdution to SimSpread, its origins and use cases

SimSpread consist of N main steps:
1.
2.
3. Construction of a metadescription-object-target network
4. Resource-spreading algorithm to predict the probability an edge between a
   object and target interact.

## Preparing our problem

For this introductory tutorial, we will use as an example the classic "Iris"
dataset proposed by R.A. Fisher, in a classification problem. Let's go ahead
and load the dataset:

````@example getting-started
using MLDatasets, DataFrames

iris = Iris().dataframe
first(iris, 3)
````

Since we will use SimSpread as a classification algorithm for a single-class
problem, we need to convert the `class` column into 3 distinct columns
corresponding to each of the classes in the dataset, i.e., one-hot encode
the class column. For this, we can use the following transformation

````@example getting-started
transform!(
    iris,
    :class => ByRow(c -> c .== "Iris-setosa") => "Is Iris-setosa?",
    :class => ByRow(c -> c .== "Iris-versicolor") => "Is Iris-versicolor?",
    :class => ByRow(c -> c .== "Iris-virginica") => "Is Iris-virginica?",
)
first(iris, 3)
````

and obtain 3 columns that encode the class for each plant.

## Data splitting

Next, we will train a model using SimSpread to predict the classes for a
subset of plants in the Iris dataset. For this, we will split our dataset
in 2 groups: training set, which will correspond to 80% of the data, and
testing set, which will correspond to the remaining 20%.

For this, will first shuffle the plants and extract the first 20% of the
dataset with the following code:

````@example getting-started
using Random

Random.seed!(1)
N = nrow(iris)
perm = randperm(N)

train_idx = last(perm, Int(0.8 * N))
test_idx = first(perm, Int(0.2 * N))

Xtrain = iris[train_idx, ["sepallength", "sepalwidth", "petallength", "petalwidth"]]
Xtest  = iris[test_idx,  ["sepallength", "sepalwidth", "petallength", "petalwidth"]]
ytrain = iris[train_idx, ["Is Iris-setosa?", "Is Iris-versicolor?", "Is Iris-virginica?"]]
ytest  = iris[test_idx,  ["Is Iris-setosa?", "Is Iris-versicolor?", "Is Iris-virginica?"]];
nothing #hide
````

The first 3 entries of the training set have the following features:

````@example getting-started
first(Xtrain, 3)
````

And the following one-hot encoded classes:

````@example getting-started
first(ytrain, 3)
````

The first 3 entries of the test set have the following features:

````@example getting-started
first(Xtest, 3)
````

And the following one-hot encoded classes:

````@example getting-started
first(ytest, 3)
````

## Meta-description preparation

As we previously mentioned, SimSpread uses a meta-description based in
similarity between entities, that is, an abstracted feature set where
entities are described by their similarity to other entities. This permits
the added flexibility of freely choosing any type of features and similarity
measurement to correctly describe the problems entities.

To generate this meta-description features, the following steps are taken:
1. A similarity matrix $S$ is obtained from the calculation of a similarity metric
   between all pairs of feature vectors of the entities on the studied dataset.
2. From $S$ we can construct a similarity-based feature matrix $S^\prime$ by applying
   the similarity threshold $\alpha$ using the following equation:
   $S^\prime_{ij}={w(i,j) \ \text{if} \ S_{ij} \ge \alpha; \ 0 \ \text{otherwise.}}$
   where $S$ corresponds to the entities similarity matrix, $S^\prime$ to the final
   feature matrix, $i$ and $j$ to entities in the studied dataset, and $w(i,j)$ the
   weighting scheme employed for feature matrix construction, which can be binary,
   $w(i,j) = S_{ij} > 0$, or continuous, $w(i,j) = (S_{ij} > 0) \times S_{ij}$.

This meta-description matrix encodes the question "Is plant _i_ similar to plant
_j_?", which is later used by the resource spreading algorithm for link prediction.

Here, we will use the Jaccard index as our similarity measure, similarity measurement
that is bound between 0 and 1:

````@example getting-started
using Distances, NamedArrays

Dtrain = 1 .- pairwise(Jaccard(), Matrix(Xtrain), dims=1)
Dtest  = 1 .- pairwise(Jaccard(), Matrix(Xtest), Matrix(Xtrain), dims=1)

Dtrain = NamedArray(Dtrain, (["E$i" for i in train_idx], ["E$i" for i in train_idx] ))
Dtest  = NamedArray(Dtest, (["E$i" for i in test_idx], ["E$i" for i in train_idx] ));
nothing #hide
````

and will use a cutoff of $J(x,y) = 0.9$, since this will conserve all comparison between
highly similar flowers:

````@example getting-started
using SimSpread

class_names = ["Is Iris-setosa?", "Is Iris-versicolor?", "Is Iris-virginica?"]

α = 0.9
Xtrain′ = featurize(Dtrain, α, true)
Xtest′  = featurize(Dtest, α, true)
ytrain′ = NamedArray(Matrix{Float64}(ytrain), (["E$i" for i in train_idx], class_names))
ytest′  = NamedArray(Matrix{Float64}(ytest), (["E$i" for i in test_idx], class_names));
nothing #hide
````

## Predicting labels with SimSpread

Now that we have all the information necessary for SimSpread, we can construct the query graph
that is used to predict links using network-based-inference resource allocation algorithm.

In first place, we need to construct the query network for label prediction:

````@example getting-started
G = construct(ytrain′, ytest′, Xtrain′, Xtest′)
````

From this, we can predict the labels as follows:

````@example getting-started
ŷtrain = predict(G, ytrain′)
ŷtest = predict(G, ytest′)

ŷtest[1:3, :]
````

As we can see, we predict the probability for each class of flower possible. To
evaluate the predictive performance as a multiclass problem, we will assign the
label with the highest score as the predicted label.
> In the example above, the predicted labels for each row in the matrix would be
> "Iris-virginica" (C3), "Iris-setosa" (C1) & "Iris-setosa" (C1).

To convert the problem from single-class to multi-class, we do the following:

````@example getting-started
class_mapper = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]

ŷ = hcat(
    vcat(test_idx, train_idx),
    vcat(
        [class_mapper[cidx] for (_, cidx) in Tuple.(argmax(ŷtest, dims=2))],
        [class_mapper[cidx] for (_, cidx) in Tuple.(argmax(ŷtrain, dims=2))]
    )
)

first(ŷ[:, 2], 3)
````

Great! Our predicted labels match what we expected. Now let's assess how good is
SimSpread in predicting the classes for the iris dataset.

## Assesing the predictive performance of the proposed model
In order to have an idea of the predictive performance of the model we constructed, we will use
two common metrics in multi-class prediction problems to evaluate the predictions for both the
training and testing sets:

1. _Accuracy_, that indicates how close a given set of predictions are to their true value, and
2. _Error rate_, that indicates the inverse of accuracy.

Let's start with accuracy:

````@example getting-started
using AlgebraOfGraphics, CairoMakie
set_aog_theme!()

df = (
    train=[Bool(i ∈ train_idx) for i in 1:N],
    y=iris[!, "class"],
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
````

As we can see, our proposed SimSpread model achieves high accuracy for both training and
testing sets. Let's see the error rates for the same grouping:

````@example getting-started
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
````

Here we also see goo performance, achieving low error rate for all the classes in both the training and
testing sets. We also can appreciate that the the testing set present a higher mean error rate than the
training set.

Let's visualize where the predicted classes fall in our training and testing sets. First, lets see our
ground truth:

````@example getting-started
df = (
    sepallength=iris[!, "sepallength"],
    petallength=iris[!, "petallength"],
    y=iris[!, "class"],
)
plt = data(df)
plt *= mapping(
    :sepallength => "Sepal Length (cm)",
    :petallength => "Petal Length (cm)",
    color=:y => "Class"
)
draw(plt; axis=(width=300, height=300))
````

We can clearly see that _setosa_ plants are completely separated from the rest of
the plants in the dataset. _Versicolor_ and _virginica_ present some overlap, which
might respond to what we have see in the predictive performance.

Let's visualize the prediction over this scatter plot to map where are the incorrect
predictions:

````@example getting-started
df = (
    sepallength=iris[!, "sepallength"],
    petallength=iris[!, "petallength"],
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
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

