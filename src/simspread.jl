"""
    Base.split(y::NamedArray, k::Int64; seed::Int64=1)

Split source nodes in `y` into `k` groups for cross-validation.

# Arguments
- `y::AbstractMatrix`: Drug-Target rectangular adjacency matrix.
- `k::Int64`: Number of groups to use in data splitting.
- `seed::Int64`: Seed used for data splitting.
"""
function Base.split(y::NamedArray, k::Int64; seed::Int64=1)
    # Get array of drugs in adjacency matrix
    sources = names(y, 1)

    # Assign fold to edges of graph
    shuffle!(MersenneTwister(seed), sources)
    groups = [[] for _ in 1:k]

    for (i, sᵢ) in enumerate(sources)
        foldᵢ = mod(i, k) + 1
        push!(groups[foldᵢ], sᵢ)
    end

    return groups
end

"""
    cutoff(x::T, α::T, weighted::Bool=false) where {T<:AbstractFloat}

Transform `x` based in SimSpread's similarity cutoff function.

# Arguments
- `x::AbstractFloat` : Value to transform
- `α::AbstractFloat` : Similarity cutoff
- `weighted::Bool` : Apply weighting function to outcome (default = false)
"""
function cutoff(x::T, α::T, weighted::Bool=false) where {T<:AbstractFloat}
    x′ = deepcopy(x)
    weight = weighted ? x′ : 1.0
    x′ = x′ ≥ α ? weight : 0.0

    return x′
end

"""
    cutoff(M::AbstractVecOrMat{T}, α::T, weighted::Bool=false) where {T<:AbstractFloat}

Transform the vector or matrix `X` based in SimSpread's similarity cutoff function.

# Arguments
- `X::AbstractVecOrMat{AbstractFloat}` : Matrix or Vector to transform
- `α::AbstractFloat` : Similarity cutoff
- `weighted::Bool` : Apply weighting function to outcome (default = false)
"""
function cutoff(X::AbstractVecOrMat{T}, α::T, weighted::Bool=false) where {T<:AbstractFloat}
    X′ = deepcopy(X)
    X′ = cutoff.(X′, α, weighted)

    return X′
end

"""
    cutoff!(x::T, α::T, weighted::Bool=false) where {T<:AbstractFloat}

Transform, in place, `x` based in SimSpread's similarity cutoff function.

# Arguments
- `x::AbstractFloat` : Value to transform
- `α::AbstractFloat` : Similarity cutoff
- `weighted::Bool` : Apply weighting function to outcome (default = false)
"""
function cutoff!(x::T, α::T, weighted::Bool=false) where {T<:AbstractFloat}
    weight = weighted ? x : 1.0
    x = x ≥ α ? weight : 0.0
end

"""
    cutoff!(X::AbstractVecOrMat{T}, α::T, weighted::Bool=false) where {T<:AbstractFloat}

Transform, in place, the vector or matrix `X` based in SimSpread's similarity cutoff function.

# Arguments
- `X::AbstractVecOrMat{AbstractFloat}` : Matrix or Vector to transform
- `α::AbstractFloat` : Similarity threshold
- `weighted::Bool` : Apply weighting function to outcome (default = false)
"""
function cutoff!(X::AbstractVecOrMat{T}, α::T, weighted::Bool=false) where {T<:AbstractFloat}
    cutoff!.(X, α, weighted)
end

"""
    featurize(X::NamedArray, α::AbstractFloat, weighted::Bool=false)

Transform the feature matrix `X` into a SimSpread feature matrix.

# Arguments
- `X::NamedArray`: Continuous feature matrix
- `α::AbstractFloat`: Featurization cutoff
- `weighted::Bool` : Apply weighting function to outcome (default = false)

# References
1. Vigil-Vásquez & Schüller (2022). De Novo Prediction of Drug Targets and Candidates by
   Chemical Similarity-Guided Network-Based Inference. International Journal of Molecular
   Sciences, 23(17), 9666. https://doi.org/10.3390/ijms23179666
"""
function featurize(X::NamedArray, α::AbstractFloat, weighted::Bool=false)
    # Filter matrix
    X′ = copy(X)
    X′.array = cutoff.(X.array, α, weighted)
    setnames!(X′, ["f$f" for f in names(X′, 2)], 2)
    return X′
end

"""
    featurize!(X::NamedArray, α::AbstractFloat, weighted::Bool=false)

Transform, in place, the feature matrix `X` into a SimSpread feature matrix.

# Arguments
- `X::NamedArray` : Continuous feature matrix
- `α::AbstractFloat` : Featurization cutoff
- `weighted::Bool` : Apply weighting function to outcome (default = false)

# References
1. Vigil-Vásquez & Schüller (2022). De Novo Prediction of Drug Targets and Candidates by
   Chemical Similarity-Guided Network-Based Inference. International Journal of Molecular
   Sciences, 23(17), 9666. https://doi.org/10.3390/ijms23179666
"""
function featurize!(X::NamedArray, α::AbstractFloat, weighted::Bool=false)
    X.array = cutoff.(X.array, α, weighted)
    setnames!(X, ["f$f" for f in names(X, 2)], 2)
end

"""
    construct(y::NamedMatrix, X::NamedMatrix, queries::AbstractVector)

Construct the query-feature-source-target network for *de novo* network-based inference
prediction and return adjacency matrix.

# Arguments
- `y::NamedMatrix`: Source-target bipartite network adjacency matrix
- `X::NamedMatrix`: Source-feature bipartite adjacency matrix
- `queries::AbstractVector`: Source nodes to use as query

# Extended help
This implementation is intended for k-fold or leave-one-out cross-validation.
"""
function construct(y::NamedMatrix, X::NamedMatrix, queries::AbstractVector)
    @assert size(y, 1) == size(X, 1) "Different number of compounds!"

    # Get names from matrices
    features = [f for f in names(X, 2) if lstrip(f, 'f') ∉ queries]
    sources = [d for d in names(X, 1) if d ∉ queries]
    targets = names(y, 2)

    @assert all(sort(features) .!= sort(sources)) "Features and drugs have the same names!"

    # Get dimensions of network
    Nqueries = length(queries)
    Nfeatures = length(features)
    Nsources = length(sources)
    Ntargets = length(targets)

    # Construct trilayered graph adjacency matrix
    Mqq = zeros(Nqueries, Nqueries)
    Mqs = zeros(Nqueries, Nsources)
    Mqf = X[queries, features].array
    Mqt = zeros(Nqueries, Ntargets)
    Msq = Mqs'
    Mss = zeros(Nsources, Nsources)
    Msf = X[sources, features].array
    Mst = y[sources, targets].array
    Mfq = Mqf'
    Mfs = Msf'
    Mff = zeros(Nfeatures, Nfeatures)
    Mft = zeros(Nfeatures, Ntargets)
    Mtq = Mqt'
    Mts = Mst'
    Mtf = Mft'
    Mtt = zeros(Ntargets, Ntargets)

    A = Matrix(
        [Mqq Mqs Mqf Mqt
            Msq Mss Msf Mst
            Mfq Mfs Mff Mft
            Mtq Mts Mtf Mtt]
    )

    namedA = NamedArray(
        A,
        (vcat(queries, sources, features, targets), vcat(queries, sources, features, targets))
    )
    namedB = deepcopy(namedA)
    namedB[queries, :] .= 0
    namedB[:, queries] .= 0

    return namedA, namedB
end

"""
    construct(ys::T, Xs::T) where {T<:Tuple{NamedMatrix,NamedMatrix}}

Construct the query-feature-source-target network for *de novo* network-based inference
prediction and return adjacency matrix.

# Arguments
- `dts::Tuple{NamedMatrix,NamedMatrix}` : Source-target bipartite graph adjacency matrices
- `dfs::Tuple{NamedMatrix,NamedMatrix}` : Source-feature bipartite graph adjacency matrices

# Extended help
This implementations is intended for time-split cross-validation or manual construction of query
network.
"""
function construct(ys::T, Xs::T) where {T<:Tuple{NamedMatrix,NamedMatrix}}
    # Unpack matrices tuples
    ytrain, ytest = ys
    Xtrain, Xtest = Xs

    @assert size(ytrain, 2) == size(ytest, 2) "Number of targets between test and training sets doens't match"
    @assert size(Xtrain, 2) == size(Xtest, 2) "Number of features between test and training sets doens't match"

    # Get names from matrices
    features = names(Xtrain, 2)
    sources = names(ytrain, 1)
    targets = names(ytrain, 2)
    queries = names(ytest, 1)

    @assert all(sort(features) .!= sort(sources)) "Features and drugs have the same names!"

    # Get dimensions of network
    Nsources = length(sources)
    Nqueries = length(queries)
    Ntargets = length(targets)
    Nfeatures = length(features)

    # Construct trilayered graph adjacency matrix
    Mqq = zeros(Nqueries, Nqueries)
    Mqs = zeros(Nqueries, Nsources)
    Mqf = Xtest.array
    Mqt = zeros(Nqueries, Ntargets)
    Msq = Mqs'
    Mss = zeros(Nsources, Nsources)
    Msf = Xtrain.array
    Mst = ytrain.array
    Mfq = Mqf'
    Mfs = Msf'
    Mff = zeros(Nfeatures, Nfeatures)
    Mft = zeros(Nfeatures, Ntargets)
    Mtq = Mqt'
    Mts = Mst'
    Mtf = Mft'
    Mtt = zeros(Ntargets, Ntargets)

    A = Matrix(
        [Mqq Mqs Mqf Mqt
            Msq Mss Msf Mst
            Mfq Mfs Mff Mft
            Mtq Mts Mtf Mtt]
    )

    namedA = NamedArray(
        A,
        (vcat(queries, sources, features, targets), vcat(queries, sources, features, targets))
    )
    namedB = deepcopy(namedA)
    namedB[queries, :] .= 0
    namedB[:, queries] .= 0

    return namedA, namedB
end

"""
    construct(ytrain::T, ytest::T, Xtrain::T, Xtest::T) where {T<:NamedMatrix}

Construct the query-feature-source-target network for *de novo* network-based inference
prediction and return adjacency matrix.

# Arguments
- `ytrain::NamedMatrix` : Training source-target bipartite graph adjacency matrix
- `ytest::NamedMatrix` : Test source-target bipartite graph adjacency matrix
- `Xtrain::NamedMatrix` : Training source-feature bipartite graph adjacency matrix
- `Xtest::NamedMatrix` : Test source-feature bipartite graph adjacency matrix

# Extended help
This implementations is intended for time-split cross-validation or manual construction of query
network.
"""
function construct(ytrain::T, ytest::T, Xtrain::T, Xtest::T) where {T<:NamedMatrix}
    construct((ytrain, ytest), (Xtrain, Xtest))
end

"""
    construct(y::NamedMatrix, X::NamedMatrix)

Construct the feature-source-target network for network-based inference
prediction and return adjacency matrix.

# Arguments
- `y::NamedMatrix` : Source-target bipartite graph adjacency matrix
- `X::NamedMatrix` : Source-feature bipartite graph adjacency matrix
"""
function construct(y::NamedMatrix, X::NamedMatrix)
    # Get names from matrices
    features = names(X, 2)
    sources = names(y, 1)
    targets = names(y, 2)

    @assert all(sort(features) .!= sort(sources)) "Source and feature nodes have the same names"

    # Get dimensions of network
    Nsources = length(sources)
    Ntargets = length(targets)
    Nfeatures = length(features)

    # Construct trilayered graph adjacency matrix
    Mss = zeros(Nsources, Nsources)
    Msf = X.array
    Mst = y.array
    Mfs = Msf'
    Mff = zeros(Nfeatures, Nfeatures)
    Mft = zeros(Nfeatures, Ntargets)
    Mts = Mst'
    Mtf = Mft'
    Mtt = zeros(Ntargets, Ntargets)

    A = Matrix([Mss Msf Mst; Mfs Mff Mft; Mts Mtf Mtt])

    index_names = vcat(sources, features, targets)

    return NamedArray(A, (index_names, index_names))
end

"""
    spread(G::AbstractMatrix{Float64})

Calculate the transfer matrix for the adyacency matrix of the trilayered feature-source-target
network.

# Arguments
- `G::AbstractMatrix{Float64}`: Trilayered feature-source-target network adjacency matrix.

# Extended help
Potential interactions between nodes in a graph can be identified by using resource diffusion
processes in the feature-source-target network, namely aforementioned graph `G`. For each node
nᵢ in the network, it has initial resources located in both its neighboring nodes and its
features. Initially, each feature and each neighboring node of nᵢ equally spread their resources
to neighboring nodes. Subsequently, each of those nodes equally spreads its resources to
neighbor nodes. Thus, nᵢ will obtain final resources located in several neighboring nodes,
suggesting that nᵢ may have potential interactions with these nodes.

# References
1. Wu, et al (2016). SDTNBI: an integrated network and chemoinformatics tool for
   systematic prediction of drug–target interactions and drug repositioning. Briefings in
   Bioinformatics, bbw012. https://doi.org/10.1093/bib/bbw012
2. Vigil-Vásquez & Schüller (2022). De Novo Prediction of Drug Targets and Candidates by
   Chemical Similarity-Guided Network-Based Inference. International Journal of Molecular
   Sciences, 23(17), 9666. https://doi.org/10.3390/ijms23179666
"""
function spread(G::AbstractMatrix{Float64})
    W = G ./ k(G)
    replace!(W, Inf => 0.0)
    replace!(W, NaN => 0.0)

    return W
end

spread(G::AbstractMatrix{Bool}) = spread(AbstractMatrix{Float64}(G))

function spread(G::NamedMatrix)
    W = copy(G)
    W.array = spread(Matrix{Float64}(G.array))

    return W
end


"""
    predict(I::Tuple{T,T}, ytest::T; GPU::Bool=false) where {T<:NamedMatrix}

Predict interactions between query and target nodes using *de novo* network-based inference
model proposed by Wu, et al (2016).

# Arguments
- `I::Tuple{NamedMatrix,NamedMatrix}`: Feature-source-target trilayered adjacency matrices
- `ytest::NamedMatrix`: Query-target bipartite adjacency matrix
- `GPU::Bool`: Use GPU acceleration for calculation (default = false)

# References
1. Wu, et al (2016). SDTNBI: an integrated network and chemoinformatics tool for
   systematic prediction of drug–target interactions and drug repositioning. Briefings in
   Bioinformatics, bbw012. https://doi.org/10.1093/bib/bbw012
2. Vigil-Vásquez & Schüller (2022). De Novo Prediction of Drug Targets and Candidates by
   Chemical Similarity-Guided Network-Based Inference. International Journal of Molecular
   Sciences, 23(17), 9666. https://doi.org/10.3390/ijms23179666
"""
function predict(I::Tuple{T,T}, ytest::T; GPU::Bool=false) where {T<:NamedMatrix}
    # GPU calculations helper functions
    _useGPU(x::AbstractArray) = GPU ? CuArray{Float32}(x) : x

    # Target prediction using NBI
    A, B = I
    W = spread(B.array)
    F = copy(A)

    Aarr = _useGPU(A.array)
    Warr = _useGPU(W)
    F.array = Aarr * Warr^2

    # Free GPU memory
    if GPU
        CUDA.unsafe_free!(Aarr)
        CUDA.unsafe_free!(Warr)
    end

    yhat = F[names(ytest, 1), names(ytest, 2)]
    return yhat
end
predict(A::T, B::T, ytest::T; GPU::Bool=false) where {T<:NamedMatrix} =
    predict((A, B), ytest; GPU=GPU)

"""
    predict(A::T, ytrain::T; GPU::Bool=false) where {T<:NamedMatrix}

Predict interactions between query and target nodes using *de novo* network-based inference
model proposed by Wu, et al (2016).

# Arguments
- `A::NamedMatrix`: Feature-source-target trilayered adjacency matrix
- `ytrain::NamedMatrix`: Source-target bipartite adjacency matrix
- `GPU::Bool`: Use GPU acceleration for calculation (default = false)

# References
1. Wu, et al (2016). SDTNBI: an integrated network and chemoinformatics tool for
   systematic prediction of drug–target interactions and drug repositioning. Briefings in
   Bioinformatics, bbw012. https://doi.org/10.1093/bib/bbw012
2. Vigil-Vásquez & Schüller (2022). De Novo Prediction of Drug Targets and Candidates by
   Chemical Similarity-Guided Network-Based Inference. International Journal of Molecular
   Sciences, 23(17), 9666. https://doi.org/10.3390/ijms23179666
"""
function predict(A::T, ytrain::T; GPU::Bool=false) where {T<:NamedMatrix}
    # GPU calculations helper functions
    _useGPU(x::AbstractArray) = GPU ? CuArray{Float32}(x) : x

    # Target prediction using NBI
    W = spread(A.array)
    F = copy(A)

    Aarr = _useGPU(A.array)
    Warr = _useGPU(W)
    F.array = Aarr * Warr^2

    # Free GPU memory
    if GPU
        CUDA.unsafe_free!(Aarr)
        CUDA.unsafe_free!(Warr)
    end

    yhat = F[names(ytrain, 1), names(ytrain, 2)]
    return yhat
end

"""
    clean!(yhat::NamedArray, A::NamedArray, y::NamedArray)

Flag, in place, erroneous prediction from cross-validation splitting.

# Arguments
- `yhat::NamedArray`: Predicted source-target bipartite adjacency matrix
- `A::NamedArray`: Initial resource source-target resources adjacency matrix
- `y::NamedArray`: Ground-truth source-target bipartite adjacency matrix
"""
function clean!(yhat::NamedArray, A::NamedArray, y::NamedArray)
    for (tᵢ, k) in zip(names(y, 2), k(A[names(y, 2),:]))
        if k == 0
            yhat[:, tᵢ] .= -99
        end
    end
end

"""
    save(filepath::String, yhat::NamedMatrix, y::NamedMatrix; delimiter::Char='\t')

Store predictions as a table in the given file path.

# Arguments
- `filepath::String`: Output file path
- `yhat::NamedArray`: Predicted source-target bipartite adjacency matrix
- `y::NamedArray`: Ground-truth source-target bipartite adjacency matrix
- `delimiter::Char`: Delimiter used to write table (default = '\\t')

# Extended help
Table format is:
```
fold, source, target, score, label
```
"""
function save(filepath::String, yhat::NamedMatrix, y::NamedMatrix; delimiter::Char='\t')
    # Get name arrays
    queries = names(y, 1)
    targets = names(y, 2)

    # Save file
    open(filepath, "a+") do f
        for qᵢ in queries, tᵢ in targets
            row = [
                findfirst(id -> id == qᵢ, queries),
                qᵢ,
                tᵢ,
                yhat[qᵢ, tᵢ],
                y[qᵢ, tᵢ]
            ]

            write(f, join(row, delimiter))
        end
    end
end

"""
    save(filepath::String, fidx::Int64, yhat::NamedMatrix, y::NamedMatrix; delimiter::Char='\t')

Store cross-valudation predictions as a table in the given file path.

# Arguments
- `filepath::String`: Output file path
- `fidx::Int64`: Numeric fold ID
- `yhat::NamedArray`: Predicted source-target bipartite adjacency matrix
- `y::NamedArray`: Ground-truth source-target bipartite adjacency matrix
- `delimiter::Char`: Delimiter used to write table (default = '\\t')

# Extended help
Table format is:
```
fold, source, target, score, label
```
"""
function save(filepath::String, fidx::Int64, yhat::NamedMatrix, y::NamedMatrix; delimiter::Char='\t')
    # Get name arrays
    queries = names(y, 1)
    targets = names(y, 2)

    # Save file
    open(filepath, "a+") do f
        for qᵢ in queries, tᵢ in targets
            row = [
                fidx,
                qᵢ,
                tᵢ,
                yhat[qᵢ, tᵢ],
                y[qᵢ, tᵢ]
            ]

            write(f, join(row, delimiter))
        end
    end
end
