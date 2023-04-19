"""
    spread(G::AbstractMatrix{Float64})

Calculate the transfer matrix for the adyacency matrix of the trilayered feature-source-target
network.

# Arguments
- `F₀::AbstractMatrix{Float64}`: Trilayered feature-source-target network adjacency matrix.

# Example
Refer to tutorial notebooks for examples.

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

spread(F₀::AbstractMatrix{Bool}) = spread(AbstractMatrix{Float64}(F₀))

function spread(namedF₀::NamedMatrix)
    namedW = copy(namedF₀)
    namedW.array = spread(Matrix{Float64}(namedF₀.array))

    return namedW
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
    cutoff!(x::T, α::T, weighted::Bool=false) where {T<:AbstractFloat}

Transform `x` based in SimSpread's similarity cutoff function, overwriting `x`.

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
    cutoff(x::AbstractMatrix{T}, α::T, weighted::Bool=false) where {T<:AbstractFloat}

Transform vector or matrix `M` based in SimSpread's similarity cutoff function.

# Arguments
- `M::AbstractVecOrMat{AbstractFloat}` : Matrix or Vector to transform
- `α::AbstractFloat` : Similarity cutoff
- `weighted::Bool` : Apply weighting function to outcome (default = false)
"""
function cutoff(M::AbstractVecOrMat{T}, α::T, weighted::Bool=false) where {T<:AbstractFloat}
    M′ = deepcopy(M)
    M′ = cutoff.(M′, α, weighted)

    return M′
end

"""
    cutoff!(M::AbstractVecOrMat{T}, α::T, weighted::Bool=false) where {T<:AbstractFloat}

Transform vector or matrix `M` based in SimSpread's similarity cutoff function, overwriting `M`.

# Arguments
- `M::AbstractVecOrMat{AbstractFloat}` : Matrix or Vector to transform
- `α::AbstractFloat` : Similarity threshold
- `weighted::Bool` : Apply weighting function to outcome (default = false)
"""
function cutoff!(M::AbstractVecOrMat{T}, α::T, weighted::Bool=false) where {T<:AbstractFloat}
    cutoff!.(M, α, weighted)
end

"""
    prepare(DT::NamedMatrix, DF::NamedMatrix, Cs::AbstractVector)

Prepare query-feature-source-target network adjacency matrix for *de novo* network-based inference
prediction.

# Arguments
- `DT::NamedMatrix`: Source-target bipartite network adjacency matrix
- `DF::NamedMatrix`: Source-feature bipartite adjacency matrix
- `Cs::AbstractVector`: Source nodes to whom predict targets

# Extended help
This implementation is for k-fold or leave-one-out cross-validation.
"""
function prepare(DT::NamedMatrix, DF::NamedMatrix, Cs::AbstractVector)
    @assert size(DT, 1) == size(DF, 1) "Different number of compounds!"

    # Get names from matrices
    Fs = [f for f in names(DF, 2) if lstrip(f, 'f') ∉ Cs]
    Ds = [d for d in names(DF, 1) if d ∉ Cs]
    Ts = names(DT, 2)

    @assert all(sort(Fs) .!= sort(Ds)) "Features and drugs have the same names!"

    # Get dimensions of network
    Nc = length(Cs)
    Nf = length(Fs)
    Nd = length(Ds)
    Nt = length(Ts)

    # Construct trilayered graph adjacency matrix
    Mcc = zeros(Nc, Nc)
    Mcd = zeros(Nc, Nd)
    Mcf = DF[Cs, Fs].array
    Mct = zeros(Nc, Nt)

    Mdc = Mcd'
    Mdd = zeros(Nd, Nd)
    Mdf = DF[Ds, Fs].array
    Mdt = DT[Ds, Ts].array

    Mfc = Mcf'
    Mfd = Mdf'
    Mff = zeros(Nf, Nf)
    Mft = zeros(Nf, Nt)

    Mtc = Mct'
    Mtd = Mdt'
    Mtf = Mft'
    Mtt = zeros(Nt, Nt)

    A = Matrix(
        [Mcc Mcd Mcf Mct
            Mdc Mdd Mdf Mdt
            Mfc Mfd Mff Mft
            Mtc Mtd Mtf Mtt]
    )

    namedA = NamedArray(A, (vcat(Cs, Ds, Fs, Ts), vcat(Cs, Ds, Fs, Ts)))
    namedB = deepcopy(namedA)
    namedB[Cs, :] .= 0
    namedB[:, Cs] .= 0

    return namedA, namedB
end

"""
    prepare(dts::T, dfs::T) where {T<:Tuple{NamedMatrix,NamedMatrix}}

Prepare query-feature-source-target network adjacency matrix for *de novo* network-based inference
prediction.

# Arguments
- `dts::Tuple{NamedMatrix,NamedMatrix}` : Source-target bipartite graph adjacency matrices
- `dfs::Tuple{NamedMatrix,NamedMatrix}` : Source-feature bipartite graph adjacency matrices
"""
function prepare(dts::T, dfs::T) where {T<:Tuple{NamedMatrix,NamedMatrix}}
    # Unpack matrices tuples
    DT₀, DT₁ = dts
    DF₀, DF₁ = dfs

    @assert size(DT₀, 2) == size(DT₁, 2) "Different number of targets!"
    @assert size(DF₀, 2) == size(DF₁, 2) "Different number of features!"

    # Get names from matrices
    F₀ = names(DF₀, 2)
    D₀ = names(DT₀, 1)
    T₀ = names(DT₀, 2)
    D₁ = names(DT₁, 1)

    @assert all(sort(F₀) .!= sort(D₀)) "Features and drugs have the same names!"

    # Get dimensions of network
    Nd = length(D₀)
    Nc = length(D₁)
    Nt = length(T₀)
    Nf = length(F₀)

    # Construct trilayered graph adjacency matrix
    Mcc = zeros(Nc, Nc)
    Mcd = zeros(Nc, Nd)
    Mcf = DF₁.array
    Mct = zeros(Nc, Nt)
    Mdc = Mcd'
    Mdd = zeros(Nd, Nd)
    Mdf = DF₀.array
    Mdt = DT₀.array
    Mfc = Mcf'
    Mfd = Mdf'
    Mff = zeros(Nf, Nf)
    Mft = zeros(Nf, Nt)
    Mtc = Mct'
    Mtd = Mdt'
    Mtf = Mft'
    Mtt = zeros(Nt, Nt)

    A = Matrix(
        [Mcc Mcd Mcf Mct
            Mdc Mdd Mdf Mdt
            Mfc Mfd Mff Mft
            Mtc Mtd Mtf Mtt]
    )

    namedA = NamedArray(A, (vcat(D₁, D₀, F₀, T₀), vcat(D₁, D₀, F₀, T₀)))
    namedB = deepcopy(namedA)
    namedB[D₁, :] .= 0
    namedB[:, D₁] .= 0

    return namedA, namedB
end
prepare(Xtrain::T, Xtest::T, ytrain::T, ytest::T) where {T<:NamedMatrix} =
    prepare((Xtrain, Xtest), (ytrain, ytest))

"""
    split(DT::NamedArray, k::Int64, rng::Int64)

Split all possible `D` into `k` groups for cross-validation.

# Long description
Split drugs `D` into `k` groups, extract their edges and append to cross-validation group.

# Arguments
- `DT::AbstractMatrix`: Drug-Target rectangular adjacency matrix.
- `k::Int64`: Number of groups to use in data splitting.
- `rng::Int64`: Seed used for data splitting.
"""
function Base.split(G::NamedArray, ngroups::Int64; seed::Int64=1)
    # Get array of drugs in adjacency matrix
    D = names(G, 1)

    # Assign fold to edges of graph
    shuffle!(MersenneTwister(seed), D)
    groups = [[] for _ in 1:ngroups]

    for (i, dᵢ) in enumerate(D)
        foldᵢ = mod(i, ngroups) + 1
        push!(groups[foldᵢ], dᵢ)
    end

    return groups
end

"""
    featurize(M::NamedArray, α::AbstractFloat, weighted::Bool)

Transform, continuous feature into binary feature based in a given cutoff α, either in
a binary or weighted fashion.

# Arguments
- `M::AbtractMatrix`: Continuous feature matrix
- `α::AbstractFloat`: Featurization cutoff
- `weighted::Bool` : Apply weighting function to outcome (default = false)
"""
function featurize(M::NamedArray, α::AbstractFloat, weighted::Bool=false)
    # Filter matrix
    Mf = copy(M)
    Mf.array = cutoff.(M.array, α, weighted)
    setnames!(Mf, ["f$f" for f in names(Mf, 2)], 2)
    return Mf
end

"""
    featurize!(M::NamedArray, α::AbstractFloat, weighted::Bool = false)

Transform, in place, continuous feature into binary feature based in a given cutoff α, either in
a binary or weighted fashion.

# Arguments
- `M::NamedArray` : Continuous feature matrix
- `α::AbstractFloat` : Featurization cutoff
- `weighted::Bool` : Apply weighting function to outcome (default = false)

# References
1. Vigil-Vásquez & Schüller (2022). De Novo Prediction of Drug Targets and Candidates by
   Chemical Similarity-Guided Network-Based Inference. International Journal of Molecular
   Sciences, 23(17), 9666. https://doi.org/10.3390/ijms23179666
"""
function featurize!(M::NamedArray, α::AbstractFloat, weighted::Bool)
    M.array = cutoff.(M.array, α, weighted)
    setnames!(M, ["f$f" for f in names(M, 2)], 2)
end

"""
    predict(I::Tuple{T,T}, ST::T; GPU::Bool=false) where {T<:NamedMatrix}

Predict interactions between query and target nodes using *de novo* network-based inference
model proposed by Wu, et al (2016).

# Arguments
- `I::Tuple{NamedMatrix,NamedMatrix}`: Feature-source-target trilayered adjacency matrices
- `ST::NamedMatrix`: Source-target biaprtite adjacency matrix
- `GPU::Bool`: Use GPU acceleration for calculation (default = false)

# References
1. Wu, et al (2016). SDTNBI: an integrated network and chemoinformatics tool for
   systematic prediction of drug–target interactions and drug repositioning. Briefings in
   Bioinformatics, bbw012. https://doi.org/10.1093/bib/bbw012
2. Vigil-Vásquez & Schüller (2022). De Novo Prediction of Drug Targets and Candidates by
   Chemical Similarity-Guided Network-Based Inference. International Journal of Molecular
   Sciences, 23(17), 9666. https://doi.org/10.3390/ijms23179666
"""
function predict(I::Tuple{T,T}, ST::T; GPU::Bool=false) where {T<:NamedMatrix}
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

    R = F[names(ST, 1), names(ST, 2)]
    return R
end
predict(A::T, B::T, ST::T; GPU::Bool=false) where {T<:NamedMatrix} =
    predict((A, B), ST; GPU=GPU)

"""
    predict(A::T, ST::T; GPU::Bool=false) where {T<:NamedMatrix}

Predict interactions between query and target nodes using *de novo* network-based inference
model proposed by Wu, et al (2016).

# Arguments
- `A::NamedMatrix`: Feature-source-target trilayered adjacency matrix
- `ST::NamedMatrix`: Source-target bipartite adjacency matrix
- `GPU::Bool`: Use GPU acceleration for calculation (default = false)

# References
1. Wu, et al (2016). SDTNBI: an integrated network and chemoinformatics tool for
   systematic prediction of drug–target interactions and drug repositioning. Briefings in
   Bioinformatics, bbw012. https://doi.org/10.1093/bib/bbw012
2. Vigil-Vásquez & Schüller (2022). De Novo Prediction of Drug Targets and Candidates by
   Chemical Similarity-Guided Network-Based Inference. International Journal of Molecular
   Sciences, 23(17), 9666. https://doi.org/10.3390/ijms23179666
"""
function predict(A::T, ST::T; GPU::Bool=false) where {T<:NamedMatrix}
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

    R = F[names(ST, 1), names(ST, 2)]
    return R
end

"""
    clean!(R::NamedArray, A::NamedArray, DT::NamedArray)

Flag, in place, erroneous prediction from cross-validation splitting.

# Arguments
- `R::NamedArray`: Predicted drug-target interactions adjacency matrix
- `A::NamedArray`: *de novo* NBI initial resources adjacency matrix
- `DT::NamedArray`: Ground-truth drug-target interactions adjacency matrix
"""
function clean!(R::NamedArray, A::NamedArray, DT::NamedArray)
    # Clean predictions adjacency matrix R from disconnected targets
    disconnected = 0
    for (tᵢ, k) in zip(names(DT, 2), k(A[names(DT, 1), names(DT, 2)]))
        if k == 0
            disconnected += 1
            R[:, tᵢ] .= -99
        end
    end
end

"""
    save(filepath::String, R::NamedMatrix, DT::NamedMatrix)

Store predictions as a table in the given file path.

# Arguments
- `filepath::String`: Output file path
- `R::NamedArray`: Drug-target predictions matrix
- `DT::NamedMatrix`: Drug-target interactions adjacency matrix

# Extended help
Table format is:
```
fold, compound ID, target ID, score, TP
```
"""
function save(filepath::String, R::NamedMatrix, DT::NamedMatrix)
    # Get name arrays
    Cnames = names(DT, 1)
    Tnames = names(DT, 2)

    # Save file
    open(filepath, "a+") do f
        for Cᵢ in Cnames, Tᵢ in Tnames
            write(f, "$(findfirst(id -> id == Cᵢ, Cnames)); \"$Cᵢ\"; \"$Tᵢ\"; $(R[Cᵢ,Tᵢ]); $(DT[Cᵢ,Tᵢ])\n")
        end
    end
end

"""
    save(filepath::String, fidx::Int64, C::AbstractVector, R::NamedMatrix, DT::NamedMatrix)

Store cross-valudation predictions as a table in the given file path.

# Arguments
- `filepath::String` : Output file path
- `fidx::Int64` : Numeric fold ID
- `C::AbstractVector` : Test set compounds / ligands
- `R::NamedMatrix` : Predicted drug-targetinteraction adjacency  matrix
- `DT::NamedMatrix` : Ground-truth drug-target interactions adjacency matrix

# Extended help
Table format is:
```
Fold ID, Compound ID, Target ID, SimSpread Score, True-Positive state
```
"""
function save(filepath::String, fidx::Int64, C::AbstractVector, R::NamedMatrix, DT::NamedMatrix)
    open(filepath, "a+") do io
        for c in C, t in names(DT, 2)
            write(io, "$fidx, \"$c\", \"$t\", $(R[c,t]), $(DT[c,t])\n")
        end
    end
end
