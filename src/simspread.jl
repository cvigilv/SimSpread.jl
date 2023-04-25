"""
    Base.split(DT::NamedArray, k::Int64, rng::Int64)

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
    cutoff(x::T, α::T, weighted::Bool=false) where {T<:AbstractFloat}

Transform `x` based in SimSpread's similarity cutoff function.

# Arguments
- `x::T` : Value to transform
- `α::T` : Similarity matrix
- `weighted::Bool` : Apply weighting function to outcome (default = false)
"""
function cutoff(x::T, α::T, weighted::Bool=false) where {T<:AbstractFloat}
    x′ = deepcopy(x)
    weight = weighted ? x′ : 1.0
    x′ = x′ ≥ α ? weight : 0.0

    return x′
end

"""
    cutoff(x::AbstractMatrix{T}, α::T, weighted::Bool=false) where {T<:AbstractFloat}

Transform `M` based in SimSpread's similarity cutoff function.

# Arguments
- `M::T` : Matrix or Vector to transform
- `α::T` : Similarity threshold
- `weighted::Bool` : Apply weighting function to outcome (default = false)
"""
function cutoff(M::AbstractVecOrMat{T}, α::T, weighted::Bool=false) where {T<:AbstractFloat}
    M′ = deepcopy(M)
    M′ = cutoff.(M′, α, weighted)

    return M′
end

"""
    featurize(M::NamedArray, α::Float64, β::Float64, weighted::Bool)

Convert continuous feature into binary feature based in 2 cutoffs: (i) α for strong-ties and
(ii) β for weak-ties. Weighted version of function weights binary features with it's real
value.

# Arguments
- `M::AbtractMatrix`: Continuous feature matrix
- `α::AbstractFloat`: Strong-ties cutoff
- `weighted::Bool`: Flag for feature weighting using real value
"""
function featurize(M::NamedArray, α::AbstractFloat, weighted::Bool)
    # Filter matrix
    Mf = copy(M)
    Mf.array = cutoff.(M.array, α, weighted)
    setnames!(Mf, ["f$f" for f in names(Mf, 2)], 2)
    return Mf
end

"""
    prepare(DT::NamedMatrix, DF::NamedMatrix, Cs::AbstractVector)

Prepare compound-feature-drug-target network adjacency matrix for *de novo* NBI prediction.

# Arguments
- `DT::NamedMatrix`: Drug-Target adjacency matrix
- `DF::NamedMatrix`: Drug-Feature adjacency matrix
- `Cs::AbstractVector`: Compounds to predict targets

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

Prepare compound-feature-drug-target network adjacency matrix for *de novo* NBI predictions.

# Arguments
- `dts::Tuple{NamedMatrix,NamedMatrix}` : Drug-Target adjacency matrices
- `dfs::Tuple{NamedMatrix,NamedMatrix}` : Drug-Feature adjacency matrices

# Extended help
This implementation is for time-split cross-validation.
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

"""
    spread(F₀::AbstractMatrix{Float64})

# Arguments
- `F₀::AbstractMatrix{Float64}`: Trilayered SimSpread network adjacency matrix

# References
1. Wu, et al (2016). SDTNBI: an integrated network and chemoinformatics tool for
   systematic prediction of drug–target interactions and drug repositioning. Briefings in
   Bioinformatics, bbw012. https://doi.org/10.1093/bib/bbw012
2. Vigil-Vásquez & Schüller (2022). De Novo Prediction of Drug Targets and Candidates by
   Chemical Similarity-Guided Network-Based Inference. International Journal of Molecular
   Sciences, 23(17), 9666. https://doi.org/10.3390/ijms23179666
"""
function spread(F₀::AbstractMatrix{Float64})
    # Transfer matrix
    W = F₀ ./ k(F₀)
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
    predict(A::NamedMatrix, B::NamedMatrix, names::Tuple; GPU::Bool)

TODO: Add short description to `predict`

# Arguments
- `A::NamedMatrix`: Compound-Feature-Drug-Target initial resources adjacency matrix
- `B::NamedMatrix`: Feature-Drug-Target initial resources adjacency matrix
- `names::Tuple`: Rows & columns named indices
- `GPU::Bool`: (default = false)
"""
function predict(A::T, B::T, names::Tuple; GPU::Bool=false) where {T<:NamedMatrix}
    # GPU calculations helper functions
    _useGPU(x::AbstractArray) = GPU ? CuArray{Float32}(x) : x

    # Target prediction using NBI
    W = spread(B.array)
    F = begin
        F = copy(A)

        Aarr = _useGPU(A.array)
        Warr = _useGPU(W)
        F.array = Aarr * Warr^2

        # Free GPU memory
        if GPU
            CUDA.unsafe_free!(Aarr)
            CUDA.unsafe_free!(Warr)
        end
        return F
    end

    R = F[Symbol.(names[1]), Symbol.(names[2])]

    return R
end

"""
    predict(I::Tuple{T,T}, DT::T; GPU::Bool) where {T<:NamedArray}

TODO: Add short description to `predict`

# Arguments
- `I::Tuple{NamedMatrix,NamedMatrix}`: *De novo* initial resources adjacency matrices
- `DT::NamedMatrix`: Drug-target adjacency matrix
- `GPU::Bool`: Use GPU acceleration for calculation (default = false)
"""
function predict(I::Tuple{T,T}, DT::NamedMatrix; GPU::Bool=false) where {T<:NamedMatrix}
    # GPU calculations helper functions
    _useGPU(x::AbstractArray) = GPU ? CuArray{Float32}(x) : x

    # Target prediction using NBI
    A, B = I
    W = spread(B.array)
    F = begin
        F = copy(A)

        Aarr = _useGPU(A.array)
        Warr = _useGPU(W)
        F.array = Aarr * Warr^2

        # Free GPU memory
        if GPU
            CUDA.unsafe_free!(Aarr)
            CUDA.unsafe_free!(Warr)
        end

        return F
    end

    R = F[names(DT, 1), names(DT, 2)]
    return R
end

"""
    clean!(R::NamedArray, A::NamedArray, DT::NamedArray)

Flag errors from cross-validation splitting in place.

# Arguments
- `R::NamedArray`: Predicted drug-target interactions adjacency matrix
- `A::NamedArray`: *de novo* NBI initial resources adjacency matrix
- `DT::NamedArray`: Ground-truth drug-target interactions adjacency matrix
"""
function clean!(R::NamedArray, A::NamedArray, DT::NamedArray)
    targets = names(DT, 2)

    for (tᵢ, degree) in zip(targets, k(A[targets, :]))
        if degree == 0
            R[:, tᵢ] .= -99
            R[tᵢ, :] .= -99
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
