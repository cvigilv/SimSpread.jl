module SimSpread

import DelimitedFiles.writedlm
import Downloads.download

using Base
using CUDA
using DelimitedFiles
using MLBase
using NamedArrays
using Random
using StatsBase
using Trapz

include("graphs.jl")
include("performance.jl")
include("simspread.jl")
include("utils.jl")
include("examples.jl")

export writedlm,
    # General utilities
    read_namedmatrix,
    k,

    # SimSpread
    spread,
    cutoff,
    cutoff!,
    featurize,
    featurize!,
    split,
    construct,
    predict,
    clean!,
    save,

    # Performance assessment
    BEDROC,
    AuPRC,
    AuROC,
    f1score,
    mcc,
    accuracy,
    balancedaccuracy,
    recall,
    precision,
    recallatL,
    precisionatL,
    meanperformance,
    meanstdperformance,
    maxperformance,
    validity_ratio,

    # Examples
    getyamanishi
end
