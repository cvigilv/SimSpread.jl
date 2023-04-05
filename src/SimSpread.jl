module SimSpread

using Base
using CUDA
using DelimitedFiles
using MLBase
using NamedArrays
using Random
using StatsBase
using Trapz

import DelimitedFiles.writedlm

include("graphs.jl")
include("performance.jl")
include("simspread.jl")
include("utils.jl")

export writedlm,
    # General utilities
    read_namedmatrix,
    k,

    # SimSpread
    spread,
    cutoff,
    pcutoff,
    featurize,
    pfeaturize,
    split,
    prepare,
    prepare!,
    predict,
    predict!,
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
    maxperformance
end
