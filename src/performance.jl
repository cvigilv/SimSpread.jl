"""
    BEDROC(y, yhat; rev = true, α = 20.0)

The Boltzmann Enhanced Descrimination of the Receiver Operator Characteristic (BEDROC) score
is a modification of the Receiver Operator Characteristic (ROC) score that allows for a factor
of *early recognition*.

Score takes a value in interval [0, 1] indicating degree to which the predictive model employed
detects (early) the positive class.

# Arguments
- `y::AbstractArray`: Binary class labels. 1 for positive class, 0 otherwise.
- `̂yhat::AbstractArray`: Prediction values.
- `rev::Bool`: True if high values of ``yhat`` correlates to positive class (default = true).
- `α::AbstractFloat`: Early recognition parameter (default = 20.0).

# References
1. Truchon, J.-F., & Bayly, C. I. (2007). Evaluating Virtual Screening Methods:  Good and
Bad Metrics for the “Early Recognition” Problem. Journal of Chemical Information and Modeling,
47(2), 488–508. https://doi.org/10.1021/ci600426e
"""
function BEDROC(y::AbstractVector{Bool}, yhat::AbstractVector; rev::Bool=true, α::AbstractFloat=20.0)
    @assert length(y) == length(yhat) "The number of scores must be equal to the number of labels"

    N = length(y)
    n = sum(y .== 1)

    order = sortperm(yhat; rev=rev)
    rᵢ = findall(!iszero, y[order] .== 1)
    s = sum(exp.(-α * rᵢ / N))

    Rₐ = n / N
    rand_sum = Rₐ * (1 - exp(-α)) / (exp(α / N) - 1)
    fac = (Rₐ * sinh(α / 2) / (cosh(α / 2) - cosh(α / 2 - α * Rₐ)))
    cte = 1 / (1 - exp(α * (1 - Rₐ)))

    return s * fac / rand_sum + cte
end

"""
    AuROC(y::AbstractArray{Bool}, yhat::AbstractVector{Number})

Area under the Receiver Operator Characteristic curve using the trapezoidal rule.

# Arguments
- `y::AbstractArray`: Binary class labels. 1 for positive class, 0 otherwise.
- `̂yhat::AbstractArray`: Prediction values.
"""
function AuROC(y::AbstractVector{Bool}, yhat::AbstractVector)
    @assert length(y) == length(yhat) "The number of scores must be equal to the number of labels"

    # Calculate confusion matrices for each threshold
    thresholds = sort(unique(yhat))
    confusion = roc(y, yhat, thresholds)

    # Calculate true positive and false positive rates
    tpr = true_positive_rate.(confusion)
    fpr = false_positive_rate.(confusion)

    # Calculate area under the curve
    auc = abs(trapz(fpr, tpr))
    return auc
end

"""
    AuPRC(y::AbstractArray{Bool}, yhat::AbstractArray{Number})

Area under the Precision-Recall curve using the trapezoidal rule.

# Arguments
- `y::AbstractArray`: Binary class labels. 1 for positive class, 0 otherwise.
- `̂yhat::AbstractArray`: Prediction values.
"""
function AuPRC(y::AbstractVector{Bool}, yhat::AbstractVector)
    @assert length(y) == length(yhat) "The number of scores must be equal to the number of labels"

    # Calculate confusion matrices for each threshold
    thresholds = sort(unique(yhat))
    confusion = roc(y, yhat, thresholds)

    # Calculate true positive and false positive rates
    recalls = recall.(confusion)
    precisions = precision.(confusion)

    # Calculate area under the curve
    auc = abs(trapz(precisions, recalls))

    return auc
end

# def eP(df, L, col_score="score", col_pred="tp"):
#     P_L = precision_at_L(df, L, col_score, col_pred)
#     D = sum(df[col_pred].values)
#     M = len(df["ligand"].unique())
#     N = len(df["target"].unique())
#
#     return P_L * M * N / D
#
#
# def eR(df, L, col_score="score", col_pred="tp"):
#     R_L = recall_at_L(df, L, col_score, col_pred)
#     N = len(df["target"].unique())
#
#     return R_L * N / L

"""
    f1score(tn::T, fp::T, fn::T, tp::T) where {T<:Integer)

The harmonic mean between precision and recall

# Arguments
- `tn::Integer` True negatives
- `fp::Integer` False postives
- `fn::Integer` False negatives
- `tp::Integer` True positives
"""
function f1score(tn::T, fp::T, fn::T, tp::T) where {T<:Integer}
    numerator = tp
    denominator = tp + 0.5 * (fp + fn)

    if denominator == 0
        return NaN
    else
        return numerator / denominator
    end
end
f1score(confusion::ROCNums) = f1score(confusion.tn, confusion.fp, confusion.fn, confusion.tp)


"""
    mcc(tn::T, fp::T, fn::T, tp::T) where {T<:Integer}
Matthews correlation coefficient, a special case of the phi coeficient
Performance metric used for overcoming the class imbalance issues

# Arguments
- `tn::Integer` True negatives
- `fp::Integer` False postives
- `fn::Integer` False negatives
- `tp::Integer` True positives

# Reference
1.Chicco, D., Jurman, G. The advantages of the Matthews correlation coefficient
(MCC) over F1 score and accuracy in binary classification evaluation.
BMC Genomics 21, 6 (2020).
"""
function mcc(tn::T, fp::T, fn::T, tp::T) where {T<:Integer}
    numerator = (tp * tn) - (fp * fn)
    denominator = sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    if denominator == 0
        return NaN
    else
        return numerator / denominator
    end
end
mcc(confusion::ROCNums) = mcc(confusion.tn, confusion.fp, confusion.fn, confusion.tp)


"""
    accuracy(tn::T, fp::T, fn::T, tp::T) where {T<:Integer}
The number of all correct predictions divided by the total predicitions

# Arguments
- `tn::Integer` True negatives
- `fp::Integer` False postives
- `fn::Integer` False negatives
- `tp::Integer` True positives
"""
function accuracy(tn::T, fp::T, fn::T, tp::T) where {T<:Integer}
    numerator = tp + tn
    denominator = (tp + tn) + (fp + fn)

    if denominator == 0
        return NaN
    else
        return numerator / denominator
    end
end
accuracy(confusion::ROCNums) = accuracy(confusion.tn, confusion.fp, confusion.fn, confusion.tp)


"""
    balancedaccuracy(tn::T, fp::T, fn::T, tp::T) where {T<:Integer}

The arithmetic mean of sensitivity and specificity,
its use case is when dealing with imbalanced data

# Arguments
- `tn::Integer` True negatives
- `fp::Integer` False postives
- `fn::Integer` False negatives
- `tp::Integer` True positives
"""
function balancedaccuracy(tn::T, fp::T, fn::T, tp::T) where {T<:Integer}
    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)

    return (tpr + tnr) / 2
end
balancedaccuracy(confusion::ROCNums) = balancedaccuracy(confusion.tn, confusion.fp, confusion.fn, confusion.tp)


"""
    recall(tn::T, fp::T, fn::T, tp::T) where {T<:Integer}
The fraction of positive samples correctly predicted as postive

# Arguments
- `tn::Integer` True negatives
- `fp::Integer` False postives
- `fn::Integer` False negatives
- `tp::Integer` True positives
"""
function recall(tn::T, fp::T, fn::T, tp::T) where {T<:Integer}
    p = tp + fn

    if p == 0
        return NaN
    else
        return tp / p
    end
end
recall(confusion::ROCNums) = recall(confusion.tn, confusion.fp, confusion.fn, confusion.tp)


"""
    precision(tn::T, fp::T, fn::T, tp::T) where {T<:Integer}
The fraction of positive predictions that are correct

# Arguments
- `tn::Integer` True negatives
- `fp::Integer` False postives
- `fn::Integer` False negatives
- `tp::Integer` True positives
"""
function precision(tn::T, fp::T, fn::T, tp::T) where {T<:Integer}
    d = tp + fp

    if d == 0
        return NaN
    else
        return tp / d
    end
end
precision(confusion::ROCNums) = precision(confusion.tn, confusion.fp, confusion.fn, confusion.tp)

function performanceatL(
    y::AbstractVector{Bool},
    yhat::AbstractVector,
    metric::Function,
    L::Integer=20
)
    @assert length(y) == length(yhat) "The number of scores must be equal to the number of labels"

    # Sort predictions by score
    order = sortperm(yhat, rev=true)

    # Cuantify performance using top L predictions
    yₗ = Int64.(first(y[order], L))
    yhatₗ = Int64.(ones(L))

    return metric(roc(yₗ, yhatₗ))
end

"""
    maxperf(confusion::ROCNums, metric::Function)

Get maximum performance of a given metric over a set of confusion matrices.

"""
function maxperformance(confusion, metric::Function)
    performance = metric.(confusion)
    return maximum(performance)
end

"""
    meanperf(confusion::ROCNums, metric::Function)

Get mean performance of a given metric over a set of confusion matrices.
"""
function meanperformance(confusion::ROCNums, metric::Function)
    performance = metric.(confusion)
    return mean(performance)
end

"""
    meanstdperf(confusion::ROCNums, metric::Function)

Get mean and standard deviation performance of a given metric over a set of confusion matrices.
"""
function meanstdperformance(confusion::ROCNums, metric::Function)
    performance = metric.(confusion)
    return mean_and_std(performance)
end
