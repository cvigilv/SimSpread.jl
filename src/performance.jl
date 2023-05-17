"""
    BEDROC(y::AbstractVector{Bool}, yhat::AbstractVector; rev::Bool=true, α::AbstractFloat=20.0)

The Boltzmann Enhanced Descrimination of the Receiver Operator Characteristic (BEDROC) score
is a modification of the Receiver Operator Characteristic (ROC) score that allows for a factor
of *early recognition*.

Score takes a value in interval [0, 1] indicating degree to which the predictive model employed
detects (early) the positive class.

# Arguments
- `y::AbstractArray`: Binary class labels. 1 for positive class, 0 otherwise.
- `̂yhat::AbstractArray`: Prediction scores.
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
    AuROC(y::AbstractVector{Bool}, yhat::AbstractVector)

Area under the Receiver Operator Characteristic curve using the trapezoidal rule.

# Arguments
- `y::AbstractArray`: Binary class labels. 1 for positive class, 0 otherwise.
- `̂yhat::AbstractArray`: Prediction scores.
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
    AuPRC(y::AbstractVector{Bool}, yhat::AbstractVector)

Area under the Precision-Recall curve using the trapezoidal rule.

# Arguments
- `y::AbstractArray`: Binary class labels. 1 for positive class, 0 otherwise.
- `̂yhat::AbstractArray`: Prediction scores.
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
    auc = abs(trapz(recalls, precisions))

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
    f1score(tn::T, fp::T, fn::T, tp::T) where {T<:Integer}

The harmonic mean between precision and recall

# Arguments
- `tn::Integer` True negatives
- `fp::Integer` False postives
- `fn::Integer` False negatives
- `tp::Integer` True positives
"""
function f1score(tn::T, fp::T, fn::T, tp::T) where {T<:Integer}
    @assert tn + fp + fn + tp > 0 "Confusion matrix sums zero!"

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
    mcc(a::T, b::T, ϵ::AbstractFloat = 0.0001) where {T<:Integer}

Matthews correlation coefficient using calculus approximation for when
FN+TN, FP+TN, TP+FN or TP+FP equals zero.

# Arguments
- `a::Integer` = Value of position `a` in confusion matrix
- `b::Integer` = Value of position `b` in confusion matrix
- `ϵ::AbstractFloat` = Approximation coefficient (default = floatmin(Float64))

# Extended help
The confusion matrix in a binary prediction is comprised of 4 distinct positions:
```
                    | Predicted positive     Predicted negative
    ----------------+--------------------------------------------
    Actual positive |  True positives (TP)   False negatives (FN)
    Actual negative | False positives (FP)    True negatives (TN)
```

In the case a row or column of the confusion matrix equals zero, MCC is undefined.
Therefore, to correctly use MCC with this approximation, arguments `a` and `b` are
defined as follows:

- If "Predictive positive" column is zero, `a` is TN and `b` is FN
- If "Predictive negative" column is zero, `a` is TP and `b` is FP
- If "Actual positive" row is zero, `a` is TN and `b` is FP
- If "Actual negative" row is zero, `a` is TP and `b` is FN

# Reference
1.Chicco, D., Jurman, G. The advantages of the Matthews correlation coefficient
(MCC) over F1 score and accuracy in binary classification evaluation.
BMC Genomics 21, 6 (2020).
"""
function mcc(a::T, b::T, ϵ::AbstractFloat=floatmin(Float64)) where {T<:Integer}
    return (a*ϵ - b*ϵ)/ sqrt((a+b)*(a+ϵ)*(b+ϵ)*(ϵ+ϵ))
end

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
    @assert tn + fp + fn + tp > 0 "Confusion matrix sums zero!"

    # Calculate row and column-wise sum of confusion matrix
    p_pred = tp + fp
    n_pred = fn + tn
    p_actual = tp + fn
    n_actual = fp + tn

    # Use approximation if the sum of either a row or column is zero
    # NOTE: Based in the reference bellow, some confusion matrices can take an
    #       indefinite form 0/0, therefore to ensure correct handling of those
    #       cases we opt to reimplement the calculus approximation instead of
    #       returning a "missing" or "nan" value. Please refer to the reference
    #       section in the docstrings for more information regarding this case.
    if p_pred == 0
        return mcc(tn, fn)
    elseif n_pred == 0
        return mcc(tp, fp)
    elseif p_actual == 0
        return mcc(tn, fp)
    elseif n_actual == 0
        return mcc(tp, fn)
    else
        numerator = (tp * tn) - (fp * fn)
        denominator = sqrt(p_pred * n_pred * p_actual * n_actual)

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
    @assert tn + fp + fn + tp > 0 "Confusion matrix sums zero!"

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
    @assert tn + fp + fn + tp > 0 "Confusion matrix sums zero!"

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
    @assert tn + fp + fn + tp > 0 "Confusion matrix sums zero!"

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
    @assert tn + fp + fn + tp > 0 "Confusion matrix sums zero!"

    d = tp + fp

    if d == 0
        return NaN
    else
        return tp / d
    end
end
precision(confusion::ROCNums) = precision(confusion.tn, confusion.fp, confusion.fn, confusion.tp)

"""
    recallatL(y, yhat, grouping, L)

Get recall@L as proposed by Wu, et al (2017).

# Arguments
- `y::AbstractVector`: Binary class labels. 1 for positive class, 0 otherwise.
- `̂yhat::AbstractVector`: Prediction score.
- `grouping::AbstractVector`: Group labels.
- `L::Integer`: Length to consider to calculate metrics (default = 20).
"""
function recallatL(y, yhat, grouping, L::Integer=20)
    @assert L > 0 "Please use a list length greater than 0 (L > 0)"
    @assert length(y) == length(yhat) "Number of predictions and labels don't match"
    @assert length(y) > L "Number of labels is less than length (L > y)"
    @assert length(yhat) > L "Number of predictions is less than length (L > yhat)"

    performance = []
    for group in unique(grouping)
        # Get prediction-label pairs for group
        y_g = y[grouping.==group]
        yhat_g = yhat[grouping.==group]

        # Sort predictions by score
        order = sortperm(yhat_g, rev=true)
        y_g = y_g[order]
        yhat_g = y_g[order]

        # Calculate recall@n for given group
        Xi = sum(y_g)
        Xi_L = sum(first(y_g, L))

        if Xi > 0
            push!(performance, Xi_L / Xi)
        else
            push!(performance, NaN)
        end
    end

    return mean(performance)
end

"""
    precisionatL(y, yhat, grouping, L::Integer=20)

Get precision@L as proposed by Wu, et al (2017).

# Arguments
- `y::AbstractVector`: Binary class labels. 1 for positive class, 0 otherwise.
- `̂yhat::AbstractVector`: Prediction score.
- `grouping::AbstractVector`: Group labels.
- `L::Integer`: Length to consider to calculate metrics (default = 20).
"""
function precisionatL(y, yhat, grouping, L::Integer=20)
    @assert L > 0 "Please use a list length greater than 0 (L > 0)"
    @assert length(y) == length(yhat) "Number of predictions and labels don't match"
    @assert length(y) > L "Number of labels is less than length (L > y)"
    @assert length(yhat) > L "Number of predictions is less than length (L > yhat)"

    performance = []
    for group in unique(grouping)
        # Get prediction-label pairs for group
        y_g = y[grouping.==group]
        yhat_g = yhat[grouping.==group]

        # Sort predictions by score
        order = sortperm(yhat_g, rev=true)
        y_g = y_g[order]
        yhat_g = y_g[order]

        # Calculate precisionj@n for given group
        Xi_L = sum(first(y_g, L))
        push!(performance, Xi_L / L)
    end

    return mean(performance)
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
  maxperformance(y::AbstractVector, yhat::AbstractVector, metric::Function)

Brief description of intended functionality

# Arguments
- `y::AbstractVector` : 
- `yhat::AbstractVector` : 
- `metric::Function` : 

# Extended help
Longer description of intended functionality

# References

"""
function maxperformance(y::AbstractVector, yhat::AbstractVector, metric::Function)
    thresholds = sort(unique(yhat))
    confusion = roc(y, yhat, thresholds)

    return maxperformance(confusion, metric)
end

"""
    meanperf(confusion::ROCNums, metric::Function)

Get mean performance of a given metric over a set of confusion matrices.
"""
function meanperformance(confusion::ROCNums, metric::Function)
    performance = metric.(confusion)
    return mean(performance)
end

function meanperformance(y::AbstractVector, yhat::AbstractVector, metric::Function)
    thresholds = sort(unique(yhat))
    confusion = roc(y, yhat, thresholds)

    return meanperformance(confusion, metric)
end

"""
    meanstdperf(confusion::ROCNums, metric::Function)

Get mean and standard deviation performance of a given metric over a set of confusion matrices.
"""
function meanstdperformance(confusion::ROCNums, metric::Function)
    performance = metric.(confusion)
    return mean_and_std(performance)
end

function meanstdperformance(y::AbstractVector, yhat::AbstractVector, metric::Function)
    thresholds = sort(unique(yhat))
    confusion = roc(y, yhat, thresholds)

    return meanstdperformance(confusion, metric)
end

"""
    count_zeros(yhat::AbstractVector)
Counts the number of empty predictions (score equal to 0) in a array of scores
"""
function count_zeros(yhat::AbstractVector)
    count = 0
    for x in yhat
        if x == 0
            count += 1
        end
    end
    return count
end
