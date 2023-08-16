```@meta
CurrentModule = SimSpread
```

# API
## Core
```@docs
spread
cutoff
cutoff!
featurize
featurize!
construct
predict
save
```
## Cross-validation 
```@docs
split
clean!
```

## Performance assessment
Several evaluation metrics are implemented in the package, which can be that can be classified
into three groups: (i) overall performance, (ii) early recognition, and (iii) binary prediction 
performance.

### Overall performance
This metrics represent classical evaluation metrics that make use of the complete list of
prediction to assess predictive performance.
```@docs
AuPRC
AuROC
```

### Early recognition performance
Due to the roots of SimSpread (target prediction in drug discovery), we include evaluation
metrics that aim to assess predictive performance of the best predictions obtained from a model.

In virtual screening, only the best predictions obtained from a model are selected for
posterior experimental validation. Therefore, understanding the predictive performance of a
model for these predictions is essential to (1) make accurate predictions that will translate
to biological activity and (2) understand the limitations of the model. The metrics discussed
here can be evaluated at a given cut-off rank, considering only the topmost results returned
by the predictive method, hence informing of the predictive performance of the model for only
the best predictions.

```@docs
recallatL
precisionatL
BEDROC
```

### Binary prediction performance
A common practice in predictive modelling is to assign a score or probability threshold
for the predictions obtained from a model and manually select or cherry-pick predictions for
validation. In order to evaluate the predictive performance under this paradigm, we implement
a series of metrics that are meant for binary classification, that is, a link exists or not 
based in a given threshold, from which statistical moments can be calculated to retrieve a 
notion of predictive performance as a decision boundary changes.
```@docs
f1score
mcc
accuracy
balancedaccuracy
recall
precision
meanperformance
meanstdperformance
maxperformance
```

### Other metrics
```@docs
validity_ratio
```

## Miscellaneous utilities
```@docs
read_namedmatrix
k
getyamanishi
```
