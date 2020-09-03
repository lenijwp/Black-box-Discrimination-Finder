The configuration file here contains the information that must be provided before running the codes:
```
params: the size of input features.
inputs_bounds: the upper and lower boundaries of each feature.
model_path: the path of the target model.(We load the target model and only query for the label. For the real systems, it's easy to migrate.)
data_path: the path of trainning data. It's needed for ADF.
protected_attri & pro_at: the protected attribute.

The above parameters need to be set in the Dat.
```
