# Black-box Discrimination Finder (BDF)
There is a black-box fairness testing method based on shadow models, which contains two different strategis.
We implement our method as `BDF_global` and `BDF_local`.
As the same time, we re-implement the AEQUITAS and ADF in `AEQ` and `ADF`.
All of the above algorithms are placed in the `Algorithms`.


### Overview of BDF
![image](https://github.com/lenijwp/Black-box-Discrimination-Finder/blob/master/Overview.png)


### Structure
```
- Algorithms/                 
    - BDF_local.py            
    - BDF_global.py          
    - AEQ.py                  
    - ADF.py                  
- Utils/                      
    - input_config.py         
- Demo/
    - Datasets/
    - Shadowmodels/
    - Targetmodels/
    - Results/
```

### Dependencies
```
python==3.6.10
tensorflow==1.12.0
keras==2.2.4
```


### Demo
We provide three datasets and the three target models trained separately on them, which are used in our experiments as runnable demos.
We set the target model from `census` and protected attribute `gender` defaultly in `input_config`.
Here we show how to run the code:

```
cd Black-box-Discrimination-Finder
python Algorithms/xxx.py
```
`xxx` needs to be replaced with the name of a specific algorithm.
`input_config` contains parameters that must be provided about the target model.
We have shown the parameters for the three demos in this file.
If you want to use other cases, you need to change the `input_config`.
Notice that the target model is loaded in our code to simulate the black box scenario, if you want to use it in a real scenario, you need to rewrite part of the code.
The other parameters are placed directly in the algorithm codes.
We set them by default the configuration used in our experiments.
If necessary, just modify them.



