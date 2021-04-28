# Black-box Discrimination Finder (BDF)
The Black-box Discrimination Finder (a.k.a BDF) is a fairness testing tool which only needs to query the target DNN model/system for predicted labels and   generate individual discriminative samples as many as possible. 

As the same time, we also re-implement the AEQUITAS, ADF, and SG, which are previous fairness testing approaches.

All of the above algorithms are placed in the `Algorithms`.



![OverView](E:\Research\黑盒公平性\Black-box-Discrimination-Finder\OverView.png)

​												         	 	**Fig. Overview of BDF**

### Structure

```
- Algorithms/                 
    - BDF.py            
    - SG.py          
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

All the information of dataset and protected attributes is in `input_config`. 
Here we show how to run the code:

```
cd Black-box-Discrimination-Finder
python Algorithms/xxx.py
```
`xxx` needs to be replaced with the name of a specific algorithm.
We have shown the parameters for the three demos in codes.
If you want to use you own dataset, you need to change the `input_config`.
Notice that the target model is loaded in our code to simulate the black box scenario, if you want to use it in a real scenario, you just need to rewrite part of the codes.
We set them by default the configuration used in our experiments.
If necessary, just modify them.



