# Black-box Discrimination Finder (BDF)
There is a black-box fairness testing method based on shadow models, which contains two different strategis.
We implement our method as `BDF_global` and `BDF_local`.<font color='red'> aaa </font>
As the same time, we re-implement the AEQUITAS and ADF in `AEQ` and `ADF`.
All of the above algorithms are placed in the `Algorithms`.
Black-box Fairness Testing based on Shadow Models

### Dependencies
```
python==3.6.10

tensorflow==1.12.0

keras==2.2.4
```
### Demo
We provide three datasets and the three target models trained separately on them, which are used in our experiments as runnable demos.
### Overview of BDF
![image](https://github.com/lenijwp/Black-box-Discrimination-Finder/blob/master/Overview.png)


