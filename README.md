# ML_template
A template for building machine models with different preprocess methods and generate report
## Dependency
Package                       Version        
----------------------------- ---------------
imblearn                      0.0            
joblib                        1.0.0          
matplotlib                    3.2.2          
numpy                         1.19.5         
pandas                        1.1.5          
scikit-learn                  0.22.2.post1
tabulate                      0.8.7          
Python			                  3
## Info
template.ipynb is an old version, but it still works.
predict_template.py is the new version. I have currently refactored the template.ipynb with object-oriented and design pattern.
So, if you want to add new prediction models, it should be easier with changing less code.
## Alert
1. Some ML models has no functions like fit, predict, predict_proba, and so on. As a result. If this code raise NoAttributeException, 
you may need to create a new class with inheritting Model class, and overwrite the origin functions.
2. One instance of metric class will save all experiments who have used it. So, if you have lots of experiments and need to seperate
them apart, you should instance more than once.
## Future Work
The preprocessing steps is controlled by parameters and predifined in class. In the future, I will use the factory method to 
synthesis the prepreocessor instead of predefine it.
## Usage
1. Loading data: Loading data with csv file, and input data and ground truth is seperated. If raising any error, this code will generate demo data aotomatically.
2. The official usageis attached below.
```python
generater = DataGenerater()
preprocessor = DataPreprocessor(generater.X, generater.Y)
preprocessor.preprocess(pca_n_components=5)
metric = Metrics()
workflow = WorkFlow(preprocessor.x_train, preprocessor.y_train, preprocessor.x_test, preprocessor.y_test, preprocessor.steps)
workflow.addModel(Model(RandomForestClassifier(), metric))
workflow.addModel(Model(DecisionTreeClassifier(), metric))
workflow.addModel(Model(KNeighborsClassifier(n_neighbors=3), metric))
workflow.fitAllModel()
workflow.testAllModel()
metric.show_results()
metric.plotConfusionMatrix()
```
## Download
git clone https://github.com/MasterLKH180cm/fastMLtemplate.git
## Advanced
You can use any other ML packages to fit your own need.
If you want to use "deep learning", you can just add training code and validation code here or just add validation code.
