#Loading the data using pandas library
#The dataset used in this story is publicly available and was created by Dr. William H. Wolberg, physician at the University
#Of Wisconsin Hospital at Madison, Wisconsin, USA. To create the dataset Dr. Wolberg used fluid samples, taken from patients
#with solid breast masses and an easy-to-use graphical computer program called Xcyt, which is capable of perform the analysis
#of cytological features based on a digital scan. The program uses a curve-fitting algorithm, to compute ten features from
#each one of the cells in the sample

import pandas as pd
import pandas as pd
import seaborn as sns
import  matplotlib.pyplot as plt
cancer = pd.read_csv('cancer_dataset.csv')
#selects the first twelve instances of the data
cancero= cancer.iloc[:, 1:12]
# prints the data shape, i.e the number of cells and rows
print(cancer.shape)
#select features and labels from the data set
#This is done by excluding instance 0-ID and instance 11-Label
features, labels= cancero.iloc[:, 1:10], cancero.loc[:,['diagnosis']]
#print(features)
#print(labels)
#Vizualizing the data

#Generating distribution plot
#Generating a matrix-type plot which plots area - perimeter - texture - radius
plt.style.use(['ggplot']) 
fig= sns.pairplot(cancero.loc[:,'diagnosis':'area_mean'], hue="diagnosis");
fig.savefig('mytree.png')
#plt.show()

#Generating a swarm plot in which which plots all the features together 
#To create such a graph, we need to normalize the data 
#This is done by applying the formula (value - Mean )/ std
data = pd.DataFrame(features)
data_n_2 = (data - data.mean()) / (data.std())  
data = pd.concat([labels,data_n_2.iloc[:]],axis=1)
data = pd.melt(data,id_vars="diagnosis",
                    var_name="features",
                    value_name='value')
plt.figure(figsize=(10,10))
sns.swarmplot(x="features", y="value", hue="diagnosis", data=data)
plt.xticks(rotation=45);
plt.show()
