# -*- coding: utf-8 -*-
"""Liner regression drug solubility

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1acgM4KyuP0XfnZzcU0IkWCxWqtgziMdW

# 1. Let's install some of the packages:
"""

! wget https://repo.anaconda.com/miniconda/Miniconda3-py37_4.8.2-Linux-x86_64.sh
! chmod +x Miniconda3-py37_4.8.2-Linux-x86_64.sh
! bash ./Miniconda3-py37_4.8.2-Linux-x86_64.sh -b -f -p /usr/local
! conda install -c rdkit rdkit -y
import sys
sys.path.append('/usr/local/lib/python3.7/site-packages/')

"""# 2. Now let's download the dataset for drug discovery"""

! wget https://raw.githubusercontent.com/dmy-collab/shared-projects/main/Linear%20regression%20drug%20solubility/compounds_list.csv

"""# 2.1 Reading the file"""

import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('max_seq_item', None)
pd.set_option('display.width', 1000)

sol = pd.read_csv('compounds_list.csv')

"""> Initial reviev

"""

print(sol.head())
print(sol.shape)
nulls = sol.isnull().sum().to_frame()   # searching for missing values
for index, row in nulls.iterrows():
    print(index, row[0])

"""# 3. Calculate molecular descriptors in rdkit

# 3.1 Importing RDKIT for further convertion of molecule from the SMILES string to an rdkit object
"""

from rdkit import Chem

mol_list = [Chem.MolFromSmiles(element) for element in sol.SMILES]

print(len(mol_list))

"""# 3.2. Calculate molecular descriptors

To predict LogS (log of the aqueous solubility) we will need to use use of 4 molecular descriptors:

1.   cLogP (Octanol-water partition coefficient)
2.   MW (Molecular weight)
3.   RB (Number of rotatable bonds)
4.   AP (Aromatic proportion = number of aromatic atoms / total number of heavy atoms)

Rdkit readily computes only the first 3. As for the AP descriptor, we will calculate this by manually computing the ratio of the number of aromatic atoms to the total number of heavy atoms which rdkit can compute.

# 3.2.1. LogP, MW and RB
"""

import numpy as np
from rdkit.Chem import Descriptors

def generate(smiles, verbose=False):

    moldata= []
    for elem in smiles:
        mol=Chem.MolFromSmiles(elem) 
        moldata.append(mol)
       
    baseData= np.arange(1,1)
    i=0  
    for mol in moldata:        
       
        desc_MolLogP = Descriptors.MolLogP(mol)
        desc_MolWt = Descriptors.MolWt(mol)
        desc_NumRotatableBonds = Descriptors.NumRotatableBonds(mol)
           
        row = np.array([desc_MolLogP,
                        desc_MolWt,
                        desc_NumRotatableBonds])   
    
        if(i==0):
            baseData=row
        else:
            baseData=np.vstack([baseData, row])
        i=i+1      
    
    columnNames=["MolLogP","MolWt","NumRotatableBonds"]   
    descriptors = pd.DataFrame(data=baseData,columns=columnNames)
    
    return descriptors

df = generate(sol.SMILES)
df

"""# 3.2.2. Aromatic proportion
> # 3.2.1.1. Number of aromatic atoms


Here, we will create a custom function to calculate the Number of aromatic atoms. With this descriptor we can use it to subsequently calculate the AP descriptor.
"""

def AromaticAtoms(m):
  aromatic_atoms = [m.GetAtomWithIdx(i).GetIsAromatic() for i in range(m.GetNumAtoms())]
  aa_count = []
  for i in aromatic_atoms:
    if i==True:
      aa_count.append(1)
  sum_aa_count = sum(aa_count)
  return sum_aa_count

desc_AromaticAtoms = [AromaticAtoms(element) for element in mol_list]
desc_AromaticAtoms

"""# 3.2.1.2. Number of heavy atoms
Here, we will use an existing function for calculating the Number of heavy atoms.
"""

desc_HeavyAtomCount = [Descriptors.HeavyAtomCount(element) for element in mol_list]
desc_HeavyAtomCount

"""# 3.2.1.3. Computing the Aromatic Proportion (AP) descriptor"""

desc_AromaticProportion = [AromaticAtoms(element)/Descriptors.HeavyAtomCount(element) for element in mol_list]
desc_AromaticProportion

"""*Converting to dataframe:*"""

df_desc_AromaticProportion = pd.DataFrame(desc_AromaticProportion, columns=['AromaticProportion'])
df_desc_AromaticProportion

"""# 3.3. Combining all computed descriptors into 1 dataframe

# X matrix (features for further modeling)
"""

X = pd.concat([df,df_desc_AromaticProportion], axis=1)
X

"""# 3.4. Y matrix"""

sol.head()

"""*Assigning the second column (index 1) to the Y matrix (labels for further modeling)*"""

Y = sol.iloc[:,1]
Y

"""Splitting the dataset into the Training set and Test set:"""

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=101)

"""# Training the Linear Regression model and evaluation:"""

from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix, classification_report

model = linear_model.LinearRegression()
model.fit(X_train, Y_train)

"""# *Predict X_train*"""

Y_pred_train = model.predict(X_train)

# Commented out IPython magic to ensure Python compatibility.
print('Coefficients:', model.coef_)
print('Intercept:', model.intercept_)
print('Mean squared error (MSE): %.2f'
#       % mean_squared_error(Y_train, Y_pred_train))
print('Coefficient of determination (R^2): %.2f'
#       % r2_score(Y_train, Y_pred_train))

"""# *Predict X_test*"""

Y_pred_test = model.predict(X_test)

# Commented out IPython magic to ensure Python compatibility.
print('Coefficients:', model.coef_)
print('Intercept:', model.intercept_)
print('Mean squared error (MSE): %.2f'
#       % mean_squared_error(Y_test, Y_pred_test))
print('Coefficient of determination (R^2): %.2f'
#       % r2_score(Y_test, Y_pred_test))

"""# Now let's use entire dataset for model training (For Comparison)"""

full = linear_model.LinearRegression()
full.fit(X, Y)

full_pred = model.predict(X)

# Commented out IPython magic to ensure Python compatibility.
print('Coefficients:', full.coef_)
print('Intercept:', full.intercept_)
print('Mean squared error (MSE): %.2f'
#       % mean_squared_error(Y, full_pred))
print('Coefficient of determination (R^2): %.2f'
#       % r2_score(Y, full_pred))

"""# Scatter plot of experimental vs. predicted LogS"""

import matplotlib.pyplot as plt

"""# *Quick check of the variable dimensions of Train and Test sets*"""

Y_train.shape, Y_pred_train.shape

Y_test.shape, Y_pred_test.shape

"""# Vertical plot"""

plt.figure(figsize=(5,11))

# 2 row, 1 column, plot 1
plt.subplot(2, 1, 1)
plt.scatter(x=Y_train, y=Y_pred_train, c="#7CAE00", alpha=0.3)

z = np.polyfit(Y_train, Y_pred_train, 1)
p = np.poly1d(z)
plt.plot(Y_test,p(Y_test),"#F8766D")

plt.ylabel('Predicted LogS')


# 2 row, 1 column, plot 2
plt.subplot(2, 1, 2)
plt.scatter(x=Y_test, y=Y_pred_test, c="#619CFF", alpha=0.3)

z = np.polyfit(Y_test, Y_pred_test, 1)
p = np.poly1d(z)
plt.plot(Y_test,p(Y_test),"#F8766D")

plt.ylabel('Predicted LogS')
plt.xlabel('Experimental LogS')

plt.savefig('plot_vertical_logS.png')
plt.savefig('plot_vertical_logS.pdf')
plt.show()

"""# Horizontal plot"""

plt.figure(figsize=(11,5))

# 1 row, 2 column, plot 1
plt.subplot(1, 2, 1)
plt.scatter(x=Y_train, y=Y_pred_train, c="#7CAE00", alpha=0.3)

z = np.polyfit(Y_train, Y_pred_train, 1)
p = np.poly1d(z)
plt.plot(Y_test,p(Y_test),"#F8766D")

plt.ylabel('Predicted LogS')
plt.xlabel('Experimental LogS')

# 1 row, 2 column, plot 2
plt.subplot(1, 2, 2)
plt.scatter(x=Y_test, y=Y_pred_test, c="#619CFF", alpha=0.3)

z = np.polyfit(Y_test, Y_pred_test, 1)
p = np.poly1d(z)
plt.plot(Y_test,p(Y_test),"#F8766D")

plt.xlabel('Experimental LogS')

plt.savefig('plot_horizontal_logS.png')
plt.savefig('plot_horizontal_logS.pdf')
plt.show()