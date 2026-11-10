import pandas as pd
import seaborn as sb

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.cluster import k_means, OPTICS
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
df = pd.read_excel('Dry_Bean_Dataset.xlsx')

# Train the dataset on the data an output the groups
x = df.iloc[:,:-1]

km = k_means(x, n_clusters=7, random_state=41)
km[-2] 

results = pd.DataFrame(
    data=[
        df.iloc[:,0],
        km[-2]
    ],
    index=["Sample_id","Group"],
).transpose()
