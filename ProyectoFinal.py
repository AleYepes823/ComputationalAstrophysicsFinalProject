import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn

Nombres = {0:'Brown Dwarf',1:'Red Dwarf',2:'White Dwarf',3:'Main Sequence',4:'Supergiant',5:'Hypergiant'}

df_stars = pd.read_csv('6 class csv.csv')

# #TODO Heatmap

# sns.heatmap(df_stars.corr(), xticklabels=df_stars.corr().columns, yticklabels=df_stars.corr().columns,annot = True)
# plt.xticks(fontsize = 'xx-small');plt.yticks(fontsize = 'xx-small')
# plt.savefig('Images/HeatMapCorrelation.png',pad_inhes = 1.5)
# plt.close()

# #TODO Hertzprung Russell Diagram

# fig, ax = plt.subplots(1)
# GroupedData = df_stars.groupby('Star type')
# for GroupName,group in GroupedData:
#     print(type(GroupName))
#     ax.scatter(group['Temperature (K)'], group['Luminosity(L/Lo)'],label = Nombres[GroupName])
# ax.legend()
# ax.set_xlabel("Temperature (K)")
# ax.set_ylabel("Luminosity (L/Lo)")
# ax.set_ylim(df_stars['Luminosity(L/Lo)'].min(),df_stars['Luminosity(L/Lo)'].max())
# plt.xscale('log')
# plt.yscale('log')
# ax.invert_xaxis()
# plt.savefig('Images/HRDiagram.png')
# plt.close()

# #TODO Dimension Reduction

from sklearn.decomposition import PCA
pca = PCA(n_components=2)

df_stars_data = df_stars.drop(columns=["Star color", "Star type", "Spectral Class"])
# X = np.array(df_stars_data)
# #print(df_stars_data)


# #! SIN  NORMALIZAR
# pca = PCA()
# pca.fit(X)
# pca_data = pca.transform(X)

# per_var = np.round(pca.explained_variance_ratio_*100, decimals = 3)
# labels = ['PC' + str(x) for x in range(1,len(per_var)+1)]

# print(per_var)
# print(labels)

# pca_df = pd.DataFrame(pca.transform(X), columns=labels)
# pca_df['Star type'] = df_stars['Star type']

# fig, ax = plt.subplots(1,figsize=(10, 10))
# Grouped = pca_df.groupby('Star type')
# for name, group in Grouped:
#     ax.scatter(group.PC1, group.PC2,label = Nombres[name])
# plt.legend()
# plt.title("Star observations projected on the first 2 components after PCA")
# plt.xlabel('PC1 - {0}%'.format(per_var[0]))
# plt.ylabel('PC2 - {0}%'.format(per_var[1]))

# #for sample in pca_df.index:
# #    plt.annotate(sample, (pca_df.PC1.loc[sample], pca_df.PC2.loc[sample]))
# plt.savefig('Images/Correlation.png')
# plt.draw()
# plt.pause(5)
# plt.close()

# #! NORMALIZADO

from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler

#scaled_data = scale(df_stars_data)

scaler = StandardScaler()
scaler.fit(df_stars_data)

pca = PCA()
pca.fit(scaler.transform(df_stars_data))
pca_data = pca.transform(scaler.transform(df_stars_data))

per_var = np.round(pca.explained_variance_ratio_*100, decimals = 1)
labels = ['PC' + str(x) for x in range(1,len(per_var)+1)]

# print(per_var)
# print(labels)

pca_df = pd.DataFrame(pca_data, columns=labels)
pca_df['Star type'] = df_stars['Star type']

# fig, ax = plt.subplots(1,figsize=(10, 10))
# Grouped = pca_df.groupby('Star type')
# for name, group in Grouped:
#     ax.scatter(group.PC1, group.PC2,label = Nombres[name])
# plt.legend()
# plt.title("Star observations projected on the first 2 components after normalized PCA")
# plt.xlabel('PC1 - {0}%'.format(per_var[0]))
# plt.ylabel('PC2 - {0}%'.format(per_var[1]))

# #for sample in pca_df.index:
# #    plt.annotate(sample, (pca_df.PC1.loc[sample], pca_df.PC2.loc[sample]))
# plt.savefig('Images/NormalizedCorrelation.png')
# plt.draw()
# plt.pause(5)
# plt.close()

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
# Sizes = 0.3 +0.05*np.arange(9)

# for size in Sizes:
#     TrainFeature, TestFeature, TrainTarget, TestTarget = train_test_split(pca_df[['PC1','PC2']],pca_df['Star type'],random_state = 60, test_size = size)
#     print(size,len(TrainFeature))
#     fig, axes = plt.subplots(3,sharex=True)
#     y1 = [];y2 = [];y3 = []; xs = []
#     for i in range(2,51):
#         tree = DecisionTreeClassifier(max_depth = i)
#         tree.fit(TrainFeature,TrainTarget)
#         prediction = tree.predict(TestFeature)
#         Discrepancie = np.abs(prediction-TestTarget)
#         N = len(Discrepancie)
#         Discrepancie = Discrepancie.where(Discrepancie != 0).dropna()
#         n = len(Discrepancie)
#         y1.append(round(n*100/N,1))
#         y2.append(Discrepancie.mean())
#         y3.append(tree.score(TestFeature,TestTarget))
#         xs.append(i)
#     axes[0].plot(xs,y1,'b-')
#     axes[0].set_ylabel('Number of Errors Percentage')
#     axes[1].plot(xs,y2,'r-')
#     axes[1].set_ylabel('Mean Error Discrepancie')
#     axes[2].plot(xs,y3,'g-')
#     axes[2].set_ylabel('Score')
#     axes[2].set_xlabel('Decission Tree Max Depth')
#     fig.suptitle(f'Test size {size:.2f}')
#     plt.savefig(f'Images/TestSize{size:.2f}.png',bbox_inches = 'tight')
#     plt.close()

TrainFeature, TestFeature, TrainTarget, TestTarget = train_test_split(pca_df[['PC1','PC2']],pca_df['Star type'],random_state = 60, test_size = 0.7)
tree = DecisionTreeClassifier(max_depth = 8)
tree.fit(TrainFeature,TrainTarget)
print('Hola')
prediction = tree.predict(TestFeature)
print(prediction)
cm = confusion_matrix(np.asarray(TestTarget),prediction)
sns.heatmap(cm,annot = True)
plt.show()

fig = plt.figure(figsize=(12, 12))
sklearn.tree.plot_tree(tree, filled=True)
plt.show()