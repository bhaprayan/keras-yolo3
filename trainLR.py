import pandas as pd
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
import sklearn.metrics as metrics
import matplotlib.pyplot as plt

blabels = pd.read_csv('bad_subtask_losses.csv')
glabels = pd.read_csv('good_subtask_losses.csv')

blabels['label'] = 0
glabels['label'] = 1

train_file = pd.concat((glabels,blabels))

train_file['xy_loss'] = train_file['x_loss'] + train_file['y_loss']
train_file['wh_loss'] = train_file['w_loss'] + train_file['h_loss']

X = np.empty(shape=(train_file['xy_loss'].shape[0],1))

X[:,0] = train_file['xy_loss']
# X[:,1] = train_file['wh_loss']

y = train_file['label']

clf = LogisticRegression(random_state=0, solver='lbfgs')

clf.fit(X,y)

probs = clf.predict_proba(X)
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(y, preds)
roc_auc = metrics.auc(fpr, tpr)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.savefig('roc.png')
