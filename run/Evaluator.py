import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.manifold import TSNE
# import matplotlib.pyplot as plt
import warnings
from torch.utils.data import Subset
from log_init import _init_log
import logging
# _init_log('./', 'politifact', 'MVGRL')
def draw_plot(datadir, DS, embeddings, fname, max_nodes=None):
    return
    graphs = read_graphfile(datadir, DS, max_nodes=max_nodes)
    labels = [graph.graph['label'] for graph in graphs]

    labels = preprocessing.LabelEncoder().fit_transform(labels)
    x, y = np.array(embeddings), np.array(labels)
    print('fitting TSNE ...')
    x = TSNE(n_components=2).fit_transform(x)

    plt.close()
    df = pd.DataFrame(columns=['x0', 'x1', 'Y'])

    df['x0'], df['x1'], df['Y'] = x[:,0], x[:,1], y
    sns.pairplot(x_vars=['x0'], y_vars=['x1'], data=df, hue="Y", size=5)
    plt.legend()
    plt.savefig(fname)


def Logistic_classify(x, y, K, seed, data_reverse = True):

    acc_list, recall_list, prec_list, f1ma_list, f1mi_list = [], [], [], [], []
    kf = StratifiedKFold(n_splits=K, shuffle=True, random_state=seed)
    ctr = 0
    for train_index, test_index in kf.split(x, y):
        if data_reverse == True:
            train_index, test_index = test_index, train_index
            # test
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        classifier = LogisticRegression(max_iter=1000)
        classifier.fit(x_train, y_train)
        y_pred = classifier.predict(x_test)
        acc_list.append(accuracy_score(y_test, y_pred))
        recall_list.append(recall_score(y_test, y_pred))
        prec_list.append(precision_score(y_test, y_pred))
        f1ma_list.append(f1_score(y_test, y_pred, average='macro'))
        f1mi_list.append(f1_score(y_test, y_pred, average='micro'))
        ctr += 1
        if ctr == 20:
            break
    print(K, 'Kold ','LogisticRegression----Acc:', np.mean(acc_list), '|Recall:', np.mean(recall_list), '|Prec', np.mean(prec_list),
          '|F1_ma', np.mean(f1ma_list))
    # logging.info('LogisticRegression----Acc:', np.mean(acc_list), '|Recall:', np.mean(recall_list), '|Prec', np.mean(prec_list),
    #       '|F1_ma', np.mean(f1ma_list))


def svc_classify(x, y, K, seed, search=True, data_reverse = True):
    kf = StratifiedKFold(n_splits=K, shuffle=True, random_state=seed)
    acc_list, recall_list, prec_list, f1ma_list, f1mi_list = [], [], [], [], []
    ctr = 0
    for train_index, test_index in kf.split(x, y):
        if data_reverse == True:
            train_index, test_index = test_index, train_index

        # test
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1)
        if search:
            params = {'C':[0.001, 0.01,0.1,1,10,100,1000]}
            classifier = GridSearchCV(SVC(), params, cv=5, scoring='accuracy', verbose=0)
        else:
            classifier = SVC(C=10)
        classifier.fit(x_train, y_train)
        y_pred = classifier.predict(x_test)
        acc_list.append(accuracy_score(y_test, y_pred))
        recall_list.append(recall_score(y_test, y_pred))
        prec_list.append(precision_score(y_test, y_pred))
        f1ma_list.append(f1_score(y_test, y_pred, average='macro'))
        f1mi_list.append(f1_score(y_test, y_pred, average='micro'))
        ctr += 1
        if ctr == 20:
            break
    print(K, 'Kold ','SVC----Acc:', np.mean(acc_list), '|Recall:', np.mean(recall_list), '|Prec', np.mean(prec_list),
          '|F1_ma', np.mean(f1ma_list))

def randomforest_classify(x, y, K, seed, search=True, data_reverse = True):
    kf = StratifiedKFold(n_splits=K, shuffle=True, random_state=seed)
    acc_list, recall_list, prec_list, f1ma_list, f1mi_list = [], [], [], [], []
    ctr = 0
    for train_index, test_index in kf.split(x, y):
        if data_reverse == True:
            train_index, test_index = test_index, train_index
        # test
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        if search:
            params = {'n_estimators': [100, 200, 500, 1000]}
            classifier = GridSearchCV(RandomForestClassifier(), params, cv=5, scoring='accuracy', verbose=0)
        else:
            classifier = RandomForestClassifier()
        classifier.fit(x_train, y_train)
        y_pred = classifier.predict(x_test)
        acc_list.append(accuracy_score(y_test, y_pred))
        recall_list.append(recall_score(y_test, y_pred))
        prec_list.append(precision_score(y_test, y_pred))
        f1ma_list.append(f1_score(y_test, y_pred, average='macro'))
        f1mi_list.append(f1_score(y_test, y_pred, average='micro'))
        ctr += 1
        if ctr == 20:
            break
    print(K, 'Kold ','RandomForest----Acc:', np.mean(acc_list), '|Recall:', np.mean(recall_list), '|Prec', np.mean(prec_list),
          '|F1_ma', np.mean(f1ma_list))


def linearsvc_classify(x, y, K, seed, search=True, data_reverse = True):
    kf = StratifiedKFold(n_splits=K, shuffle=True, random_state=seed)
    acc_list, recall_list, prec_list, f1ma_list, f1mi_list = [], [], [], [], []
    ctr = 0
    for train_index, test_index in kf.split(x, y):
        if data_reverse == True:
            train_index, test_index = test_index, train_index
        # test
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        if search:
            params = {'C':[0.001, 0.01,0.1,1,10,100,1000]}
            classifier = GridSearchCV(LinearSVC(), params, cv=5, scoring='accuracy', verbose=0)
        else:
            classifier = LinearSVC(C=10)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            classifier.fit(x_train, y_train)
        y_pred = classifier.predict(x_test)
        acc_list.append(accuracy_score(y_test, y_pred))
        recall_list.append(recall_score(y_test, y_pred))
        prec_list.append(precision_score(y_test, y_pred))
        f1ma_list.append(f1_score(y_test, y_pred, average='macro'))
        f1mi_list.append(f1_score(y_test, y_pred, average='micro'))
        ctr += 1
        if ctr==20:
            break
    print(K, 'Kold ', 'LinearSVM----Acc:', np.mean(acc_list), '|Recall:', np.mean(recall_list), '|Prec', np.mean(prec_list),
          '|F1_ma', np.mean(f1ma_list))


def MLP_classify(x, y, K, seed, data_reverse = True):
    kf = StratifiedKFold(n_splits=K, shuffle=True, random_state=seed)
    acc_list, recall_list, prec_list, f1ma_list, f1mi_list = [], [], [], [], []
    ctr = 0
    for train_index, test_index in kf.split(x, y):
        if data_reverse == True:
            train_index, test_index = test_index, train_index
        # test
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        classifier = MLPClassifier()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            classifier.fit(x_train, y_train)
        y_pred = classifier.predict(x_test)
        acc_list.append(accuracy_score(y_test, y_pred))
        recall_list.append(recall_score(y_test, y_pred))
        prec_list.append(precision_score(y_test, y_pred))
        f1ma_list.append(f1_score(y_test, y_pred, average='macro'))
        f1mi_list.append(f1_score(y_test, y_pred, average='micro'))
        ctr += 1
        if ctr == 20:
            break
    print(K, 'Kold ','MLP----Acc:', np.mean(acc_list), '|Recall:', np.mean(recall_list), '|Prec', np.mean(prec_list),
          '|F1_ma', np.mean(f1ma_list))

def linearsvc_classify_valid(x, y, search):
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)
    accuracies = []
    accuracies_val = []
    for train_index, test_index in kf.split(x, y):

        # test
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        if search:
            params = {'C':[0.001, 0.01,0.1,1,10,100,1000]}
            classifier = GridSearchCV(LinearSVC(), params, cv=5, scoring='accuracy', verbose=0)
        else:
            classifier = LinearSVC(C=10)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            classifier.fit(x_train, y_train)
        accuracies.append(accuracy_score(y_test, classifier.predict(x_test)))

        # val
        val_size = len(test_index)
        test_index = np.random.choice(train_index, val_size, replace=False).tolist()
        train_index = [i for i in train_index if not i in test_index]

        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        if search:
            params = {'C':[0.001, 0.01,0.1,1,10,100,1000]}
            classifier = GridSearchCV(LinearSVC(), params, cv=5, scoring='accuracy', verbose=0)
        else:
            classifier = LinearSVC(C=10)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            classifier.fit(x_train, y_train)
        accuracies_val.append(accuracy_score(y_test, classifier.predict(x_test)))

    return np.mean(accuracies_val), np.mean(accuracies)

def logistic_classify_valid(x, y):

    nb_classes = np.unique(y).shape[0]
    xent = nn.CrossEntropyLoss()
    hid_units = x.shape[1]

    accs = []
    accs_val = []
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)
    for train_index, test_index in kf.split(x, y):

        # test
        train_embs, test_embs = x[train_index], x[test_index]
        train_lbls, test_lbls= y[train_index], y[test_index]

        train_embs, train_lbls = torch.from_numpy(train_embs).cuda(), torch.from_numpy(train_lbls).cuda()
        test_embs, test_lbls= torch.from_numpy(test_embs).cuda(), torch.from_numpy(test_lbls).cuda()


        log = LogReg(hid_units, nb_classes)
        log.cuda()
        opt = torch.optim.Adam(log.parameters(), lr=0.01, weight_decay=0.0)

        best_val = 0
        test_acc = None
        for it in range(100):
            log.train()
            opt.zero_grad()

            logits = log(train_embs)
            loss = xent(logits, train_lbls)

            loss.backward()
            opt.step()

        logits = log(test_embs)
        preds = torch.argmax(logits, dim=1)
        acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
        accs.append(acc.item())

        # val
        val_size = len(test_index)
        test_index = np.random.choice(test_index, val_size, replace=False).tolist()
        train_index = [i for i in train_index if not i in test_index]

        train_embs, test_embs = x[train_index], x[test_index]
        train_lbls, test_lbls= y[train_index], y[test_index]

        train_embs, train_lbls = torch.from_numpy(train_embs).cuda(), torch.from_numpy(train_lbls).cuda()
        test_embs, test_lbls= torch.from_numpy(test_embs).cuda(), torch.from_numpy(test_lbls).cuda()


        log = LogReg(hid_units, nb_classes)
        log.cuda()
        opt = torch.optim.Adam(log.parameters(), lr=0.01, weight_decay=0.0)

        best_val = 0
        test_acc = None
        for it in range(100):
            log.train()
            opt.zero_grad()

            logits = log(train_embs)
            loss = xent(logits, train_lbls)

            loss.backward()
            opt.step()

        logits = log(test_embs)
        preds = torch.argmax(logits, dim=1)
        acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
        accs_val.append(acc.item())

    return np.mean(accs_val), np.mean(accs)

def svc_classify_valid(x, y, search):
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)
    accuracies = []
    accuracies_val = []
    for train_index, test_index in kf.split(x, y):

        # test
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1)
        if search:
            params = {'C':[0.001, 0.01,0.1,1,10,100,1000]}
            classifier = GridSearchCV(SVC(), params, cv=5, scoring='accuracy', verbose=0)
        else:
            classifier = SVC(C=10)
        classifier.fit(x_train, y_train)
        accuracies.append(accuracy_score(y_test, classifier.predict(x_test)))

        # val
        val_size = len(test_index)
        test_index = np.random.choice(train_index, val_size, replace=False).tolist()
        train_index = [i for i in train_index if not i in test_index]

        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1)
        if search:
            params = {'C':[0.001, 0.01,0.1,1,10,100,1000]}
            classifier = GridSearchCV(SVC(), params, cv=5, scoring='accuracy', verbose=0)
        else:
            classifier = SVC(C=10)
        classifier.fit(x_train, y_train)
        accuracies_val.append(accuracy_score(y_test, classifier.predict(x_test)))

    return np.mean(accuracies_val), np.mean(accuracies)

def k_fold_split(dataset, lengths, seed):
    generator = torch.Generator().manual_seed(seed)
    if sum(lengths) != len(dataset):
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")
    indices = torch.randperm(sum(lengths), generator=generator).tolist()
    data_list = []
    lens, off_lens = lengths[0], lengths[-1]
    train_set_k1, test_set_k1 = Subset(dataset, indices[:lens]), Subset(dataset, indices[lens:])
    train_set_k2, test_set_k2 = Subset(dataset, indices[lens:lens*2]), Subset(dataset, indices[0: lens]+indices[lens*2:])
    train_set_k3, test_set_k3 = Subset(dataset, indices[lens*2:lens*3]), Subset(dataset, indices[0: lens*2] + indices[lens*3:])
    train_set_k4, test_set_k4 = Subset(dataset, indices[lens*3:lens*4]), Subset(dataset, indices[0: lens*3] + indices[lens*4:])
    train_set_k5, test_set_k5 = Subset(dataset, indices[lens*4:]), Subset(dataset, indices[:lens*4])
    data_list.append((train_set_k1, test_set_k1))
    data_list.append((train_set_k2, test_set_k2))
    data_list.append((train_set_k3, test_set_k3))
    data_list.append((train_set_k4, test_set_k4))
    data_list.append((train_set_k5, test_set_k5))
    return data_list

def get_metrics(y_pred, y_test):
    acc = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    pre = precision_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_micro = f1_score(y_test, y_pred, average='micro')
    return (acc, f1_macro, f1_micro, pre, recall)