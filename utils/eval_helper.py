from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, roc_auc_score, average_precision_score
import torch
import random
"""
	Utility functions for evaluating the model performance
"""


def eval_deep(log, loader):
	"""
	Evaluating the classification performance given mini-batch data
	"""

	# get the empirical batch_size for each mini-batch

	data_size = len(loader.dataset.indices)
	batch_size = loader.batch_size
	# print(data_size, batch_size)
	if data_size % batch_size == 0:
		size_list = [batch_size] * (data_size//batch_size)
	else:
		size_list = [batch_size] * (data_size // batch_size) + [data_size % batch_size]
	# size_list = [batch_size] * (data_size // batch_size) + [data_size % batch_size]
	# print(log, size_list)
	# print(len(log), len(size_list))
	assert len(log) == len(size_list)

	accuracy, f1_macro, f1_micro, precision, recall = 0, 0, 0, 0, 0

	prob_log, label_log = [], []

	for batch, size in zip(log, size_list):
		pred_y, y = batch[0].data.cpu().numpy().argmax(axis=1), batch[1].data.cpu().numpy().tolist()
		prob_log.extend(batch[0].data.cpu().numpy()[:, 1].tolist())
		label_log.extend(y)

		accuracy += accuracy_score(y, pred_y) * size
		f1_macro += f1_score(y, pred_y, average='macro') * size
		f1_micro += f1_score(y, pred_y, average='micro') * size
		precision += precision_score(y, pred_y, zero_division=0) * size
		recall += recall_score(y, pred_y, zero_division=0) * size

	auc = roc_auc_score(label_log, prob_log)
	ap = average_precision_score(label_log, prob_log)

	return accuracy/data_size, f1_macro/data_size, f1_micro/data_size, precision/data_size, recall/data_size, auc, ap

def acc_f1(output, labels, average='binary'):
	preds = output.max(1)[1].type_as(labels)
	if preds.is_cuda:
		preds = preds.cpu()
		labels = labels.cpu()
	accuracy = accuracy_score(preds, labels)
	f1 = f1_score(preds, labels, average=average)
	return accuracy, f1

def metrics(output, labels, average='binary'):
	preds = output.max(1)[1].type_as(labels)
	if preds.is_cuda:
		preds = preds.cpu()
		labels = labels.cpu()
	accuracy = accuracy_score(preds, labels)
	f1 = f1_score(preds, labels, average=average)
	f1_micro = f1_score(preds, labels, average='micro')
	precision = precision_score(preds, labels)
	recall = recall_score(preds, labels)
	# return accuracy, f1
	return accuracy, f1, f1_micro, precision, recall

# def few_shot_split(dataset, train_shotnum, val_shotnum, classnum, tasknum):
#     #task中的train，val，test无交叉；但不同task之间的train，val，test可以交叉
#     #train_shotnum代表train中的shotnum，val_shotnum同理
#     #classnum代表总共有几类
#     #先将dataset中的数据按照class排序，然后随机从中选出数据来放入train val，剩下的放进test即可
#     train=[]
#     val=[]
#     test=[]
#     length=len(dataset)
#     #统计每类各有多少张图
#     classcount=torch.zeros(classnum)
#     #统计每类的第一个元素在dataset中的索引位置
#     class_start_index=torch.zeros(classnum)
#     label_before=1e6
#     count=0
#     for data in dataset:
#         classcount[data.y]+=1
#         if label_before != data.y:
#             label_before=data.y
#             class_start_index[data.y]=count
#         count+=1
#     #print('classcount:',classcount)
#     #print(class_start_index)
#     class_start_index=class_start_index.int()
#     for task in range(tasknum):
#         train_index=[]
#         val_index=[]
#         test_index=list(range(0, length))
#         for c in range(classnum):
#             if c!=classnum-1:
#                 index=random.sample(range(class_start_index[c], class_start_index[c+1]-1), train_shotnum+val_shotnum)
#             else:
#                 #从每类中的元素中选出train_shotnum+val_shotnum的元素，再将这些元素按照所需数目分别加入train和val中
#                 index=random.sample(range(class_start_index[c], length-1), train_shotnum+val_shotnum)
#             train_index=train_index+index[0:train_shotnum]
#             val_index=val_index+index[train_shotnum:len(index)]
#         train.append([dataset[i]for i in train_index])
#         val.append([dataset[i]for i in val_index])
#         #剩下的元素全部加入test
#         train_val_index=train_index+val_index
#         train_val_index.sort(reverse=True)
#         for i in train_val_index:
#             test_index.pop(i)
#         test.append([dataset[i]for i in test_index])
#     return train, val, test

def few_shot_split(dataset, train_shotnum, classnum):
	train, test = [], []
	length = len(dataset)
	classcount = torch.zeros(classnum)
	class_start_index = torch.zeros(classnum)
	label_before = 1e6
	count = 0
	for data in dataset:
		classcount[data.y] += 1
		if label_before != data.y:
			label_before = data.y
			class_start_index[data.y] = count
		count += 1
	class_start_index = class_start_index.int()
	train_index = []
	test_index = list(range(0, length))
	for c in range(classnum):
		if c != classnum-1:
			index = random.sample(range(class_start_index[c], class_start_index[c+1]-1), train_shotnum)
		else:
			#从每类中的元素中选出train_shotnum+val_shotnum的元素，再将这些元素按照所需数目分别加入train和val中
			index = random.sample(range(class_start_index[c], length-1), train_shotnum)
		train_index = train_index + index[0:train_shotnum]
	#剩下的元素全部加入test
	train_index.sort(reverse=True)
	for i in train_index:
		test_index.pop(i)
	return train_index, test_index
