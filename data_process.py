import numpy as np
from sklearn.semi_supervised import LabelPropagation
dict = {'WWW': 0, 'MAIL': 1, 'FTP-CONTROL':2,'FTP-PASV':3,'ATTACK':4, 'P2P':5,'DATABASE':6,'FTP-DATA':7,'MULTIMEDIA':8,'SERVICES':9,'INTERACTIVE':10,'GAMES':11}
unique_label =['WWW', 'MAIL', 'FTP-CONTROL', 'FTP-PASV', 'ATTACK', 'P2P', 'DATABASE', 'FTP-DATA', 'MULTIMEDIA', 'SERVICES', 'INTERACTIVE', 'GAMES']
label_prop_model = LabelPropagation()

def data_create(url, filename):
    matrix1 = np.loadtxt(open(url, "rb"), dtype="str", delimiter=",", skiprows=253)
    colomn1 = np.array(matrix1[:, 159], dtype="float")
    colomn1 = colomn1.reshape(-1, 1)
    colomn2 = np.array(matrix1[:, 164], dtype="float")
    colomn2 = colomn2.reshape(-1, 1)
    colomn3 = np.array(matrix1[:, 165], dtype="float")
    colomn3 = colomn3.reshape(-1, 1)
    colomn4 = np.array(matrix1[:, 180], dtype="float")
    colomn4 = colomn4.reshape(-1, 1)
    colomn5 = np.array(matrix1[:, 185], dtype="float")
    colomn5 = colomn5.reshape(-1, 1)
    colomn6= np.array(matrix1[:, 186], dtype="float")
    colomn6 = colomn6.reshape(-1, 1)
    colomn7 = np.array(matrix1[:, 194], dtype="float")
    colomn7 = colomn7.reshape(-1, 1)
    colomn8 = np.array(matrix1[:, 197], dtype="float")
    colomn8 = colomn8.reshape(-1, 1)
    colomn9 = np.array(matrix1[:, 199], dtype="float")
    colomn9 = colomn9.reshape(-1, 1)
    colomn10 = np.array(matrix1[:, 200], dtype="float")
    colomn10 = colomn10.reshape(-1, 1)
    colomn11 = np.array(matrix1[:, 204], dtype="float")
    colomn11 = colomn11.reshape(-1, 1)
    colomn12 = np.array(matrix1[:, 206], dtype="float")
    colomn12 = colomn12.reshape(-1, 1)
    colomn13 = np.array(matrix1[:, 207], dtype="float")
    colomn13 = colomn13.reshape(-1, 1)
    colomn14= np.array(matrix1[:, 211], dtype="float")
    colomn14 = colomn14.reshape(-1, 1)
    colomn15 = np.array(matrix1[:, 30], dtype="float")
    colomn15 = colomn15.reshape(-1, 1)
    colomn16 = np.array(matrix1[:, 31], dtype="float")
    colomn16 = colomn16.reshape(-1, 1)
    data = np.concatenate((colomn1, colomn2,colomn3, colomn4,colomn5, colomn6,colomn7, colomn8,colomn9, colomn10, colomn11, colomn12,colomn13, colomn14,colomn15, colomn16), axis=1)
    print(type(data))
    np.save(filename, data)

#no use
def label_create(url):
    matrix1 = np.loadtxt(open(url, "rb"), dtype="str", delimiter=",", skiprows=253)
    label = np.array(matrix1[:, 248], dtype="str")
    np.save("train_label",label)

def int_label_create(url, filename) :
    my_matrix = np.loadtxt(open(url, "rb"), dtype="str", delimiter=",", skiprows=253)
    label = np.array(my_matrix[:,248])
    int_label = np.zeros(len(label), dtype= 'int')
    count = 0
    for label_item in unique_label:
        equal_to_label = (label == label_item)
        int_label[equal_to_label] = count
        count = count + 1
    np.save(filename,int_label)

data_create("/Users/xumengyi/Downloads/Moore/entry01.weka.allclass.arff","entry01_data")
int_label_create("/Users/xumengyi/Downloads/Moore/entry01.weka.allclass.arff", "entry01_label")

print(np.load("entry01_label.npy"))
print(np.load("entry01_data.npy"))

'''
train_data = data_create("/Users/xumengyi/Downloads/Moore/entry01.weka.allclass.arff")
train_label = int_label_create("/Users/xumengyi/Downloads/Moore/entry01.weka.allclass.arff")
label_prop_model.fit(train_data, train_label)
test_data = data_create("/Users/xumengyi/Downloads/Moore/entry02.weka.allclass.arff ")
test_label = int_label_create("/Users/xumengyi/Downloads/Moore/entry02.weka.allclass.arff")
print(label_prop_model.score(test_data,test_label))
'''
#python对文本进行操作，生成可执行文件



#测试将numpy数据保存到文件
'''
data_create("/Users/xumengyi/Desktop/test.arff")
train_data = np.load("train_data.npy")
print(type(train_data))
print(train_data)
'''