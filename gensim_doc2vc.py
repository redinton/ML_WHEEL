'''
Author: Jason
Time: 2018.2.20 17:05
'''

import gensim
from gensim.models.doc2vec import Doc2Vec,LabeledSentence

LabeledSentence = gensim.models.doc2vec.LabeledSentence


def prepare_data(corpus):
    data_pre = [z.split() for z in corpus]
    return data_pre
'''
    gensim的Doc2Vec应用于训练要求每一篇文章/句子有一个唯一标识的label.
    我们使用Gensim自带的LabeledSentence方法. 标识的格式为"TRAIN_i"和"TEST_i"，其中i为序号
'''
def labelizeReviews(reviews, label_type):
    labelized = []
    for i,v in enumerate(reviews):
        label = '%s_%s'%(label_type,i)
        labelized.append(LabeledSentence(v, [label]))
    return labelized
 
##读取向量
def getVecs(model, corpus, size):
    vecs = [np.array(model.docvecs[z.tags[0]]).reshape((1, size)) for z in corpus]
    return np.concatenate(vecs)

##对数据进行训练
def train(x_train,x_test,size = 400,epoch_num=10):
    #实例DM和DBOW模型
    model_dm = gensim.models.Doc2Vec(min_count=1, window=10, size=size, sample=1e-3, negative=5, workers=3)
    model_dbow = gensim.models.Doc2Vec(min_count=1, window=10, size=size, sample=1e-3, negative=5, dm=0, workers=3)

    #使用所有的数据建立词典
    model_dm.build_vocab(x_train+x_test)
    model_dbow.build_vocab(x_train+x_test)
    
    model_dm.train(x_train+x_test,total_examples=model_dm.corpus_count,epochs=model_dm.iter)
    model_dbow.train(x_train+x_test,total_examples=model_dbow.corpus_count,epochs=model_dbow.iter)
    
    '''
    #进行多次重复训练，每一次都需要对训练数据重新打乱，以提高精度
    all_train_reviews = np.array(x_train)
    for epoch in range(epoch_num):
        perm = np.random.permutation(all_train_reviews.shape[0])
        model_dm.train(all_train_reviews[perm],)
        model_dbow.train(all_train_reviews[perm])
    
    #训练测试数据集
    x_test = np.array(x_test)
    for epoch in range(epoch_num):
        perm = np.random.permutation(x_test.shape[0])
        model_dm.train(x_test[perm])
        model_dbow.train(x_test[perm])
    '''

    return model_dm,model_dbow

##将训练完成的数据转换为vectors
def get_vectors(model_dm,model_dbow):

    #获取训练数据集的文档向量
    train_vecs_dm = getVecs(model_dm, x_train, size)
    train_vecs_dbow = getVecs(model_dbow, x_train, size)
    train_vecs = np.hstack((train_vecs_dm, train_vecs_dbow))
    #获取测试数据集的文档向量
    test_vecs_dm = getVecs(model_dm, x_test, size)
    test_vecs_dbow = getVecs(model_dbow, x_test, size)
    test_vecs = np.hstack((test_vecs_dm, test_vecs_dbow))

    return train_vecs,test_vecs



# 1. 将清洗后的预料处理成放入gensim中的格式
x_train = prepare_data(clean_train)
x_test = prepare_data(clean_test)
print (x_train[0])
# ['two', 'big', 'brown', 'dog', 'run', 'through', 'the', 'snow']

# 2. 给训练数据和测试数据打上标记 - doc2vc必须步骤
x_train = labelizeReviews(x_train, 'TRAIN')
x_test = labelizeReviews(x_test, 'TEST')

print (type(x_train[0]))
print (type(x_train))

# 3. 训练得到训练数据和测试数据每一份doc对应的vec

#设置向量维度和训练次数
size,epoch_num=100,10
#对数据进行训练，获得模型
model_dm,model_dbow = train(x_train,x_test,size,epoch_num)
#从模型中抽取文档相应的向量
train_vecs,test_vecs = get_vectors(model_dm,model_dbow)