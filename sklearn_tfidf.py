from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.metrics.pairwise import pairwise_distances


vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 5000)

transformer = TfidfTransformer()
whole = vectorizer.fit_transform(whole_corpus)
tf_idf = transformer.fit_transform(whole)

whole = whole.toarray()
tf_idf = tf_idf.toarray()

sen1 = tf_idf[:2250]
sen2 = tf_idf[2250:]

data = pd.DataFrame(pd.concat([train['id'],test['id']],axis=0,ignore_index=True))

'''
计算向量相似度
'''
for metric_str in ['cityblock','cosine','euclidean','l1','l2','manhattan']:
    value = []
    for i in range(0,sen1.shape[0]):
        #print (pairwise_distances([sen1[i],sen2[i]],metric='cosine'))
        #data[metric_str][i] = 1 - pairwise_distances([sen1[i],sen2[i]],metric=metric_str)[0][1]
        value.append(pairwise_distances([sen1[i],sen2[i]],metric=metric_str)[0][1])
        #print (pairwise_distances([sen1[i],sen2[i]],metric=metric_str)[0][1])
        #print (data[metric_str][i])
    data[metric_str+'tf_idf'] = pd.Series(value)
    
data.to_csv('add_simi.csv',index=False)