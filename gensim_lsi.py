from gensim import corpora,similarities,models
from gensim.models import Word2Vec

# 1. 先将whole_corpus中每一句sen按照空格切分成一个list放入 corpora_documents

def prepare_corpus(whole_corpus):
	corpora_documents = []
	for sen in whole_corpus:
		corpora_documents.append(sen.split())
	return corpora_documents

def gen_dict(corpora_documents):
	dictionary = corpora.Dictionary(corpora_documents)
	dictionary.save('dict.txt') #保存生成的词典
	return dictionary

def gen_corpus(dictionary,corpora_documents):
	corpus = [dictionary.doc2bow(text) for text in corpora_documents]
	corpora.MmCorpus.serialize('corpuse.mm', corpus)#保存生成的语料
	# corpus = corpora.MmCorpus('corpuse.mm')  # 加载保存的语料
	return corpus

def gen_tfidf(corpus):
	tfidf_model = models.TfidfModel(corpus)
	corpus_tfidf = tfidf_model[corpus]
	return corpus_tfidf

def gen_lsi(corpus_tfidf):
	lsi = models.LsiModel(corpus_tfidf,num_topics=5)
	corpus_lsi = lsi[corpus_tfidf]  
	lsi.save('/tmp/model.lsi') # same for tfidf, lda, 
	lsi = models.LsiModel.load('/tmp/model.lsi')

	return corpus_lsi

	# print (lsi)
	# LsiModel(num_terms=1945, num_topics=10, decay=1.0, chunksize=20000)
	# print (corpus_lsi[0])
	# [(0, -0.19591060740046518), (1, -0.45795954501435981), (2, -0.1791969415198883), (3, -0.084827324470280421), (4, -0.10994447686201088), (5, -0.099089054366497328)]

def example():
	test_data_3 = '长沙街头发生砍人事件致6人死亡'
	test_cut_raw_3 = list(jieba.cut(test_data_3))# 1.分词 
	test_corpus_3 = dictionary.doc2bow(test_cut_raw_3)  # 2.转换成bow向量
	test_corpus_tfidf_3 = tfidf_model[test_corpus_3]  # 3.计算tfidf值
	test_corpus_lsi_3 = lsi[test_corpus_tfidf_3]  # 4.计算lsi值

def get_lsi_vec():
	whole_sen_1 = corpus_lsi[:2250]
	whole_sen_2 = corpus_lsi[2250:]

	value = []
	for i in range(0,len(whole_sen_1)):
		#print (pairwise_distances([sen1[i],sen2[i]],metric='cosine'))
		#data[metric_str][i] = 1 - pairwise_distances([sen1[i],sen2[i]],metric=metric_str)[0][1]
		sen1 = [value for (topic,value) in whole_sen_1[i]]
		sen2 = [value for (topic,value) in whole_sen_2[i]]
		value.append(pairwise_distances([sen1,sen2],metric=metric_str)[0][1])
		#print (pairwise_distances([sen1[i],sen2[i]],metric=metric_str)[0][1])
		#print (data[metric_str][i])
	data[metric_str+'lsi'] = pd.Series(value)