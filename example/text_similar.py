import pandas as pd
import jieba
from gensim import corpora, models, similarities
import numpy as np
from tqdm import tqdm
import turicreate as tc


def bow2vec(corpus_tfidf,dictionary):
    vec = []
    length = max(dictionary) + 1
    for content in tqdm(corpus_tfidf):
        sentense_vectors = np.zeros(length)
        for co in content:
            sentense_vectors[co[0]]=co[1]
        vec.append(sentense_vectors)
    return vec

def vec2SFrame(text_list):
    texts = [[word for word in jieba.cut(document, cut_all=True)] for document in text_list] # 分词
    dictionary = corpora.Dictionary(texts)                   # 词典
    corpus = [dictionary.doc2bow(text) for text in texts]    # 矢量化
    tfidf = models.TfidfModel(corpus)                        # TFIDF
    corpus_tfidf =tfidf[corpus]    
    svec = bow2vec(corpus_tfidf,dictionary)
    return tc.SFrame(pd.DataFrame(svec))

def similar(svec_tc,label = None,distance = 'cosine',topn = 10):
    '''
    输入:
        svec_tc是SFrame格式内容
    输出:
        simliar_dataframe也是SFrame格式内容
    label:如果有,那么就可以更好地定位什么序号是什么东西
    distance:选择距离公式‘euclidean’, ‘squared_euclidean’, ‘manhattan’, ‘levenshtein’(文字距离), ‘jaccard’, ‘weighted_jaccard’, cosine’
    topn:每个个案选择前10个内容
    '''
    model = tc.nearest_neighbors.create(svec_tc, features=svec_tc.column_names(),distance = distance) # 建模
    sim_graph = model.similarity_graph(k=topn)  # 挑选top 10相似的
    simliar_dataframe = sim_graph.edges.to_dataframe()  # 变成dataframe格式
    if label:
        simliar_dataframe['src_id_ch'] = [label[i] for i in simliar_dataframe.__src_id]
        simliar_dataframe['dst_id_ch'] = [label[i] for i in simliar_dataframe.__dst_id]
    simliar_dataframe['simliar'] = 1 - simliar_dataframe['distance']
    return simliar_dataframe

 if __name__ == '__main__':
    data = pd.read_csv('/mnt/GSH.csv') # 导入数据
    svec = vec2SFrame(list(data.test))  # 文本列
    similar_dataframe = similar(svec,label = data.name)
    #similar_dataframe[simliar_dataframe['simliar']>0.8]
    similar_dataframe.to_csv('/mnt/similiar.csv',encoding = 'utf-8',index = False)
