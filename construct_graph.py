import os
import torch
import pickle
import json
import re
import pandas as pd
import logging
import math
import os
from dataclasses import dataclass, field
from typing import Optional
import torch
from tqdm import tqdm
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device=torch.device('cuda:0')
from transformers import BertModel, BertTokenizer, BertConfig

#指定文件夹路径
root_path='ZHIHU-16K/'
review_path='ZHIHU-16K/reviews'
user_path='ZHIHU-16K/users'
save_feature_path='ZHIHU-16K/feature'
save_edge_path='ZHIHU-16K/edge'
#指定模型名称
model_name='bert-base-chinese'

def save_tensor(tensor,path):
    with open(path,'wb') as f:
        pickle.dump(tensor,f)
def get_edges(data):
    edges=[]
    map_edges=[]
    id_map={}
    id_map['0']=0
    #从1开始计数id，0是文章id
    id=1
    #先统计所有存在的评论id
    for comment in data:
        #双向图
        comment_id=comment['comment_id']
        id_map[comment_id]=id
        id+=1
        for child_comment in comment['child_comment']:
            child_comment_id=child_comment['comment_id']
            id_map[child_comment_id]=id
            id+=1
    #对所有存在的id构图
    for comment in data:
        #双向图
        comment_id=comment['comment_id']
        edges.append(['0',comment_id])
        edges.append([comment_id,'0'])
        for child_comment in comment['child_comment']:
            child_comment_id=child_comment['comment_id']
            reply_id=child_comment['reply_comment_id']
            reply_root_comment_id=child_comment['reply_root_comment_id']
            #选择存在的id进行构图，因为可能存在删除评论的问题
            if reply_id in id_map.keys():
                edges.append([reply_id,child_comment_id])
                edges.append([child_comment_id,reply_id])
            elif reply_root_comment_id in id_map.keys():
                edges.append([reply_root_comment_id,child_comment_id])
                edges.append([child_comment_id,reply_root_comment_id])
            else:
                edges.append(['0',child_comment_id])
                edges.append([child_comment_id,'0'])
    #将edges中保存的id map到标准id
    for e in edges:
        map_edges.append([id_map[e[0]],id_map[e[1]]])
    return map_edges
def get_str(author_info,attr):
    if attr=='business':
        return attr+':'+author_info[attr]['name']+';'
    elif attr=='employments':
        str=''
        for l in author_info[attr]:
            if 'company' in l.keys():
                str+=l['company']['name']
                str+=','
        #去掉最后一个逗号
        if str!='':
            str=str[:-1]
        return attr+':'+str+';'
    else:
        return attr+':'+author_info[attr]+';'
def get_anonymous_str(attr):
    return attr+':'+';'
def get_token_list(str,tokenizer):
    my_List=tokenizer.encode(str)
    my_List=my_List[:512] if len(my_List)>512 else my_List
    return my_List
def get_feature(comment,tokenizer,bert):
    hot=1 if comment['hot'] else 0#是否热门
    is_author=1 if comment['is_author'] else 0# 是否为回答作者
    is_author_top=1 if comment['is_author_top'] else 0# 是否被作者置顶
    author_is_anonymous=1 if comment['author_is_anonymous'] else 0# 获取评论的作者是否匿名
    like_count=int(comment['like_count']) #获取被点赞数量
    content='content:'+comment['content'] # 获取评论的内容
    content_tensor=torch.tensor(get_token_list(content,tokenizer)).unsqueeze(0)
    content_embedding= bert(content_tensor.to(device))[1]
    #获取作者信息
    url_token=comment['url_token'] #获取作者的唯一标识
    author_exist=False
    if os.path.exists(user_path+'/'+url_token+'.json'):
        with open(user_path+'/'+url_token+'.json','r',encoding='utf-8') as f:
            author_info=json.load(f)
        if author_info['id']!='empty':
            author_exist=True
    #如果作者匿名或者作者不存在，做特殊处理
    if author_is_anonymous==1 or author_exist==False:
        follower_count=-1
        following_count=-1
        mutual_followees_count=-1
        answer_count=-1
        question_count=-1
        articles_count=-1
        author_content=get_anonymous_str('name')+get_anonymous_str('headline')+get_anonymous_str('description')+get_anonymous_str('business')+get_anonymous_str('employments')
        author_tensor=torch.tensor(get_token_list(author_content,tokenizer)).unsqueeze(0)
        author_embedding= bert(author_tensor.to(device))[1]
    else:
        follower_count=int(author_info['follower_count'])
        following_count=int(author_info['following_count'])
        mutual_followees_count=int(author_info['mutual_followees_count'])
        answer_count=int(author_info['answer_count'])
        question_count=int(author_info['question_count'])
        articles_count=int(author_info['articles_count'])
        author_content=get_str(author_info,'name')+get_str(author_info,'headline')+get_str(author_info,'description')+get_str(author_info,'business')+get_str(author_info,'employments')
        author_tensor=torch.tensor(get_token_list(author_content,tokenizer)).unsqueeze(0)
        author_embedding= bert(author_tensor.to(device))[1]
    feature=[hot,is_author,is_author_top,author_is_anonymous,like_count,follower_count,following_count,mutual_followees_count,answer_count,question_count,articles_count]
    #加入两种编码的信息
    feature.extend(content_embedding.to('cpu').tolist()[0])
    feature.extend(author_embedding.to('cpu').tolist()[0])
    return feature
def get_article_feature(article,url_token,tokenizer,bert):
    hot=-1 #是否热门
    is_author=1 # 是否为回答作者
    is_author_top=-1 # 是否被作者置顶
    author_is_anonymous=-1 # 获取评论的作者是否匿名
    like_count=-1 #获取被点赞数量
    content='content:'+article # 获取评论的内容
    content_tensor=torch.tensor(get_token_list(content,tokenizer)).unsqueeze(0)
    content_embedding= bert(content_tensor.to(device))[1]
    #获取作者信息
    author_exist=False
    if os.path.exists(user_path+'/'+url_token+'.json'):
        with open(user_path+'/'+url_token+'.json','r',encoding='utf-8') as f:
            author_info=json.load(f)
        if author_info['id']!='empty':
            author_exist=True
    #如果作者匿名或者作者不存在，做特殊处理
    if author_is_anonymous==1 or author_exist==False:
        follower_count=-1
        following_count=-1
        mutual_followees_count=-1
        answer_count=-1
        question_count=-1
        articles_count=-1
        author_content=get_anonymous_str('name')+get_anonymous_str('headline')+get_anonymous_str('description')+get_anonymous_str('business')+get_anonymous_str('employments')
        author_tensor=torch.tensor(get_token_list(author_content,tokenizer)).unsqueeze(0)
        author_embedding= bert(author_tensor.to(device))[1]
    else:
        follower_count=int(author_info['follower_count'])
        following_count=int(author_info['following_count'])
        mutual_followees_count=int(author_info['mutual_followees_count'])
        answer_count=int(author_info['answer_count'])
        question_count=int(author_info['question_count'])
        articles_count=int(author_info['articles_count'])
        author_content=get_str(author_info,'name')+get_str(author_info,'headline')+get_str(author_info,'description')+get_str(author_info,'business')+get_str(author_info,'employments')
        author_tensor=torch.tensor(get_token_list(author_content,tokenizer)).unsqueeze(0)
        author_embedding= bert(author_tensor.to(device))[1]
    feature=[hot,is_author,is_author_top,author_is_anonymous,like_count,follower_count,following_count,mutual_followees_count,answer_count,question_count,articles_count]
    #加入两种编码的信息
    feature.extend(content_embedding.to('cpu').tolist()[0])
    feature.extend(author_embedding.to('cpu').tolist()[0])
    return feature
def get_features(data,article,user_file,tokenizer,bert):
    features=[]
    #先加入文章本身的特征
    article_feature=get_article_feature(article,user_file,tokenizer,bert)
    features.append(article_feature)
    #再加入评论的特征
    for comment in data:
        features.append(get_feature(comment,tokenizer,bert))
        for child_comment in comment['child_comment']:
            features.append(get_feature(child_comment,tokenizer,bert))
    return features
def construct_graph(name,article,user_file,tokenizer,bert):
    edges=[]
    features=[]
    data=[]
    if os.path.exists(review_path+'/'+name):
        with open(review_path+'/'+name,'r',encoding='utf-8') as f:
            data=json.load(f)
    #如果评论文件存在,data就不是空列表.否则不做处理即可，该图只有一个节点
    edges=get_edges(data)
    #构建特征的时候需要把文章本身和用户一起加进来
    features=get_features(data,article,user_file,tokenizer,bert)
    return torch.tensor(features),torch.tensor(edges)
def get_name_list():
    file_list=[]
    train_folder=root_path+'ZHIHU_train/text/'
    val_folder=root_path+'ZHIHU_val/text/'
    test_folder=[root_path+'ZHIHU_test'+str(i)+'/'+'text/' for i in range(13)]
    file_list.extend(os.listdir(train_folder))
    file_list.extend(os.listdir(val_folder))
    for f in test_folder:
        file_list.extend(os.listdir(f))
    new_list=[x.replace('.txt','') for x in file_list]
    return new_list
#根据文件名称对应ad表格中的行,并且找到原始文本，评论文件名，作者文件名
def name_to_info(name):
    # 使用正则表达式匹配数字和字符部分
    match = re.match(r'(\d+)(\D+)(\d+)', name)
    topic,cla,row=match.group(1),match.group(2),match.group(3)
    df = pd.read_excel(root_path+'original_files/'+str(topic)+'/'+cla+'_merge.xlsx',dtype=str)
    content = df.iloc[int(row), 3]
    review = df.iloc[int(row), 12]
    user_token = df.iloc[int(row), 13]
    return str(content),str(review),str(user_token)
def construct(name_list,tokenizer,bert):
    for name in tqdm(name_list):
        content,review,user_token=name_to_info(name)
        features,edges=construct_graph(review+'.json',content,user_token,tokenizer,bert)
        #保存两个tensor列表到两个文件夹中
        save_tensor(features,save_feature_path+'/'+name+'.pkl')
        save_tensor(edges,save_edge_path+'/'+name+'.pkl')
def get_bert():
    #初始化一个bert model
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name, add_pooling_layer=True)
    model.to(device)
    return tokenizer,model

#初始化编码模型
tokenizer,bert=get_bert()
#获得所有需要构建图的文件名
name_list=get_name_list()
#对所有需要构图的文件构图并保存特征
construct(name_list,tokenizer,bert)