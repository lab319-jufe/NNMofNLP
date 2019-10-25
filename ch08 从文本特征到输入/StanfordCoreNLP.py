from stanfordcorenlp import StanfordCoreNLP
 
nlp = StanfordCoreNLP(r'E:/OneDrive/Documents/Ma&St-learning/NLP/stanford-corenlp-full-2018-10-05/', lang='zh')    #使用 lang='en'
 
sentence = '''项目组结合北京教科院的一些要求，讨论了一个初步的框架，
随着数据的更新以及刚开始几份报告的总结，
分析框架逐渐得到了完善，最终演变成下图的分析框架。'''
 
# print(nlp.word_tokenize(sentence))
# print(nlp.pos_tag(sentence))
print(nlp.parse(sentence))
print(nlp.dependency_parse(sentence))