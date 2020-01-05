#! /usr/bin/sh

# 生成训练语料
echo """dogs chase cats
dogs bark
cats meow
dogs chase birds
cats chase birds
dogs chase the cats
the birds chirp""" >corpus.txt

# 生成测试语料
echo """cats meow
dogs chase the birds
birds chirp
Wang dogs Zhi Guo """ >test_corpus.txt 

echo "训练语料"
cat corpus.txt

echo
echo "测试语料"
cat test_corpus.txt

# 训练ngram语言模型
echo "训练："
ngram-count -text corpus.txt -order 2 -write corpus_katz.count -lm corpus_katz.lm

echo
echo "查看计数文件"
cat corpus_katz.count

echo
echo "查看模型文件"
cat corpus_katz.lm

# 预测句子概率
echo "预测："
ngram -lm corpus_katz.lm -ppl test_corpus.txt -debug 2




