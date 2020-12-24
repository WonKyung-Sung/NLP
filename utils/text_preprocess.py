#-*- coding: utf-8 -*-
# WK Module 

import re 
import numpy as np 
import pandas as pd


def HELP():
    a = '''
    '''
    print(a)

_BASE_CODE, _CHOSUNG, _JUNGSUNG = 44032, 588, 28
_CHOSUNG_LIST = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
_JUNGSUNG_LIST = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ']
_JONGSUNG_LIST = ['_', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

def sent2syllable(data):
    if type(data) == str: data = [data]
    return [[syllable for syllable in str(sentence)] for sentence in data]

def _silable2char3(string):
    '''
    음절 형태소 분석기 (초 중 종 성 위치 지킴)
    
    data: 데이터 (str or list(Series))    
    
    예시 
    
    '''
    result_pre = []
    for keyword in string:
        if re.match('.*[ㄱ-ㅎㅏ-ㅣ가-힣]+.*', keyword) is not None:
            try:
                char_code = ord(keyword) - _BASE_CODE
                char1 = int(char_code / _CHOSUNG)
                result_pre.append(_CHOSUNG_LIST[char1])

                char2 = int((char_code - (_CHOSUNG * char1)) / _JUNGSUNG)
                result_pre.append(_JUNGSUNG_LIST[char2])

                char3 = int((char_code - (_CHOSUNG * char1) - (_JUNGSUNG * char2)))
                result_pre.append(_JONGSUNG_LIST[char3])
            except: 
                result_pre.append(keyword)
                result_pre.append(keyword)
                result_pre.append(keyword)         

        else: 
            result_pre.append(keyword)
            result_pre.append(keyword)
            result_pre.append(keyword)
    return result_pre

def _silable2char(string):
    result_pre = []
    for keyword in string:
        if re.match('.*[ㄱ-ㅎㅏ-ㅣ가-힣]+.*', keyword) is not None:
            try:
                char_code = ord(keyword) - _BASE_CODE
                char1 = int(char_code / _CHOSUNG)
                result_pre.append(_CHOSUNG_LIST[char1])

                char2 = int((char_code - (_CHOSUNG * char1)) / _JUNGSUNG)
                result_pre.append(_JUNGSUNG_LIST[char2])

                char3 = int((char_code - (_CHOSUNG * char1) - (_JUNGSUNG * char2)))
                if char3 != 0:
                    result_pre.append(_JONGSUNG_LIST[char3])
            except: 
                result_pre.append(keyword)      

        else: 
            result_pre.append(keyword)
    return result_pre


def sent2char3(data):
    '''
    자소 형태소 분석기 (초 중 종 성 위치 지킴)
    
    data: 데이터 (str or list(Series))    
    
    예시 
    
    '''
    if type(data) == str: data = [data]
    syllables = sent2syllable(data)
    return [_silable2char3(string) for string in syllables]

def sent2char(data):
    '''
    자소 형태소 분석기 
    
    data: 데이터 (str or list(Series))    
    
    예시 
    
    '''
    if type(data) == str: data = [data]
    syllables = sent2syllable(data)
    return [_silable2char(string) for string in syllables]

def sent2khaiii(data, tag=False):
    '''
    카이 형태소 분석기
    
    data: 데이터 (str or list(Series))
    tag: 태그 포함 여부 (False or True)          
    
    예시 
        khaii(["우리집에 왜 왔니", "왜 왔니"])
        >>> [['우리', '집', '에', '왜', '오', '았', '니'], ['왜', '오', '았', '니']]

        khaii(["우리집에 왜 왔니", "왜 왔니"], tag=True)
        >>> [['우리/NP', '집/NNG', '에/JKB', '왜/MAG', '오/VV', '았/EP', '니/EC'],
            ['왜/MAG', '오/VV', '았/EP', '니/EC']]
    '''
    import khaiii
    api = khaiii.KhaiiiApi()
    api.open()
    if type(data) == str: data = [data]
    return [[a.lex + "/" + a.tag if tag ==True else a.lex  for word in api.analyze(str(sent)) for a in word.morphs] if str(sent).strip() else (sent) for sent in data]

def sent2okt(data, tag="morphs"):
    '''
    tag == morphs     #형태소 분석
    tag == nouns      #명사 분석
    tag == phrases    #구(Phrase) 분석
    tag == pos        #형태소 분석 태깅
    '''
    from konlpy.tag import Okt
    okt = eval("Okt()." + tag)
    if type(data) == str: data = [data]
    return [okt(str(sent)) for sent in data]

def sent2bert(data, MAX_LEN=500):
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
    def bert_tokenizer(sent, MAX_LEN=MAX_LEN):
        # attention_mask = encoded_dict["attention_mask"]
        # token_type_id=encoded_dict["token_type_ids"]
        return tokenizer.encode_plus(
            text=sent,
            add_special_tokens=True,
            max_length =MAX_LEN,
            pad_to_max_length=True,
            return_attention_mask=True)['input_ids']
    
    if type(data) == str: data = [data]
    return [bert_tokenizer(str(sent)) for sent in data]

def vocabulary_maker(data, min_count = 10, vocabulary_size=None):    
    '''
    vocabulary 만드는 함수 
    
    input 설명 
        data = 
            [ [ㄷ, ㅣ, _, ㅈ, ㅏ, _, ㅇ, ㅣ, ㄴ, ㅇ, ㅡ, ㄹ],
              [ㅍ, ㅗ, ㄹ, ㄹ, ㅣ, _, ㅅ, ㅡ, _, ㅅ, ㅡ, _ ] ]
        min_count = 최소 몇개 이상 출현한 단어를 사용 할것인지
        vocabulary_size = 몇개의 vocabulary를 만들 것인지

    output 설명 (dictionary, vocabulary_size)
        dictionary = {'PAD': 0, 'UNK': 1, ' ': 2, '_': 3,  'ㅇ': 4,  'ㅏ': 5, 'ㄴ': 6}
    '''
    
    from collections import Counter
    import numpy as np

    words = [word for sent in data for word in sent]

    if vocabulary_size == None:
        vocabulary_size = len(list(set(words)))

    vocabulary = [("PAD", min_count+1), ("UNK", min_count+1)] + Counter(words).most_common(vocabulary_size - 1)
    vocabulary = np.array([word for word, count in vocabulary if count > min_count])
    dictionary = {word: code for code, word in enumerate(vocabulary)}
    vocabulary_size = len(dictionary)
    return dictionary


def cohesion_score(data, min_count=10):
    from collections import defaultdict
    L = defaultdict(int)
    for sent in data:
        for eojeol in str(sent).split():
            for e in range(1, len(eojeol)+1):
                subword = eojeol[:e]
                L[subword] += 1
    print('num subword = %d' % len(L))

    def get_cohesion(word, min_count=10):
        if (not word) or ((word in L) == False): 
            return 0.0
        n = len(word)
        if n == 1:
            return 0
        word_freq = L.get(word, 0)
        base_freq = L.get(word[:1], 0)
        if base_freq == 0:
            return 0.0
        else:
            return np.power((word_freq / base_freq), 1 / (n - 1))

    return  {word:get_cohesion(word) for word, count in L.items() if count >= min_count and len(word) >= 1}

# print('n computed = {}'.format(len(cohesion_score)))

class LTokenizer:
    
    def __init__(self, scores=None, default_score=0.0):
        self.scores = scores if scores else {}
        self.ds = default_score
        
    def tokenize(self, sentence, tolerance=0.0, flatten=True, remove_r=False):
        tokens = [self._eojeol_to_lr(token, tolerance) for token in sentence.split()]        
        if remove_r:
            tokens = [token[0] for token in tokens]        
        if (flatten) and (remove_r == False):
            tokens = [subtoken for token in tokens for subtoken in token if subtoken]        
        return tokens
    
    def _eojeol_to_lr(self, token, tolerance=0.0):
        n = len(token)
        if n <= 2:
            return (token, '')
        
        candidates = [(token[:e], token[e:]) for e in range(2, n + 1)]
        candidates = [(self.scores.get(t[0], self.ds), t[0], t[1]) for t in candidates]
        
        if tolerance > 0:
            max_score = max([c[0] for c in candidates])
            candidates = [c for c in candidates if (max_score - c[0]) <= tolerance]
            best = sorted(candidates, key=lambda x:len(x[1]), reverse=True)[0]
        else:
            best = sorted(candidates, key=lambda x:(x[0], len(x[1])), reverse=True)[0]
            
        return (best[1], best[2])

    
def tokenizer_c(data, min_count=50, score=0):
    c_score = cohesion_score(data, min_count=50)
    tokenizer = LTokenizer(c_score, score)
    coi1 = [tokenizer.tokenize(str(doc)) for doc in data]
    return coi1, tokenizer, c_score
