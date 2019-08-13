import re


def mask_word_list(filename):

    # 7.04
    # label_list = ['no', 'never', 'little', 'few', 'nobody', 'nothing', 'none', 'neither', 'hardly', 'not',
    #               'lot', 'many', 'everybody', 'everythine', 'all', 'both', 'either',
    #               'am\'t', 'isn\'t', 'aren\'t', 'wasn\'t', 'weren\'t', 'don\'t', 'doesn\'t', 'didn\'t', 'hasn\'t',
    #               'haven\'t', 'hadn\'t', 'can\'t', 'couldn\'t', 'shouldn\'t', 'wouldn\'t', 'won\'t', 'willn\'t',
    #               'ain\'t', 'is', 'are', 'was', 'were', 'do', 'did', 'does', 'has', 'have', 'had', 'should', 'will',
    #               'can', 'could', 'would', '\'s', '\'d', '\'m', '\'ve', '\'re', '\'ll']

    # 6.27
    # label_list = ['no', 'never', 'little', 'few', 'nobody', 'nothing', 'none', 'neither', 'hardly', 'not',
    #               'am\'t', 'isn\'t', 'aren\'t', 'wasn\'t', 'weren\'t', 'don\'t', 'doesn\'t', 'didn\'t', 'hasn\'t',
    #               'haven\'t', 'hadn\'t', 'can\'t', 'couldn\'t', 'shouldn\'t', 'wouldn\'t', 'won\'t', 'willn\'t',
    #               'ain\'t', 'is', 'are', 'was', 'were', 'do', 'did', 'does', 'has', 'have', 'had', 'should', 'will',
    #               'can', 'could', 'would', '\'s', '\'d', '\'m', '\'ve', '\'re', '\'ll']

    # 7.06
    label_list = ['no', 'never', 'little', 'few', 'nobody', 'nothing', 'none', 'neither', 'hardly', 'not',
                  'lot', 'many', 'everythine', 'all', 'both', 'either',
                  'am\'t', 'isn\'t', 'aren\'t', 'wasn\'t', 'weren\'t', 'don\'t', 'doesn\'t', 'didn\'t', 'hasn\'t',
                  'haven\'t', 'hadn\'t', 'can\'t', 'couldn\'t', 'shouldn\'t', 'wouldn\'t', 'won\'t', 'willn\'t',
                  'ain\'t', 'is', 'are', 'was', 'were', 'do', 'did', 'does', 'has', 'have', 'had', 'should', 'will',
                  'can', 'could', 'would', '\'s', '\'d', '\'m', '\'ve', '\'re', '\'ll', 'wtf']

    # label_list = ['no', 'never', 'little', 'few', 'nobody', 'nothing', 'none', 'neither', 'seldom', 'hardly']

    for line in open(filename):
        label = line[:-1]
        if label not in label_list:
            label_list.append(label)

    return label_list



def makeData(srcFile, sentword_word):


    print('Processing %s  ...' % srcFile)
    srcF = open(srcFile, "r")
    srcSet = srcF.readlines()
    number = len(srcSet)
    n = 0

    num = 0
    change = 0
    for i in range(len(srcSet)):

        if i%100==0:
            print("now: ", i)

        sent = srcSet[i].strip()
        sent = re.sub('alot', 'a lot', sent)
        sent = re.split(' ', sent)
        original_sent = srcSet[i].strip()
        original_sent = re.sub('alot', 'a lot', original_sent)
        original_sent = re.split(' ', original_sent)
        length = len(sent)
        # i = 0

        # while i < length-3:
        #
        #     n_word = sent[i] + ' ' + sent[i + 1] + ' ' + sent[i + 2] + ' ' + sent[i + 3]
        #     if n_word in sentword_word:
        #         change = 1
        #         sent[i] = '<mask>'
        #         sent[i + 1] = '<mask>'
        #         sent[i + 2] = '<mask>'
        #         sent[i + 3] = '<mask>'
        #         i += 4
        #     else:
        #         i += 1
        #
        # i = 0
        # while i < length-2:
        #
        #     n_word = sent[i] + ' ' + sent[i + 1] + ' ' + sent[i + 2]
        #     if n_word in sentword_word:
        #         change = 1
        #         sent[i] = '<mask>'
        #         sent[i + 1] = '<mask>'
        #         sent[i + 2] = '<mask>'
        #         i += 3
        #     else:
        #         i += 1
        #
        # i = 0
        # while i < length-1:
        #
        #     n_word = sent[i] + ' ' + sent[i + 1]
        #     if n_word in sentword_word:
        #         change = 1
        #         sent[i] = '<mask>'
        #         sent[i + 1] = '<mask>'
        #
        #         i += 2
        #     else:
        #         i += 1

        j = 0
        o_sent = sent
        while j < length:

            n_word = sent[j]
            if n_word in sentword_word:
                change = 1
                sent[j] = '<mask>'
            j+= 1

        if change==1:

            n += 1
            s = ''
            original_s = ''
            for z in range(len(sent)):
                s = s + sent[z] + ' '

                original_s = original_s + original_sent[z] + ' '
            w.write(s + '\n')
            y.write(original_s + '\n')
        change = 0
        num += 1
    return number, n



def replace(a, b, c):
    d = a.replace(b, c)
    return d

w = open('data/yelp_7.06/filter.train0.text', 'w')
y = open('data/yelp_7.06/original.train0.text', 'w')



t = [('am not', 'am\'t'), ('is n\'t', 'isn\'t'), ('are n\'t', 'aren\'t'), ('was n\'t', 'wasn\'t'),
     ('were n\'t', 'weren\'t'), ('do n\'t', 'don\'t'), ('did n\'t', 'didn\'t'), ('does n\'t', 'doesn\'t'),
     ('has n\'t', 'hasn\'t'), ('have n\'t', 'haven\'t'), ('had n\'t', 'hadn\'t'), ('ca n\'t', 'can\'t'),
     ('could n\'t', 'couldn\'t'), ('should n\'t', 'shouldn\'t'), ('would n\'t', 'wouldn\'t'), ('wo n\'t', 'won\'t'),
     ('will n\'t', 'willn\'t'), ('ai n\'t', 'ain\'t'),
     ('is not', 'isn\'t'), ('are not', 'aren\'t'), ('was not', 'wasn\'t'), ('were not', 'weren\'t'),
     ('do not', 'don\'t'), ('does not', 'doesn\'t'), ('did not', 'didn\'t'), ('can not', 'can\'t'),
     ('could not', 'couldn\'t'), ('has not', 'hasn\'t'), ('have not', 'haven\'t'), ('had not', 'hadn\'t'),
     ('should not', 'shouldn\'t'), ('would not', 'wouldn\'t'), ('will not', 'willn\'t')]


with open('data/yelp/sentiment.train0.text', 'r') as f:
    copus = f.readlines()
    for (a, b) in t:
        copus = [replace(sent, a, b) for sent in copus]


with open('data/yelp_7.06/a.txt', 'w') as f:
    f.writelines(copus)


label_list = mask_word_list('data/sentiwordnet/negative-words.txt')

makeData('data/yelp_7.06/a.txt', label_list)


