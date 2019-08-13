import torch
from nltk import word_tokenize
import numpy as np

def makeVocab(filename):
    word_to_id = {'<PAD>':0, '<SOS>':1, '<EOS>':2, '<unk>': 3, '<mask>': 4}
    id_to_word = {0: '<PAD>', 1: '<SOS>', 2: "<EOS>", 3: '<unk>', 4: '<mask>'}
    idx = 5
    label_list = ['<SOS>', '<EOS>', '<PAD>', '<unk>', '<mask>']
    # label_list = []
    n=1
    for line in open(filename):
        if line == '\n':
            continue
        fields = line.split()
        if len(fields) > 1:
            label = ' '.join(fields[:])
        else:
            label = fields[0]
        if label not in label_list:
            print(n)
            n += 1
            word_to_id[label] = idx
            id_to_word[idx] = label
            idx += 1
            label_list.append(label)
    for i in ['<SOS>', '<EOS>', '<PAD>', '<unk>', '<mask>']:
        label_list.remove(i)

    return word_to_id, id_to_word, label_list



def makeData(srcFile, ori_srcFile, word_to_id, label = 1, if_shuff=True):

    original_text = []
    original_mask_text = []
    original_id = []
    original_mask_id = []
    original_label = []
    original_length = []
    max_length = 17
    count = 0

    print('Processing %s  ...' % srcFile)
    srcF = open(srcFile, "r")
    ori_srcf = open(ori_srcFile, "r")
    srcSet = srcF.readlines()
    ori_Set = ori_srcf.readlines()
    # labelSet = labelF.readlines()

    for i in range(len(srcSet)):
        if i % 5000 == 0:
            print("now: ", i)
        id_line = []
        mask_id_line = []
        srcSet[i] = srcSet[i].strip()
        ori_Set[i] = ori_Set[i].strip()


        text_line = ["<SOS>"] + ori_Set[i].split() + ["<EOS>"]
        text_mask_line = ["<SOS>"] + srcSet[i].split() + ["<EOS>"]
        if len(text_line) > max_length:
            continue
        original_text.append(text_line)
        original_mask_text.append(text_mask_line)
        original_label.append(label)
        original_length.append(len(text_line))


        # max_length = 18
        for j in range(len(original_text[count])):
            try:
                id = word_to_id[original_text[count][j]]

                # if original_text[i][j]=='<mask>':
                #     mask_id = 1
                # else:
                #     mask_id = 0
                # print(i)
                # print(j)
                a = original_mask_text[count][j]
                mask_id = word_to_id[a]
                # if original_mask_text[i][j] == '<mask>':
                #     mask_id = 0
                # else:
                #     mask_id = 1
            except KeyError:
                id = 3
            id_line.append(id)
            mask_id_line.append(mask_id)
        original_id.append(id_line)
        original_mask_id.append(mask_id_line)
        count += 1



    print('... padding')
    for i in range(len(original_text)):
        if original_length[i] < max_length:
            for j in range(max_length - original_length[i]):
                original_text[i].append("<PAD>")
                original_mask_text[i].append('<PAD>')
                original_id[i].append(0)
                original_mask_id[i].append(0)



    Dataset = {"text": original_text, "mask_text":original_mask_text, "length": original_length,
               "text_ids": original_id, "mask_text_ids":original_mask_id, "labels": original_label}
    return Dataset, max_length


def makeBatch(Dataset, batch_size):
    Dataset_total = []
    text = []
    mask_text =[]
    length = []
    text_ids = []
    mask_text_ids = []
    labels = []

    temp = {"text":text, "mask_text":mask_text, "length":length, "text_ids":text_ids,
            "mask_text_ids":mask_text_ids, "labels":labels}

    for i in range(len(Dataset['text'])):
        temp["text"].append(Dataset['text'][i])
        temp["mask_text"].append(Dataset['mask_text'][i])
        temp["length"].append(Dataset['length'][i])
        temp["text_ids"].append(Dataset['text_ids'][i])
        temp["mask_text_ids"].append(Dataset['mask_text_ids'][i])
        temp["labels"].append(Dataset['labels'][i])

        if ((i+1) % batch_size == 0):


            store = {"text":[row for row in temp['text']], "mask_text":[row for row in temp['mask_text']],
                     "length":[row for row in temp['length']], "text_ids":[row for row in temp['text_ids']],
                     "mask_text_ids":[row for row in temp['mask_text_ids']],"labels":[row for row in temp['labels']]}
            Dataset_total.append(store)
            temp['text'].clear()
            temp['mask_text'].clear()
            temp['length'].clear()
            temp['text_ids'].clear()
            temp['mask_text_ids'].clear()
            temp['labels'].clear()

    for i in range(len(Dataset_total)):
        raw_inputs = Dataset_total[i]
        input_text_ids = torch.LongTensor(raw_inputs["text_ids"])
        input_mask_text_ids = torch.LongTensor(raw_inputs["mask_text_ids"])
        input_labels = torch.LongTensor(raw_inputs["labels"])
        input_length = torch.tensor(raw_inputs["length"])

        #input_nn_vector = torch.FloatTensor(raw_inputs["nn_vector"])
        inputs = {"text":raw_inputs["text"], "mask_text":raw_inputs["mask_text"],
                  "labels":input_labels.cuda(),"length":input_length.cuda(),
                  "text_ids":input_text_ids.cuda(),"mask_text_ids":input_mask_text_ids.cuda()}
        Dataset_total[i] = inputs

    # print(Dataset_total[0])
    return Dataset_total



def shuff(data0, data1, if_shuff=True):
    "shuff the two dataset"

    Dataset_total = {}
    text = []
    mask_text = []
    length = []
    text_ids = []
    mask_text_ids = []
    labels = []

    temp = {"text": text, "mask_text": mask_text, "length": length, "text_ids": text_ids,
            "mask_text_ids": mask_text_ids, "labels": labels}
    temp['text'] = data0['text'] + data1['text']
    temp['mask_text'] = data0['mask_text'] + data1['mask_text']
    temp['length'] = data0['length'] + data1['length']
    temp['text_ids'] = data0['text_ids'] + data1['text_ids']
    temp['mask_text_ids'] = data0['mask_text_ids'] + data1['mask_text_ids']
    temp['labels'] = data0['labels'] + data1['labels']

    if if_shuff==True:
        perm = torch.randperm(len(temp['text']))
        original_id = [temp['text_ids'][idx] for idx in perm]
        original_mask_id = [temp['mask_text_ids'][idx] for idx in perm]
        original_label = [temp['labels'][idx] for idx in perm]
        original_length = [temp['length'][idx] for idx in perm]
        original_text = [temp['text'][idx] for idx in perm]
        original_mask_text = [temp['mask_text'][idx] for idx in perm]
        # data_total = [data_total[idx] for idx in perm]

        data_total = {"text":original_text, "mask_text":original_mask_text,
                      "labels":original_label,"length":original_length,
                      "text_ids":original_id,"mask_text_ids":original_mask_id}

        return data_total

    data_total = {"text": temp['text'], "mask_text": temp['mask_text'],
                  "labels": temp['labels'], "length": temp['length'],
                  "text_ids": temp['text_ids'], "mask_text_ids": temp['mask_text_ids']}

    return data_total

# word_to_id, id_to_word , voc = makeVocab('data/yelp/vocab_1.11')
# # _, _, sentiword1 =makeVocab('data/sentiwordnet/SentiWordNet_pos')
# # _, _, sentiword0 =makeVocab('data/sentiwordnet/SentiWordNet_nev')
# Dataset1, max_length = makeData('data/filter.new_dev1.text', 'data/yelp/sentiment.dev1.text',word_to_id, label = 1)
# Dataset0, max_length = makeData('data/filter.new_dev0.text', 'data/yelp/sentiment.dev0.text',word_to_id, label = 0)
# Dataset1_total = makeBatch(Dataset1,5)
# Dataset0_total = makeBatch(Dataset0,5)
# data_total = shuff(Dataset1, Dataset0)
# data_total = makeBatch(data_total,5)
# # print(Dataset_total[0])
# print(max_length)