import torch
import numpy as np
import torch.nn as nn
import time
import sys
from config import Config
# from model import CtrlGenModel
# from model_seq2seq3 import CtrlGenModel
from model4 import CtrlGenModel
from CNNModels import CnnTextClassifier
import torchtext.vocab as torch_vocab
import argparse
import pickle
import data_process

parser = argparse.ArgumentParser(description='main.py')


parser.add_argument('-if_eval', default=False, type=bool,
                    help="""If the result should be evaluated.""")
parser.add_argument('-if_classifier',default=False,type=bool,
                    help="""If we can test the result of the generated text.""")
# parser.add_argument('-if_saveData',default=True,type=bool,
#                     help="""If we need to save the data used.""")
parser.add_argument('-file_save', default='./reproduce.txt', type=str,
                    help="""The name of the file which saved the result of the generated text.""")
parser.add_argument('-checkpoint',default='save/amazon_7.10/no_loss_aecheckpoint_pre_loss8.3318.pt', type=str,
                    help="""The checkpoint of the model we trained before.""")
parser.add_argument('-checkpoint_path',default='', type=str,
                    help="""The path of the checkpoint.""")
parser.add_argument('-batch_size',default=128, type=int,
                    help="""Batch size.""")
parser.add_argument('-pretrain',default=1,type=int,
                    help="""pretrained epoch before add class loss""")
parser.add_argument('-datapath',default="./data/yelp/",type=str,
                    help="""Path to the dataset.""")
parser.add_argument('-gpu',default=0,type=int,
                    help="""Device to run on""")
parser.add_argument('-max_epoch',default=2,type=int,
                    help="""Max number of epochs""")



opt = parser.parse_args()


def trainModel(model, id_to_word, train_total, dev_total, test_total, g_vars, d_vars, config, max_epoch,
               batch_num, criterion1, criterion2, lambda_g, gamma, batch_size, max_length, if_eval, pretrain):

    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    sys.stdout.flush()
    batch_num_count = 0
    max_acc = 0.77
    lambda_gen = 0

    if if_eval == True:

        f1 = open('result/amazon_7.10/no_ae_loss', "w")
        total_count = 0
        acc_count = 0


        # for mn in range(len(test_total)):
        #     probs_trans, classes_trans = model(test_total[mn], max_length,if_dis=True)
        #     acc_count += torch.sum(classes_trans == test_total[mn]["labels"])
        #     total_count += len(test_total[mn]["labels"])
        # acc_d = acc_count.cpu().float() / float(total_count)
        # print(acc_d)
        # return
        for mn in range(len(test_total)):
            soft_outputs, _, probs_trans, classes_trans, _, _ = model(test_total[mn], max_length)
            # _,soft_outputs,  _, _, probs_trans, classes_trans = model(test_total[mn], max_length)
            total_count += len(test_total[mn]["labels"])
            # result2 = torch.argmax(soft_outputs,2)
            inp = torch.argmax(soft_outputs[:, :-1, :], 2)
            sentence = torch.where(test_total[mn]['mask_text_ids'][:, 1:] == 4, inp, test_total[mn]['text_ids'][:, 1:])
            for i in range(batch_size):
                p = ""
                q = ""
                m = ""
                #p = [id_to_word[j.item()] for j in eval_data[79]["text_ids"][i][:]]
                for j in test_total[mn]["text"][i][1:]:
                    if j == "<EOS>":
                        break
                    p = p+" "+j
                for j in sentence[i][:]:
                    if j == 2:
                        break
                    m = m+ " "+ id_to_word[j.item()]
                f1.write(p.strip()+'\n')
                f1.write(m.strip()+'\t'+ str(classes_trans[i].cpu())+'\n')

            # print("original: ",p)
            # # print("generated: ",q)
            # print("soft:",m)
            acc_count += torch.sum(classes_trans == (1 - test_total[mn]["labels"]))
        acc_d = acc_count.cpu().float() / float(total_count)
        print("acc_d: ", acc_d)
        return


    for epoch in range(1, max_epoch):
        batch_num_count = 0
        # shuffule
        perm = torch.randperm(len(train_total))
        train_data = [train_total[idx] for idx in perm]
        pre = True
        # pre = False


        for batch_train_data in train_data:

            inputs = train_data[batch_num_count]
            loss_g_ae = 0
            loss_trans_ae = 0
            loss_g_clas = 0



            if epoch > pretrain:
                pre = False
                # lambda_g = 2
                # gamma = max(0.001, gamma * config.gamma_decay)
            gamma = 1 / (100 ** (batch_num_count / 6300))

            # Train the classifier
            d_vars.zero_grad()
            probs, classes = model(inputs, max_length, if_dis=True)
            loss_d_clas = criterion2(probs, inputs["labels"])


            # train generator
            if pre==True:

                loss_d_clas.backward()
                d_vars.step()

                g_vars.zero_grad()
                g_outputs, soft_outputs, probs, classes, probs_trans, classes_trans = model(inputs, max_length, pre=pre,
                                                                                            gamma=gamma)
                for i in range(max_length - 1):
                    loss_g_ae += criterion1(g_outputs[:, i, :].cuda(), inputs["text_ids"][:, i + 1])
                    loss_trans_ae += criterion1(soft_outputs[:, i, :].cuda(), inputs["text_ids"][:, i + 1])
                loss_g = loss_g_ae
                loss_g.backward()
                g_vars.step()

            else:
                g_vars.zero_grad()
                g_outputs, soft_outputs, probs, classes, probs_trans, classes_trans = model(inputs, max_length,
                                                                                            gamma=gamma)
                # for i in range(len(g_outputs)):
                #     for j in range(inputs['length'][i]):
                #         if inputs['mask_text_ids'][i][j] == torch.tensor([4]).cuda():
                #             input = g_outputs[i, j-1,:].unsqueeze(0)
                #             target = inputs['text_ids'][i][j].unsqueeze(0)
                #             loss_ae =  loss_ae + criterion1(input, target)
                #
                # loss_g_ae = loss_ae / batch_size
                for i in range(max_length - 1):
                    loss_g_ae += criterion1(g_outputs[:, i, :].cuda(), inputs["text_ids"][:, i + 1])

                loss_g_clas_trans = criterion2(probs_trans, 1 - inputs["labels"])
                loss_g_clas = criterion2(probs, 1 - inputs["labels"])
                loss_g = lambda_g * loss_g_clas_trans + lambda_g * loss_g_clas
                # loss_g.backward()
                # g_vars.step()


            if (batch_num_count % 100) == 0 and (batch_num_count != batch_num):
                print("change the gamma to: " + str(gamma))
                if pre:
                    print("Epoch:{0} Step:{1} Loss_g : {2} loss_g_ae:{3}  loss_trans_ae:{4}".format(epoch, batch_num_count,
                                                                                                    loss_g, loss_g_ae, loss_trans_ae))
                else:
                    print(
                    "Epoch:{0} Step:{1} Loss_g : {2} Loss_g_ae: {3} loss_d_clas: {4} loss_g_clas: {5} loss_g_clas_trans {6}".format(
                        epoch, batch_num_count, loss_g, loss_g_ae, loss_d_clas, loss_g_clas, loss_g_clas_trans))

                # print sentence
                inp = torch.argmax(g_outputs[0], 1)
                ids = torch.where(inputs['mask_text_ids'][0, 1:] == 4, inp[:-1], inputs['text_ids'][0, 1:]).cpu().numpy()
                soft_inp = torch.argmax(soft_outputs[0], 1)
                soft_ids = torch.where(inputs['mask_text_ids'][0, 1:] == 4, soft_inp[:-1], inputs['text_ids'][0, 1:]).cpu().numpy()
                # ids = torch.argmax(g_outputs[0], 1).cpu().numpy()
                text = []
                soft_text = []

                for i in ids:
                    word = id_to_word[i]
                    text.append(word)
                for i in soft_ids:
                    word = id_to_word[i]
                    soft_text.append(word)
                print(inputs['mask_text'][0])
                print(inputs['text'][0])
                print(text)
                print(soft_text)

                if batch_num_count % 200 == 0 and epoch > pretrain:
                    total_count = 0
                    acc_count = 0
                    for mn in range(len(dev_total)):
                        eval_outputs, soft_outputs, probs, classes, probs_trans, classes_trans= model(dev_total[mn], max_length)
                        total_count += len(dev_total[mn]["labels"])
                        acc_count += torch.sum(classes == (1 - dev_total[mn]["labels"]))
                    acc = acc_count.cpu().float() / float(total_count)
                    print("acc", acc)
                    if acc > max_acc:
                        torch.save(model.state_dict(), 'save/yelp_7.06/adv_no_loss_ae' + 'checkpoint_%d_step_%d_acc_%.4f.pt'% (
                        epoch, batch_num_count, acc))

            if batch_num_count == (len(train_data) -1) and epoch==1:
                torch.save(model.state_dict(), 'save/amazon_7.10/no_loss_ae' + 'checkpoint_pre_loss%.4f.pt' % (loss_g))

            batch_num_count += 1


def make_pretrain_embeddings(glove,id_to_word,emb_dim):
    weights_matrix = []
    for i in range(len(id_to_word)):
        try:
            weights_matrix.append(list(glove.vectors[glove.stoi[id_to_word[i]]]))
        except KeyError:
            a = np.random.normal(scale=0.6, size=(emb_dim,))
            weights_matrix.append(np.random.normal(scale=0.6, size=(emb_dim,)))

    new_weight = torch.FloatTensor(weights_matrix).cuda()
    return new_weight

    
def main():
    #Config
    config = Config()    
    if torch.cuda.is_available() and opt.gpu == -1:
        print("WARNING: You have a CUDA device, so you should probably run with -gpu 0")
    if opt.gpu != -1:
        torch.cuda.set_device(opt.gpu)
        print("Using GPU: ",opt.gpu)

    word_to_id, id_to_word, _= data_process.makeVocab('data/amazon_7.10/voc')
    vocab_size = len(word_to_id)
    print("Vocab size",len(word_to_id))

    
    #define the max length of the sentence including <POS> and <EOS>
    max_length = 17
    # max_length_dev = 17

    batch_size = opt.batch_size
    print("batch_size :", batch_size)

    # Make the weight matrix.
    dic_glove = torch_vocab.GloVe(name='twitter.27B', dim=100)
    weights_matrix = make_pretrain_embeddings(dic_glove, id_to_word, config.model["embed_size"])
    # Glove used to embed the noun contents.
    # glove = torch_vocab.GloVe(name='840B', dim=300)

    Dataset_train1_total = []
    Dataset_train0_total = []
    Dataset_train_total = []
    Dataset_dev0_total = []
    Dataset_dev1_total = []
    Dataset_dev_total = []
    Dataset_test1_total = []
    Dataset_test0_total = []
    Dataset_test_total = []


    if opt.if_eval == False:
        # Dataset_train1, _ = data_process.makeData('data/yelp_1.11/filter.1.11_new_train1.text', 'data/yelp_1.11/filter.train1.text', word_to_id, label=1)
        # Dataset_train0, _ = data_process.makeData('data/yelp_1.11/filter.1.11_new_train0.text', 'data/yelp_1.11/filter.train0.text', word_to_id, label=0)
        Dataset_train1, _ = data_process.makeData('data/amazon_7.10/filter.train1.text', 'data/amazon_7.10/original.train1.text',
                                                   word_to_id, label=1)
        Dataset_train0, _ = data_process.makeData('data/amazon_7.10/filter.train0.text', 'data/amazon_7.10/original.train0.text',
                                                   word_to_id, label=0)
        # Dataset_train1,_ = data_process2.makeData('data/yelp/filter.dev1.text',word_to_id,label = 1)
        # Dataset_train0, _ = data_process2.makeData('data/yelp/filter.dev0.text', word_to_id,label = 0)
        Dataset_train = data_process.shuff(Dataset_train0, Dataset_train1, if_shuff=True)
        Dataset_train1_total = data_process.makeBatch(Dataset_train1, batch_size)
        Dataset_train0_total = data_process.makeBatch(Dataset_train0, batch_size)
        Dataset_train_total = data_process.makeBatch(Dataset_train, batch_size)


        # Dataset_dev0,_ = data_process.makeData('data/yelp_1.11/filter.1.11_new_dev0.text', 'data/yelp_1.11/filter.dev0.text', word_to_id, label = 0)
        # Dataset_dev1, _ = data_process.makeData('data/yelp_1.11/filter.1.11_new_dev1.text', 'data/yelp_1.11/filter.dev1.text', word_to_id, label = 1)
        Dataset_dev0, _ = data_process.makeData('data/amazon_7.10/filter.dev0.text', 'data/amazon_7.10/original.dev0.text',
                                                 word_to_id, label=0)
        Dataset_dev1, _ = data_process.makeData('data/amazon_7.10/filter.dev1.text', 'data/amazon_7.10/original.dev1.text',
                                                 word_to_id, label=1)
        Dataset_dev = data_process.shuff(Dataset_dev0, Dataset_dev1, if_shuff=True)
        Dataset_dev0_total = data_process.makeBatch(Dataset_dev0, batch_size)
        Dataset_dev1_total = data_process.makeBatch(Dataset_dev1, batch_size)
        Dataset_dev_total = data_process.makeBatch(Dataset_dev, batch_size)

        train1_batch_num = len(Dataset_train1_total)
        train0_batch_num = len(Dataset_train0_total)
        train_batch_num = len(Dataset_train_total)

        dev1_batch_num = len(Dataset_dev1_total)
        dev0_batch_num = len(Dataset_dev0_total)
        dev_batch_num = len(Dataset_dev_total)

        print("train1_batch_num: ", train1_batch_num)
        print("train0_batch_num: ", train0_batch_num)
        print("train_batch_num: ", train_batch_num)
        print("dev1_batch_num: ", dev1_batch_num)
        print("dev0_batch_num: ", dev0_batch_num)
        print("dev_batch_num: ", dev_batch_num)


    if opt.if_eval == True:
        Dataset_test1,_ = data_process.makeData('data/amazon_7.10/filter.test1.text', 'data/amazon_7.10/original.test1.text', word_to_id, label = 1)
        Dataset_test0, _ = data_process.makeData('data/amazon_7.10/filter.test0.text', 'data/amazon_7.10/original.test0.text', word_to_id, label = 0)
        #Dataset_test,_ = data_process2.makeData('data/yelp/reference_filter_source1.txt', 'data/yelp/reference_source1.txt',word_to_id, label = 1)


        Dataset_test = data_process.shuff(Dataset_test0, Dataset_test1, if_shuff=False)
        Dataset_test1_total = data_process.makeBatch(Dataset_test1, batch_size)
        Dataset_test0_total = data_process.makeBatch(Dataset_test0, batch_size)
        Dataset_test_total = data_process.makeBatch(Dataset_test, batch_size)

        test1_batch_num = len(Dataset_test1_total)
        test0_batch_num = len(Dataset_test0_total)
        test_batch_num = len(Dataset_test_total)
        print("test1_batch_num: ", test1_batch_num)
        print("test0_batch_num: ", test0_batch_num)
        print("test_batch_num: ", test_batch_num)
    
    
    # AE model
    model = CtrlGenModel(config,vocab_size,batch_size,weights_matrix)
    print(model)
    model = model.cuda()
    print("parameters()",model.parameters())


    # model seq2seq
    # Parameters needed to be optimized in generator.
    g_vars = torch.optim.Adam(
        [{'params': model.embedder.parameters()},
         {'params': model.encoder.parameters()},
         {'params': model.label_connector.parameters()},
         {'params': model.connector.parameters()},
         {'params': model.decoder.parameters()},
         ], lr=config.learning_rate)



    # Parameters needed to be optimized in classifier.
    d_vars = torch.optim.Adam(
        [
            {'params': model.classifier.parameters()},
            {'params': model.clas_embedder.parameters()},
        ], lr=config.learning_rate)


    criterion1 = nn.NLLLoss()
    criterion2 = nn.CrossEntropyLoss()
    criterion1 = criterion1.cuda()
    criterion2 = criterion2.cuda()

    print("Begining training.")
    # if we need to restore the model.
    if opt.checkpoint !='':
        model.load_state_dict(torch.load(opt.checkpoint))
        print('loading model...')
    file_name = opt.file_save
    lambda_n = 0
    lambda_g = 1
    gamma = 1.0

    trainModel(model, id_to_word, Dataset_train_total, Dataset_dev_total, Dataset_test_total,  g_vars, d_vars, config,
               opt.max_epoch, len(Dataset_train_total), criterion1, criterion2, lambda_g, gamma, batch_size, max_length,
               opt.if_eval, opt.pretrain)
    print("End of training!")

if __name__ == "__main__":
    main()
