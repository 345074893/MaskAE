import torch
import numpy as np
import torch.nn as nn
import time
import sys
from config import Config
# from model import CtrlGenModel
# from model_seq2seq3 import CtrlGenModel
from model_adv import CtrlGenModel
from CNNModels import CnnTextClassifier
import torchtext.vocab as torch_vocab
import argparse
import pickle
import data_process

parser = argparse.ArgumentParser(description='main.py')


parser.add_argument('-if_gen', default=False, type=bool,
                    help="""If the result should be evaluated.""")
parser.add_argument('-gan_type', default='standard', type=str,
                    help="""type of gan""")
parser.add_argument('-checkpoint', default='save/yelp_6.27/pre_mask/checkpoint_1_step_2000_loss_ae_10.2438.pt', type=str,
                    help="""The checkpoint of the model we trained before.""")
parser.add_argument('-pretrain', default=1, type=int,
                    help="""pretrained epoch before add class loss""")
parser.add_argument('-if_classifier', default=False, type=bool,
                    help="""If we can test the result of the generated text.""")
# parser.add_argument('-if_saveData',default=True,type=bool,
#                     help="""If we need to save the data used.""")
parser.add_argument('-file_save', default='./reproduce.txt', type=str,
                    help="""The name of the file which saved the result of the generated text.""")
parser.add_argument('-checkpoint_path', default='', type=str,
                    help="""The path of the checkpoint.""")
parser.add_argument('-batch_size', default=64, type=int,
                    help="""Batch size.""")
# parser.add_argument('-datapath',default="./data/yelp/",type=str,
#                     help="""Path to the dataset.""")
parser.add_argument('-gpu', default=0, type=int,
                    help="""Device to run on""")
parser.add_argument('-max_epoch', default=3,type=int,
                    help="""Max number of epochs""")
parser.add_argument('-g_step', default=1, type=int)
parser.add_argument('-d_step', default=5, type=int)



opt = parser.parse_args()

# A function to get different GAN losses
def get_gan_losses(d_out_real, d_out_fake, batch_size, gan_type):
    BCEWithLogitsLoss = nn.BCEWithLogitsLoss()
    if gan_type == 'standard':  # the non-satuating GAN loss
        d_loss_real = BCEWithLogitsLoss(d_out_real, torch.ones(batch_size, 1).cuda())
        d_loss_fake = BCEWithLogitsLoss(d_out_fake, torch.zeros(batch_size, 1).cuda())
        d_loss = d_loss_real+d_loss_fake
        g_loss = BCEWithLogitsLoss(d_out_fake, torch.ones(batch_size, 1).cuda())

    elif gan_type == 'RSGAN':
        d_loss = BCEWithLogitsLoss(d_out_real - d_out_fake, torch.ones(batch_size, 1).cuda())
        g_loss = BCEWithLogitsLoss(d_out_fake - d_out_real, torch.ones(batch_size, 1).cuda())


    return d_loss, g_loss


def trainModel(model, id_to_word, train_total, dev_total, test_total, g_vars, sentiment_vars, real_vars, config, opt,
               batch_num, criterion1, criterion2, lambda_g, gamma, batch_size, max_length):

    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    sys.stdout.flush()
    max_acc = 0.90
    lambda_gen = 0

    if opt.if_gen == True:

        f1 = open('result/2test_6.25.txt', "w")
        total_count = 0
        acc_count = 0
        for mn in range(len(test_total)):
            sentence, probs_trans, classes_trans = model(test_total[mn], max_length, if_gen=True)
            total_count += len(test_total[mn]["labels"])

            # inp = torch.argmax(outputs[:, :-1, :], 2)
            # sentence = torch.where(test_total[mn]['mask_text_ids'][:, 1:] == 4, inp, test_total[mn]['text_ids'][:, 1:])
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
                f1.write(m.strip()+'\n')

            # print("original: ",p)
            # # print("generated: ",q)
            # print("soft:",m)
            acc_count += torch.sum(classes_trans == (1 - test_total[mn]["labels"]))
        acc_d = acc_count.cpu().float() / float(total_count)
        print("acc_d: ", acc_d)

        return

    for epoch in range(2, opt.max_epoch):
        batch_num_count = 0

        # shuffule
        perm = torch.randperm(len(train_total))
        train_data = [train_total[idx] for idx in perm]
        pre = True
        if epoch == 2:
            train_g = True
        g_step =0
        d_step =0


        for inputs in train_data:

            # inputs = batch_train_data
            if epoch == opt.pretrain:

                # Train the classifier
                sentiment_vars.zero_grad()
                probs, classes = model(inputs, max_length, if_dis=True)
                loss_d_clas = criterion2(probs, inputs["labels"])

                loss_d_clas.backward()
                sentiment_vars.step()

                acc_count = torch.sum(classes == inputs["labels"])
                sentiment_class_acc = acc_count.cpu().float() / float(batch_size)

            if epoch > opt.pretrain:
                pre = False
            #     gamma = max(0.001, gamma * config.gamma_decay)

            # pretrain generator
            g_vars.zero_grad()
            if pre == True:

                # g_outputs, output_ids, lstm_logits, lstm_ids = model(inputs, max_length, pre=pre)
                g_outputs, output_ids = model(inputs, max_length, pre=pre)

                loss_g_ae = 0
                loss_g_ae_lstm = 0
                for i in range(max_length - 1):
                    loss_g_ae += criterion1(g_outputs[:, i, :].cuda(), inputs["text_ids"][:, i + 1])
                loss_g_ae = loss_g_ae/(int(max_length) - 1)
                loss_g_ae.backward()
                g_vars.step()
                # for i in range(max_length - 2):
                #     loss_g_ae_lstm += criterion1(lstm_logits[:, i, :].cuda(), inputs["text_ids"][:, i + 2])
                # loss_g_ae_lstm = loss_g_ae_lstm / (int(max_length) - 1)
                # loss_g_ae_lstm.backward()
                # real_vars.step()

                if (batch_num_count % 100) == 0 and (batch_num_count != batch_num):
                    print(
                        "Pretrain Epoch:{0} Step:{1} loss_g_ae : {2} loss_d_clas : {3} sentiment_class_acc : {4} ".format(
                            epoch, batch_num_count, loss_g_ae, loss_d_clas, sentiment_class_acc))
                    if batch_num_count % 200 == 0:
                        torch.save(model.state_dict(), 'save/yelp_6.27/pre_mask/' + 'checkpoint_%d_step_%d_loss_ae_%.4f.pt' % (
                            epoch, batch_num_count, loss_g_ae))
                    # print sentence
                    gen_text = []
                    lstm_text = []
                    for i in output_ids[0].detach().cpu().numpy():
                        word = id_to_word[i]
                        gen_text.append(word)
                    # for i in lstm_ids[0].detach().cpu().numpy():
                    #     word = id_to_word[i]
                    #     lstm_text.append(word)

                    print(inputs['text'][0])
                    print(gen_text)
                    # print(lstm_text)

                batch_num_count += 1

            else:
                if train_g == True:

                    g_vars.zero_grad()
                    g_step += 1

                    # strat adversaril train
                    "soft_output = (probs, classes, logits_real, logits_fake)"
                    soft_output_original, soft_output_trans = model(inputs, max_length, gamma=gamma)

                    original_d_loss, original_g_loss = get_gan_losses(soft_output_original[2], soft_output_original[3],
                                                                      batch_size, opt.gan_type)
                    trans_d_loss, trans_g_loss = get_gan_losses(soft_output_trans[2], soft_output_trans[3],
                                                                      batch_size, opt.gan_type)

                    loss_original_sentiment = criterion1(soft_output_original[0], inputs["labels"])
                    loss_trans_sentiment = criterion1(soft_output_trans[0], (1 - inputs["labels"]))

                    g_total_loss = original_g_loss + loss_original_sentiment + trans_g_loss + loss_trans_sentiment
                    g_total_loss.backward()
                    g_vars.step()

                    print("train generator Epoch:{0} Step:{1} total_loss:{2}".format(
                        epoch, batch_num_count, g_total_loss))
                    print(
                        "original_d_loss: {0} original_g_loss: {1} loss_original_sentiment {2}".format(original_d_loss,
                                                                                                       original_g_loss,
                                                                                                       loss_original_sentiment))
                    print(
                        "trans_d_loss: {0} trans_g_loss: {1} loss_trans_sentiment {2}".format(trans_d_loss,
                                                                                                  trans_g_loss,
                                                                                                  loss_trans_sentiment))

                    if g_step % opt.g_step == 0:
                        train_g = False

                    # print sentence
                    sentence, trans_prob, trans_classes = model(inputs, max_length, if_gen=True)
                    loss_trans_gen = criterion1(trans_prob, (1 - inputs["labels"]))
                    print('loss_trans_gen: {0} '.format(loss_trans_gen))
                    trans_text = []
                    for i in sentence[0].cpu().numpy():
                        word = id_to_word[i]
                        trans_text.append(word)

                    print(inputs['mask_text'][0])
                    print(inputs['text'][0])
                    print(trans_text)
                    print('')

                else:
                    d_step += 1
                    real_vars.zero_grad()

                    "soft_output = (probs, classes, logits_real, logits_fake)"
                    soft_output_original, soft_output_trans = model(inputs, max_length, gamma=gamma)

                    original_d_loss, original_g_loss = get_gan_losses(soft_output_original[2], soft_output_original[3],
                                                                      batch_size, opt.gan_type)
                    trans_d_loss, trans_g_loss = get_gan_losses(soft_output_trans[2], soft_output_trans[3],
                                                                batch_size, opt.gan_type)

                    d_loss_total = original_d_loss +trans_d_loss
                    d_loss_total.backward()
                    real_vars.step()

                    print("train discriminator Epoch:{0} Step:{1} d_loss:{2} original_d_loss:{3} trans_d_loss:{4}".format(
                        epoch, batch_num_count, d_loss_total, original_d_loss, trans_d_loss))
                    print("original_g_loss:{0} trans_g_loss:{1}".format(original_g_loss, trans_g_loss))


                if (batch_num_count + 1) % (opt.g_step + opt.d_step) == 0:
                    print(' ')
                    print('######################################################################################')
                    print(' ')
                    gamma = 1 / (100**(batch_num_count/6900))
                    print("change the gamma to: " + str(gamma))
                    train_g = True


                if  batch_num_count % 180 == 0:

                    # test on dev data
                    total_count = 0
                    acc_count = 0
                    for mn in range(len(dev_total)):

                        eval_outputs, probs_trans, classes_trans = model(dev_total[mn], max_length, if_gen=True)
                        total_count += len(dev_total[mn]["labels"])
                        acc_count += torch.sum(classes_trans == (1 - dev_total[mn]["labels"]))
                    acc = acc_count.cpu().float() / float(total_count)
                    print("acc", acc)
                    if acc > max_acc:
                        torch.save(model.state_dict(), 'save/yelp_6.27/adv_check/' + 'checkpoint_%d_step_%d_acc_%.4f.pt' % (
                            epoch, batch_num_count, acc))

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
        print("Using GPU: ", opt.gpu)

    word_to_id, id_to_word, _= data_process.makeVocab('data/yelp_6.27/vocab_6.27')
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

    Dataset_train_total = []
    Dataset_dev_total = []
    Dataset_test_total = []

    if opt.if_gen == False:
        # Dataset_train1, _ = data_process.makeData('data/yelp/sentiment.train1.text', 'data/yelp/sentiment.train1.text', word_to_id, label=1)
        # Dataset_train0, _ = data_process.makeData('data/yelp/sentiment.train0.text', 'data/yelp/sentiment.train0.text', word_to_id, label=0)
        Dataset_train1, _ = data_process.makeData('data/yelp_6.27/filter.train1.text', 'data/yelp_6.27/original.train1.text',
                                                   word_to_id, label=1)
        Dataset_train0, _ = data_process.makeData('data/yelp_6.27/filter.train0.text', 'data/yelp_6.27/original.train0.text',
                                                   word_to_id, label=0)
        # Dataset_train1,_ = data_process2.makeData('data/yelp/filter.dev1.text',word_to_id,label = 1)
        # Dataset_train0, _ = data_process2.makeData('data/yelp/filter.dev0.text', word_to_id,label = 0)
        Dataset_train = data_process.shuff(Dataset_train0, Dataset_train1, if_shuff=True)
        Dataset_train1_total = data_process.makeBatch(Dataset_train1, batch_size)
        Dataset_train0_total = data_process.makeBatch(Dataset_train0, batch_size)
        Dataset_train_total = data_process.makeBatch(Dataset_train, batch_size)


        Dataset_dev0, _ = data_process.makeData('data/yelp_6.27/filter.dev0.text', 'data/yelp_6.27/original.dev0.text', word_to_id, label=0)
        Dataset_dev1, _ = data_process.makeData('data/yelp_6.27/filter.dev1.text', 'data/yelp_6.27/original.dev1.text', word_to_id, label=1)
        # Dataset_dev0, _ = data_process.makeData('data/yelp/sentiment.dev0.text', 'data/yelp/sentiment.dev0.text',
        #                                          word_to_id, label=0)
        # Dataset_dev1, _ = data_process.makeData('data/yelp/sentiment.dev1.text', 'data/yelp/sentiment.dev1.text',
        #                                          word_to_id, label=1)
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


    if opt.if_gen == True:
        Dataset_test1,_ = data_process.makeData('data/yelp_6.24/filter_test1.txt', 'data/yelp_6.24/original_test1.text', word_to_id, label = 1)
        Dataset_test0, _ = data_process.makeData('data/yelp_6.24/filter_test0.txt', 'data/yelp_6.24/original_test0.text', word_to_id, label = 0)
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
         ], lr=config.generateor_learning_rate)



    # Parameters needed to be optimized in classifier.
    sentiment_vars = torch.optim.Adam(
        [
            {'params': model.sentiment_classifier.parameters()},
            {'params': model.sentiment_embedder.parameters()},
        ], lr=config.discriminator_lerning_rate)

    real_vars = torch.optim.Adam(
        [
            {'params': model.real_classifier.parameters()},
            {'params': model.real_embedder.parameters()}
        ], lr=config.discriminator_lerning_rate)


    criterion1 = nn.NLLLoss(size_average=False)
    # weight = torch.tensor([0.37, 0.63])
    criterion2 = nn.NLLLoss()
    criterion1 = criterion1.cuda()
    criterion2 = criterion2.cuda()

    print("Begining training.")
    # if we need to restore the model.
    if opt.checkpoint !='':
        model.load_state_dict(torch.load(opt.checkpoint))
        print('loading model...')
    file_name = opt.file_save
    lambda_n = 0
    lambda_g = 0
    gamma = 1.0

    trainModel(model, id_to_word, Dataset_train_total, Dataset_dev_total, Dataset_test_total, g_vars, sentiment_vars,
               real_vars, config, opt, len(Dataset_train_total), criterion1, criterion2, lambda_g, gamma, batch_size,
               max_length)
    print("End of training!")

if __name__ == "__main__":
    main()
