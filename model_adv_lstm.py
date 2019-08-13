import torch
from torch import nn
import torch.nn.functional as F
from BahdanauAttnDecoderRNN_adv import BahdanauAttnDecoderRNN
from CNNModels import CnnTextClassifier
class CtrlGenModel(nn.Module):
    def __init__(self,config,vocab_size,batch_size,weights_matrix):
        super(CtrlGenModel,self).__init__()
        #64*16(17)*100
        embed_size = config.model["embed_size"]
        hidden_size = config.model["rnn_lm"]["rnn_cell"]["num_units"]
        self.hidden_size = hidden_size
        num_layers = 2
        # self.softmax = F.log_softmax
        self.embedder = nn.Embedding(vocab_size, embed_size).cuda()
        self.embedder.load_state_dict({'weight': weights_matrix})
        self.sentiment_embedder = nn.Embedding(vocab_size,embed_size).cuda()
        self.real_embedder = nn.Embedding(vocab_size, embed_size).cuda()
        self.sentiment_embedder.load_state_dict({'weight': weights_matrix})
        self.real_embedder.load_state_dict({'weight': weights_matrix})
        self.vocab_size = vocab_size
        self.vocab_tensor = torch.LongTensor([i for i in range(vocab_size)]).cuda()
        self.batch_size = batch_size


        #The number layer can be two
        self.encoder = nn.GRU(input_size=embed_size,hidden_size=hidden_size,dropout=0.5,batch_first=True).cuda()
        self.dropout = nn.Dropout(0.5).cuda()
        self.dim_c = config.model["dim_c"]
        # self.label_connector = nn.Sequential(nn.Linear(1,hidden_size),nn.Linear(hidden_size,self.dim_c)).cuda()
        self.label_connector = nn.Linear(1, self.dim_c).cuda()
        self.connector = nn.Linear(900,hidden_size).cuda()
        self.decoder = BahdanauAttnDecoderRNN(hidden_size,embed_size,vocab_size,dropout_p=0.5).cuda()
        self.sentiment_classifier = CnnTextClassifier(num_filters=128, vocab_size=vocab_size, emb_size=embed_size, num_classes=2).cuda()
        self.lstm_connector2 = nn.Linear(hidden_size, 1).cuda()
        self.lstm_connector1 = nn.Linear(hidden_size, vocab_size).cuda()

        self.real_classifier = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, dropout=0.5,batch_first=True).cuda()

    def forward(self, inputs, sentence_length, if_dis=False, if_gen=False, pre=False, gamma=1):

        mask_index = inputs['mask_text_ids'].cuda()
        if if_dis:
            probs, classes = self.sentiment_classifier(self.sentiment_embedder(inputs["text_ids"].cuda()))
            # logits = self.sentiment_classifier(self.sentiment_embedder(inputs["text_ids"].cuda()))
            return probs, classes

        batch_size = len(inputs["text_ids"].cuda())
        # Change the vocab_tensor
        generator_embedding = self.embedder(self.vocab_tensor)  # [9380, 100]
        sentiment_embedding = self.sentiment_embedder(self.vocab_tensor)
        real_embedding = self. real_embedder(self.vocab_tensor)
        # vocab_tensor = self.vocab_tensor.expand(batch_size, self.vocab_size).cuda()

        # enc_inputs shape(64,17,100),enc_outputs shape(64,17,700),final_state shape(1,64,700)
        text_ids = inputs["text_ids"]
        ont_hot_text_label = text_ids.view(self.batch_size, sentence_length, 1).cpu()
        text_one_hot = torch.zeros(self.batch_size, sentence_length, self.vocab_size).scatter_(2, ont_hot_text_label, 1).cuda()
        one_hot_mask_label = mask_index.view(self.batch_size, sentence_length, 1).cpu()
        mask_text_onehot = torch.zeros(self.batch_size, sentence_length, self.vocab_size).scatter_(2, one_hot_mask_label, 1).cuda()
        mask_one_hot = torch.zeros(self.vocab_size).cuda()
        mask_one_hot[4] = 1

        # encode
        text_embeded = self.embedder(text_ids)
        enc_outputs, final_state = self.encoder(text_embeded)


        #Get the final_state
        z = final_state[0, :, :].cuda()
        labels = inputs["labels"].view(-1,1).float().cuda()
        c = self.label_connector(labels).cuda()
        h = torch.cat((c, z), 1).cuda()
        c_ = self.label_connector(1 - labels).cuda()
        h_ = torch.cat((c_, z), 1).cuda()


        if pre or if_gen:

            if pre:
                lstm_outputs, _ = self.real_classifier(self.real_embedder(text_ids[:, 1:]))
                lstm_output_logits = F.log_softmax(self.lstm_connector1(lstm_outputs))
                lstm_outputs_ids = torch.argmax(lstm_output_logits, 2)

                decoder_hidden = self.connector(h).unsqueeze(0)
            else:
                decoder_hidden = self.connector(h_).unsqueeze(0)

            decoder_outputs = torch.Tensor(sentence_length,batch_size,self.vocab_size).cuda()

            for i in range(sentence_length):
                if i == 0:
                    input = text_one_hot[:, 0, :]
                    decoder_output, decode_hidden = self.decoder(embedding=generator_embedding, inputs=input,
                                                                 encoder_outputs=enc_outputs, gumbel=False,
                                                                 initial_state=decoder_hidden)
                    inference_ids = torch.argmax(decoder_output, 1)
                    inference_word = self.embedder(inference_ids)
                else:

                    input = inference_word
                    decoder_output, decode_hidden = self.decoder(embedding=generator_embedding, inputs=input,
                                                                 encoder_outputs=enc_outputs, gumbel=False,
                                                                 initial_state=decode_hidden)
                    inference_ids = torch.argmax(decoder_output, 1)
                    inference_word = self.embedder(inference_ids)

                decoder_outputs[i] = decoder_output

            decoder_outputs_logits = decoder_outputs.transpose(0, 1)
            output_ids = torch.argmax(decoder_outputs_logits, 2)  # [64, 17, ids]

            if if_gen:
                # ids = torch.where(inputs['mask_text_ids'][:, 1:] == 4, output_ids[:, :-1], inputs['mask_text_ids'][:, 1:]) mask code5
                probs, classes = self.sentiment_classifier(self.sentiment_embedder(output_ids))    # mask code 7

                return output_ids, probs, classes       # mask code6

            return decoder_outputs_logits, output_ids, lstm_output_logits, lstm_outputs_ids


        # gumbel-softmax decode
        else:

            def gumbel_generate(decoder_gumbel_hidden):

                soft_outputs = torch.Tensor(sentence_length, batch_size, self.vocab_size).cuda()

                for i in range(sentence_length):
                    if i == 0:

                        input = text_one_hot[:, 0, :]
                        # decoder_soft_output [64, 9380]
                        decoder_soft_output, decoder_gumbel_hidden = self.decoder(embedding=generator_embedding, inputs=input,
                                                                                  encoder_outputs=enc_outputs, gumbel=True,
                                                                                  initial_state=decoder_gumbel_hidden,
                                                                                  gamma=gamma)

                        inference_soft_input = decoder_soft_output   # [64, 9380]


                    else:
                        # input = torch.where(mask_text_onehot[:, i, :] == mask_one_hot, inference_soft_input, mask_text_onehot[:, i, :]) mask code3
                        # input = inference_word

                        input = inference_soft_input

                        decoder_soft_output, decoder_gumbel_hidden = self.decoder(embedding=generator_embedding, inputs=input,
                                                                                  encoder_outputs=enc_outputs, gumbel=True,
                                                                                  initial_state=decoder_gumbel_hidden,
                                                                                  gamma=gamma)

                        inference_soft_input = decoder_soft_output

                    # if i==(sentence_length-1):  mask code4
                    #     soft_outputs[i] = inference_soft_input
                    # else:
                    #     soft_outputs[i] = torch.where(mask_text_onehot[:, i+1, :] == mask_one_hot, inference_soft_input, mask_text_onehot[:, i+1, :])
                    soft_outputs[i] = inference_soft_input
                soft_outputs_new = soft_outputs.transpose(0, 1)   # [64, 17, 9380]

                "sentiment classfier"
                # self.sentiment_embedder(vocab_tensor)  [64, 9380, 100]
                # sentiment_clas_input [64, 17, 100]
                sentiment_clas_input = torch.bmm(soft_outputs_new, sentiment_embedding.expand(batch_size, self.vocab_size, -1))
                probs, classes = self.sentiment_classifier(sentiment_clas_input)

                "real classifier"
                # fake sentence
                real_clas_input = torch.bmm(soft_outputs_new, real_embedding.expand(batch_size, self.vocab_size, -1))
                lstm_output_fake, _ = self.real_classifier(real_clas_input)
                output_added = torch.zeros(batch_size, self.hidden_size).cuda()
                for i in range(lstm_output_fake.size(1)):
                    output_added += lstm_output_fake[:, i, :]
                logits_fake = self.lstm_connector2(output_added)

                # real sentence
                lstm_output_real, _ = self.real_classifier(self.real_embedder(inputs['text_ids'].cuda()))
                output_added = torch.zeros(batch_size, self.hidden_size).cuda()
                for i in range(lstm_output_real.size(1)):
                    output_added += lstm_output_fake[:, i, :]
                logits_real = self.lstm_connector2(output_added)

                return probs, classes, logits_real, logits_fake


            ###############################################################


            "generate original sentiment"
            probs_original, classes_original, logits_real_original, logits_fake_original = gumbel_generate(
                self.connector(h).unsqueeze(0))

            "generate trans sentiment"
            probs_trans, classes_trans, logits_real_trans, logits_fake_trans = gumbel_generate(
                self.connector(h_).unsqueeze(0))

            soft_output_original = (probs_original, classes_original, logits_real_original, logits_fake_original)
            soft_output_trans = (probs_trans, classes_trans, logits_real_trans, logits_fake_trans)

            return soft_output_original, soft_output_trans