import torch
from torch import nn
import torch.nn.functional as F
from BahdanauAttnDecoderRNN import BahdanauAttnDecoderRNN
from CNNModels import CnnTextClassifier
class CtrlGenModel(nn.Module):
    def __init__(self,config,vocab_size,batch_size,weights_matrix):
        super(CtrlGenModel,self).__init__()
        #64*16(17)*100
        embed_size = config.model["embed_size"]
        hidden_size = config.model["rnn_lm"]["rnn_cell"]["num_units"]
        self.hidden_size = hidden_size
        num_layers = 2
        self.softmax = F.log_softmax
        self.embedder = nn.Embedding(vocab_size, embed_size).cuda()
        self.embedder.load_state_dict({'weight': weights_matrix})
        self.clas_embedder = nn.Embedding(vocab_size,embed_size).cuda()
        self.clas_embedder.load_state_dict({'weight': weights_matrix})
        self.vocab_size = vocab_size
        self.vocab_tensor = torch.LongTensor([i for i in range(vocab_size)]).cuda()
        self.batch_size = batch_size


        #The number layer can be two
        self.encoder = nn.GRU(input_size = embed_size,hidden_size = hidden_size,dropout = 0.5,batch_first = True).cuda()
        self.dropout = nn.Dropout(0.5).cuda()
        self.dim_c = config.model["dim_c"]
        self.label_connector = nn.Sequential(nn.Linear(1,hidden_size),nn.Linear(hidden_size,self.dim_c)).cuda()
        self.connector = nn.Linear(700,hidden_size).cuda()
        self.decoder = BahdanauAttnDecoderRNN(hidden_size,embed_size,vocab_size,dropout_p=0.5).cuda()
        self.classifier = CnnTextClassifier(num_filters = 128,vocab_size = vocab_size,emb_size = embed_size,num_classes = 2).cuda()

    def forward(self, inputs, sentence_length, if_dis=False, if_eval=False, if_gen=False, pre=False, gamma=1):

        mask_index = inputs['mask_text_ids'].cuda()
        if if_dis:
            probs, classes = self.classifier(self.clas_embedder(inputs["text_ids"].cuda()))
            return probs, classes
        
        
        input_length = len(inputs["text_ids"].cuda())
        # Change the vocab_tensor
        vocab_tensor = self.vocab_tensor.expand(input_length,self.vocab_size).cuda()
        enc_text_ids = inputs["text_ids"].cuda()
        #enc_inputs shape(64,16,100)
        #enc_outputs shape(64,16,700)
        #final_state shape(1,64,700)


        text_embedding = self.embedder(mask_index)
        # text_embedding = self.embedder(enc_text_ids)
        enc_outputs,final_state = self.encoder(text_embedding)
      
        #Get the final_state
        z = final_state[0,:,self.dim_c:].cuda()
        labels = inputs["labels"].view(-1,1).float().cuda()
        c = self.label_connector(labels).cuda()
        c_ = self.label_connector(1-labels).cuda()
        h = torch.cat((c,z),1).cuda()
        h_ = torch.cat((c_,z),1).cuda()


        decoder_outputs = torch.Tensor(sentence_length,input_length,self.vocab_size).cuda()
        if pre:
            decoder_hidden = self.connector(h).unsqueeze(0)
        else:
            decoder_hidden = self.connector(h_).unsqueeze(0)
        # decoder_hidden = self.connector(h).unsqueeze(0)
        for i in range(sentence_length):

            if i==0:
                input = mask_index[:, i]
                decoder_output, decode_hidden = self.decoder(embedding=self.embedder, word_input=input, encoder_outputs=enc_outputs, gumbel=False,
                                                       initial_state=decoder_hidden)
                inference_word = torch.argmax(decoder_output, 1)
            else:
                if pre==True:
                    input = enc_text_ids[:, i]
                else:
                    input = torch.where(mask_index[:, i] == 4, inference_word, mask_index[:, i])

                decoder_output, decode_hidden = self.decoder(embedding=self.embedder, word_input=input, encoder_outputs=enc_outputs, gumbel=False,
                                                       initial_state=decode_hidden)
                inference_word = torch.argmax(decoder_output, 1)

            decoder_outputs[i] = decoder_output
        decoder_outputs_new = decoder_outputs.transpose(0, 1)

        inp = torch.argmax(decoder_outputs_new, 2)
        ids = torch.where(inputs['mask_text_ids'][:, 1:] == 4, inp[:, :-1], inputs['text_ids'][:, 1:])
        probs, classes = self.classifier(self.clas_embedder(ids))

        
        #soft_output.sample id called soft_outputs 64 16 9657
        if if_eval:
            decoder_gumbel_hidden = self.connector(h_).unsqueeze(0)
            soft_outputs_ = torch.Tensor(sentence_length,input_length,self.vocab_size).cuda()

            decoder_soft_outputs,decoder_gumbel_hidden = self.decoder(embedding=self.embedder,word_input=inputs['text_ids'][:, 0].cuda(), initial_state=decoder_gumbel_hidden, encoder_outputs=enc_outputs, gumbel=True, gamma = gamma)
            soft_outputs_[0] = decoder_soft_outputs
            for di in range(1,sentence_length):
                decoder_soft_outputs,decoder_gumbel_hidden = self.decoder(embedding = self.embedder,word_input=torch.argmax(decoder_soft_outputs, 1), initial_state=decoder_gumbel_hidden, encoder_outputs=enc_outputs, gumbel=True,gamma = gamma)
                soft_outputs_[di] = decoder_soft_outputs

            clas_input = torch.bmm(soft_outputs_.transpose(0,1),self.clas_embedder(vocab_tensor))
            probs,classes = self.classifier(clas_input)
            return probs, classes
        else:
            if pre:
                decoder_gumbel_hidden = self.connector(h).unsqueeze(0)
            else:
                decoder_gumbel_hidden = self.connector(h_).unsqueeze(0)
            soft_outputs_ = torch.Tensor(sentence_length,input_length,self.vocab_size).cuda()


            for i in range(sentence_length):
                if i == 0:
                    input = mask_index[:, i]
                    decoder_soft_output, decoder_gumbel_hidden = self.decoder(embedding=self.embedder, word_input=input,
                                                                           encoder_outputs=enc_outputs, gumbel=True,
                                                                           initial_state=decoder_gumbel_hidden, gamma=gamma)
                    inference_soft_word = torch.argmax(decoder_output, 1)
                else:
                    if pre:
                        input = enc_text_ids[:, i]
                    else:
                        input = torch.where(mask_index[:, i] == 4, inference_soft_word, mask_index[:, i])

                    decoder_soft_output, decoder_gumbel_hidden = self.decoder(embedding=self.embedder, word_input=input,
                                                                           encoder_outputs=enc_outputs, gumbel=True,
                                                                           initial_state=decoder_gumbel_hidden,
                                                                           gamma=gamma)
                    inference_soft_word = torch.argmax(decoder_soft_output, 1)

                soft_outputs_[i] = decoder_soft_output

            soft_outputs_new = soft_outputs_.transpose(0, 1)


            inp = torch.argmax(soft_outputs_new[:, :-1, :], 2)
            sentence = torch.where(inputs['mask_text_ids'][:, 1:] == 4, inp, inputs['text_ids'][:, 1:])
            probs_trans, classes_trans = self.classifier(self.clas_embedder(sentence))

            return decoder_outputs_new, soft_outputs_new, probs, classes, probs_trans, classes_trans