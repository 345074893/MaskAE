import torch
import torch.nn as nn
import torch.nn.functional as F


class CnnTextClassifier(nn.Module):
    def __init__(self, vocab_size, emb_size, num_filters, num_classes, window_sizes=(3, 4, 5)):
        super(CnnTextClassifier, self).__init__()

        self.num_class = num_classes
        if isinstance(window_sizes, tuple):
            self.convs = nn.ModuleList([
                nn.Conv2d(1, num_filters, [window_size, emb_size], padding=(window_size - 1, 0))
                for window_size in window_sizes
            ])
            self.fc = nn.Linear(num_filters * len(window_sizes), self.num_class)
            self.num = True
        else:
            self.convs = nn.Conv2d(1, num_filters, (window_sizes, emb_size), padding=(window_sizes - 1, 0))
            self.fc = nn.Linear(num_filters, self.num_class)
            self.num = False

        self.clas_embedder = nn.Embedding(vocab_size, 100).cuda()


        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        x = torch.unsqueeze(x, 1)       
        xs = []
        if self.num == True:
            for conv in self.convs:
                x2 = F.relu(conv(x))
                x2 = torch.squeeze(x2, -1)
                x2 = F.max_pool1d(x2, x2.size(2))
                xs.append(x2)
            x = torch.cat(xs, 2)
        else:
            x2 = F.relu(self.convs(x))
            x2 = torch.squeeze(x2, -1)
            x = F.max_pool1d(x2, x2.size(2))

        x = x.view(x.size(0), -1)       
        logits = self.fc(x)             

        if self.num_class == 1:

            # classes = torch.where(logits > 0, torch.tensor(int(1)).cuda(), torch.tensor(int(0)).cuda())
            return logits

        else:
            probs = F.log_softmax(logits)
            classes = torch.max(probs, 1)[1]

            return probs, classes