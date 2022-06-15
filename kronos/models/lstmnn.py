import time
import copy
import torch as t, torch

import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_sequence

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
 
from kronos.lib.metrics import classification_scores 
from kronos.lib.helpers import time_me


class LSTMnn(nn.Module):
    def __init__(self, lstm_params, output_size):
        super(LSTMnn, self).__init__()
        
        self.params = {k:lstm_params[k] for k in lstm_params}

        # batch_first=True expects and returns in- and output as (batch_size,
        # no_of_sequences, size_of_sequence).  IF the input is of type
        # PackedSequence the flag doesn't matter.
        self.lstm = nn.LSTM( **lstm_params, batch_first=True).double()  
        self.fc_final = nn.Linear(lstm_params['hidden_size'], output_size) 
    
        self.bnorm = nn.BatchNorm1d(lstm_params['hidden_size'])

    def forward(self, x_sequence):
        
        if isinstance(x_sequence, torch.nn.utils.rnn.PackedSequence):
            # `output` will also be of type PackedSequence
            # `hn` holds the final hidden state of all lstm layers with the
            # shape (num of layers, batch size, hidden_state size)
            output, (hn, cn) = self.lstm( x_sequence )
            #print(type(output), type(hn), hn.shape )
            
            # We use the last hidden state of the last layer
            # with the shape (batch, hidden_state)
            final_hiddenstate = hn[-1] 
            bnormed = self.bnorm(final_hiddenstate.float())
            final = self.fc_final( bnormed )
            return final 


        # If Tensor - (Batch Size x Sequence Size x Input Size)
        else:
            # reset hidden states (ltsm memory) for every batch elements
            batch_size = x_sequence.shape[0]
            h0 = torch.zeros(self.params['num_layers'], batch_size, self.params['hidden_size'] ).double()
            c0 = torch.zeros(self.params['num_layers'], batch_size, self.params['hidden_size'] ).double()
            
            # `output` holds all the hidden states (#seq) for each input
            # of the last ltsm layer. Wrt. to the shape see comments above 
            # regarding `batch_first`.
            # `hn` holds the final hidden state of all lstm layers with the
            # shape (num of layers, batch size, hidden_state size)
            output, (hn, cn) = self.lstm( x_sequence, (h0, c0) )

            # We use the last hidden state of the last layer
            # with the shape (batch, hidden_state)
            final_hiddenstate = hn[-1] 
            final = self.fc_final( final_hiddenstate.float() )
            return final 


def collate_with_packed_sequence(batch):
    """ Prepare batch for DataLoader with PackedSequence."""

    # Sort by the sequence size (number of items in a sequence) desc.
    sorted_batch = sorted(batch, key=lambda x: len(x['X']), reverse=True) 

    return {
        'X': pack_sequence([ torch.DoubleTensor(item['X']) for item in sorted_batch]),
        'Y': torch.LongTensor( [ item['Y'] for item in sorted_batch ] )
        }



class TrainingStop():
    def __init__(self, delta, patience):
        self.best_score = None
        self.patience = patience # number of iterations/epochs in a row the
                                 # best_score doesn't improve at least by delta
        self.delta = delta       # minimum expected improvement from best_score
        self.violations = 0

        self.best_model = None       # copy of the model with the best score for inference
        self.best_model_epoch = None

    def keep_going(self, score, model, epoch):
        if not self.best_score:
            self.best_score = score
            self.best_model = copy.deepcopy(model)
            self.best_model_epoch = epoch
            return True
        elif score > self.best_score + self.delta:
            self.best_score = score
            self.best_model = copy.deepcopy(model)
            self.best_model_epoch = epoch
            self.violations = 0 
            return True
        else:
            self.violations += 1
            if self.violations > self.patience:
                return False
            return True


def evaluate_classifier(model, dataset, device, batch_size, loss_fn=None, collate_fn=None):
    # lists of class indices
    y_gt = [] 
    y_pd = []

    accum_loss = 0
    dataloader = t.utils.data.DataLoader(dataset, batch_size, collate_fn=collate_fn)
    model.eval()
    with t.no_grad():
        for batch in dataloader:
            X = batch['X'].to(device) 
            Y = batch['Y'].to(device)
            pred  = model(X)

            #print( type(pred.argmax(dim=1).tolist()), type(Y.tolist()) )
            y_gt += pred.argmax(dim=1).tolist()
            y_pd += Y.tolist()

            if loss_fn:
                loss = loss_fn(pred, Y) 
                accum_loss += loss.item()

        stats = { "y_true": y_gt, "y_pred": y_pd }
        if loss_fn:
            stats.update({"avg_loss":  accum_loss / len(dataloader)}) 

        return stats





@time_me
def training_loop(log_dir, train_dataset, validation_dataset, input_size, output_size, dry_run=False):
    with SummaryWriter(log_dir=log_dir) as writer:

        dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, drop_last=True, 
                collate_fn=collate_with_packed_sequence)

         
        lstm_params = {
            "input_size"  : input_size, # of the input sequences 
            "hidden_size" : 16, # cells per lstm
            "num_layers"  : 1, # how lstm layers
            }
        model = LSTMnn(lstm_params, output_size=output_size)

        if dry_run:
            print("[DRY RUN Model Training]")
            return model


        step_size= 1e-3
        eps = 1e-8
        
        optimizer = optim.Adam(model.parameters(), lr=step_size, eps=eps)
        loss_fn = nn.CrossEntropyLoss()

        training_stop = TrainingStop(delta=0.01, patience=5)
        epochs = 16
        for epoch in range(epochs):
            accum_loss = 0
            model.train()

            for batch in dataloader:
                optimizer.zero_grad()

                output = model(batch['X'])

                loss = loss_fn(output, batch['Y'])
                loss.backward()
                optimizer.step()
                accum_loss += loss.item()

            train_loss  = accum_loss / len(dataloader) # len returns number of batches

            # NOTE: If true, early training stoppage is ignored.
            if not validation_dataset: 
                print(f"[epoch {epoch:3}][train_loss:{train_loss:.5f}]")
                continue

            stats = evaluate_classifier(model, 
                                        validation_dataset, 
                                        device=t.device('cpu'), 
                                        batch_size=64, 
                                        loss_fn=loss_fn, 
                                        collate_fn=collate_with_packed_sequence)

            # Training and evaluation progress per epoch
            valid_loss = stats.get('avg_loss', 0)

            scores = classification_scores(y_true=stats['y_true'], y_pred=stats['y_pred'])
            print(f"[epoch {epoch:3}][train_loss:{train_loss:.5f} valid_loss:{valid_loss:.5f}]",
                  f"[accu:{scores['accuracy']:.4f} mcc:{scores['mcc']:.4f}",
                  f"f1-macro:{scores['classification_report']['macro avg']['f1-score']:.4f}",
                  f"f1-weighted:{scores['classification_report']['weighted avg']['f1-score']:.4f}",
                  f"baccu:{scores['balanced_accuracy']:.4f}]",
                  )
            #writer.add_scalar("loss/train", train_loss, epoch)
            #writer.add_scalar("loss/valid", valid_loss, epoch)
            #writer.add_scalar("accu/valid", scores['accuracy'], epoch)
            #writer.flush()
            
            if not training_stop.keep_going(score=scores['accuracy'],model=model,epoch=epoch):
                print(f"**Early training stop @ [ epoch:{training_stop.best_model_epoch:2} accu:{training_stop.best_score:.4f} ]")
                return training_stop.best_model

    return model
