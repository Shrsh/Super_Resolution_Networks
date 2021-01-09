import pytorch_lightning as pl
from argparse import ArgumentParser
import os
import numpy as np
import torch
import random
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoConfig
import transformers
from transformers.optimization import get_linear_schedule_with_warmup
import torch.nn.functional as F
from pytorch_lightning.callbacks import ModelCheckpoint
from dataloader import get_dataset_loaders
from pytorch_lightning.metrics import Metric
import gc
from sklearn.metrics import f1_score, accuracy_score, classification_report, precision_recall_fscore_support
import json
from pytorch_lightning import loggers as pl_loggers
from logger import MyLogger, LOG_LEVELS
from models import get_module


base_dir = os.path.dirname(os.path.realpath(__file__))

#allow deterministic psuedo-random-initialization
def random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class PairWiseSentenceRanking(nn.Module):
    """Head for pairwise sentence ranking task"""

    def __init__(self, hidden_size, hidden_dropout_prob=0.1):
        super(PairWiseSentenceRanking, self).__init__()
        #setence 1 transformation
        self.phi_a = nn.Linear(hidden_size, 1)
        self.phi_b = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(hidden_dropout_prob)
    
    def forward(self, sent_a, sent_b):
        return self.phi_a(self.dropout(sent_a)), self.phi_b(self.dropout(sent_b))

class SetenceOrdering(pl.LightningModule):
    def __init__(self, args):
        super(SetenceOrdering, self).__init__()
        self.config_args = args
        #load pretrained hierarchical model
        self.doc_encoder = get_module(args.model_type)(args)
        self.sentence_ranker = PairWiseSentenceRanking(self.doc_encoder.tf2.config.hidden_size, hidden_dropout_prob=args.dropout_rate)
        self.train_accuracy = pl.metrics.Accuracy()
        self.val_accuracy = pl.metrics.Accuracy()
        self.test_accuracy = pl.metrics.Accuracy()
    
    def forward(self, doc_a, doc_b, attention_mask_a=None, attention_mask_b=None,token_type_ids=None):
        #get the hidden representation for the last layer of longformer
        coherence_a = self.doc_encoder(doc_a, attention_mask=attention_mask_a, token_type_ids=token_type_ids)
        coherence_b = self.doc_encoder(doc_b, attention_mask=attention_mask_b, token_type_ids=token_type_ids)
        return self.sentence_ranker(coherence_a, coherence_b)

    def configure_optimizers(self):
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': self.config_args.weight_decay},
            {'params': [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]

        optimizer = transformers.AdamW(optimizer_grouped_parameters, lr=self.config_args.learning_rate, eps=1e-8)
        # optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=self.config_args.learning_rate, eps=1e-6)
        
        #wsj dataset count
        total_dataset_count = 4560
        total_steps = int(np.ceil((self.config_args.epochs * total_dataset_count)/(self.config_args.batch_size*self.config_args.gpus)))
        
        scheduler = {
            'scheduler': get_linear_schedule_with_warmup(optimizer, self.config_args.warmup_steps*total_steps, total_steps),
            'interval': 'step',
        }

        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        doc_a, doc_b, doc_a_mask, doc_b_mask, label_ids = batch
        #sentence score for 'sentence a' and 'sentence b' are 'phi_a' and 'phi_b' respectively
        phi_a, phi_b = self(doc_a, doc_b, attention_mask_a=doc_a_mask, attention_mask_b=doc_b_mask)
        loss = F.margin_ranking_loss(phi_a, phi_b, label_ids, margin=self.config_args.margin)
        
        pred_res = -1*torch.ones_like(label_ids)
        pred_res[phi_a.detach().requires_grad_(False).view(-1) > phi_b.detach().requires_grad_(False).view(-1)] = 1

        #get training batch accuracy       
        acc = self.train_accuracy(pred_res.long(), label_ids.long())
        pbar = {'batch_train_acc': acc}
        return {'loss': loss, 'progress_bar': pbar}

    def training_epoch_end(self, train_step_outputs):
        avg_train_loss = torch.tensor([x['loss'] for x in train_step_outputs]).mean()
        
        overall_train_acc = self.train_accuracy.compute()
        # pbar = { 'overall_val_acc': overall_val_acc}

        # self.logger.log_metrics({'avg_train_loss': avg_train_loss, 'train_acc': overall_train_acc})
        self.config_args.logger.info('epoch : %d - average train loss : %f, overall_train_acc : %f' % (self.current_epoch, avg_train_loss.item(), overall_train_acc))

        self.log('avg_train_loss', avg_train_loss, prog_bar=True)
        self.log('overall_train_acc', overall_train_acc, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        doc_a, doc_b, doc_a_mask, doc_b_mask, label_ids = batch
        #sentence score for 'sentence a' and 'sentence b' are 'phi_a' and 'phi_b' respectively
        phi_a, phi_b = self(doc_a, doc_b, attention_mask_a= doc_a_mask, attention_mask_b= doc_b_mask)
        loss = F.margin_ranking_loss(phi_a, phi_b, label_ids, margin=self.config_args.margin)
        
        pred_res = -1*torch.ones_like(label_ids)
        pred_res[(phi_a > phi_b).view(-1)] = 1

        #get validation batch accuracy
        acc = self.val_accuracy(pred_res.long(), label_ids.long())
        pbar = {'val_acc': acc}
        return {'val_loss': loss, 'progress_bar': pbar}
    
    def validation_epoch_end(self, val_step_outputs):
        avg_val_loss = torch.tensor([x['val_loss'] for x in val_step_outputs]).mean()
        overall_val_acc = self.val_accuracy.compute()
        
        # self.logger.log_metrics({'avg_val_loss': avg_val_loss, 'val_acc': overall_val_acc})
        self.config_args.logger.info('epoch : %d - average val loss : %f, overall_val_acc : %f' % (self.current_epoch, avg_val_loss.item(), overall_val_acc))

        self.log('avg_val_loss', avg_val_loss, prog_bar=True)
        self.log('overall_val_acc', overall_val_acc, prog_bar=True)
    
    def test_step(self, batch, batch_idx):
        doc_a, doc_b, doc_a_mask, doc_b_mask, label_ids = batch
        #sentence score for 'sentence a' and 'sentence b' are 'phi_a' and 'phi_b' respectively
        phi_a, phi_b = self(doc_a, doc_b, attention_mask_a= doc_a_mask, attention_mask_b= doc_b_mask)
        loss = F.margin_ranking_loss(phi_a, phi_b, label_ids, margin=self.config_args.margin)
        
        pred_res = -1*torch.ones_like(label_ids)
        pred_res[(phi_a > phi_b).view(-1)] = 1

        #using testing with validation metric (as train and test runs are different)
        acc = self.test_accuracy(pred_res.long(), label_ids.long())
        pbar = {'test_acc': acc}
        return {'test_loss': loss, 'progress_bar': pbar}

    def test_epoch_end(self, test_step_outputs):
        avg_test_loss = torch.tensor([x['test_loss'] for x in test_step_outputs]).mean()
        overall_test_acc = self.test_accuracy.compute()
        self.config_args.logger.info('average test loss : %f, overall_test_acc : %f' % (avg_test_loss.item(), overall_test_acc))

        self.log('average test loss', avg_test_loss.item())
        self.log('overall_test_acc', self.test_accuracy.compute()) 

class TextDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
    
    def val_dataloader(self):
        dev_file_path = os.path.join(os.path.abspath(self.args.dataset_path), 'dev.jsonl')
        val_dataset =  get_dataset_loaders(dev_file_path, batch_size=self.args.batch_size)
        return val_dataset

    def train_dataloader(self):
        train_file_path = os.path.join(os.path.abspath(self.args.dataset_path), 'train.jsonl')
        train_dataset =  get_dataset_loaders(train_file_path, batch_size=self.args.batch_size)
        return train_dataset

def start_training(args):
    model_name = args.logger_exp_name

    args.logger.debug('initiating training process...')
    
    final_checkpoint_path = os.path.join(args.checkpoint_path, model_name)  
    os.makedirs(final_checkpoint_path, exist_ok=True)

    # Load datasets
    dm = TextDataModule(args)

    call_back_parameters = {
        'filepath': final_checkpoint_path,
        'save_top_k': 1,
        'verbose': True,
        'monitor': 'overall_val_acc',
        'mode': 'max',
        'prefix': 'hierarchical',
    }

    # checkpoint callback to used by the Trainer
    checkpoint_callback = ModelCheckpoint(**call_back_parameters)

    model = SetenceOrdering(args)
    
    args.logger.debug(model)
    args.logger.info('Model has %d trainable parameters' % count_parameters(model))

    callback_list = []
    
    trainer = pl.Trainer(callbacks=callback_list, max_epochs=args.epochs, min_epochs=1, gradient_clip_val=args.clip_grad_norm, 
                        gpus=args.gpus, checkpoint_callback=checkpoint_callback, distributed_backend='ddp')
    #finally train the model
    args.logger.debug('about to start training loop...')
    trainer.fit(model, dm)
    args.logger.debug('training done.')

def test_dataloader(args, sub_dataset_name):
    test_file_path = os.path.join(os.path.abspath(args.dataset_path), '%s_test.jsonl'%sub_dataset_name)
    test_dataset =  get_dataset_loaders(test_file_path, batch_size=args.batch_size)
    return test_dataset

def init_testing(args):
    
    args.logger.debug('initiating inference process...')
    model = SetenceOrdering(args)

    #load trained model
    args.logger.debug('loading the model from checkpoint : %s' % args.checkpoint_path)
    trained_model = model.load_from_checkpoint(args.checkpoint_path, args=args)
    args.logger.debug('loaded model successfully !!!')

    for sub_dataset in ['Clinton', 'Enron', 'Yahoo', 'Yelp']:
        # Load datasets
        test_loader = test_dataloader(args, sub_dataset)
        
        # invoke callbacks if required
        callback_list = []
        args.logger.info('testing on dataset : %s' % sub_dataset)
        trainer = pl.Trainer(callbacks=callback_list, gpus=args.gpus, distributed_backend='ddp')
        trainer.test(trained_model, test_dataloaders=test_loader)

    args.logger.debug('testing done !!!')

if __name__ == "__main__":

    parser = ArgumentParser()

    default_checkpoint_path = os.path.join(base_dir, 'lightning_checkpoints')
    default_dataset_path = os.path.join(base_dir, 'dataset')

    #Global model configuration
    parser.add_argument("--dataset_path", type=str,
                        help="directory containing sentence order dataset", default=default_dataset_path)
    parser.add_argument('--checkpoint_path', default=default_checkpoint_path, type=str,
                        help='directory where checkpoints are stored')
    parser.add_argument('--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('--epochs', default=10, type=int,
                        help='number of total epochs to run')
    parser.add_argument('--batch_size', default=2, type=int,
                        help='adjust batch size per gpu')
    parser.add_argument('--learning_rate', default=1e-6, type=float, 
                        help='specify the learning rate')
    parser.add_argument('--clip_grad_norm', default=0.0, type=float, 
                        help='clip gradients with norm above specified value, 0 value will disable it.')
    parser.add_argument('--weight_decay', default=0.01, type=float, 
                        help='specify the weight decay.')
    parser.add_argument('--dropout_rate', default=0.1, type=float, 
                        help='specify the dropout rate for all layer, also applies to all transformer layers.')
    # parser.add_argument('--seed', default=42, type=int,
                        # help='seed value for random initialization.')
    parser.add_argument("--warmup_steps", default=0.0, type=float,
                        help="percentage of total step used as tlinear warmup while training the model.")
    parser.add_argument("--margin", default=1.0, type=float,
                        help="margin to use in pairwise sentence ranking loss.")
    parser.add_argument("--train_dataset_count", default=25000, type=int,
                        help="size of training data used for learning rate scheduler.")
    # inference
    parser.add_argument('-i', '--inference', action='store_true',
                        help='enable inference over the datasets')
    #logger configs
    parser.add_argument('--logger_exp_name', type=str, required=True, 
                        help='experiment name that will be used to store checkpoints.')
    #transformer config
    parser.add_argument('--model_type', type=str, default='hierarchical', choices = ['hierarchical', 'vanilla'],
                        help='specify type of architecture')
    parser.add_argument('--model_name', type=str, default='roberta-base',
                        help='specify pretrained transformer model to use.')
    parser.add_argument('--use_pretrained_tf2', action='store_true',
                        help='load pretrained transformer model for TF2 layer')
    parser.add_argument('--sentence_pooling', type=str, default='none', choices = ['sum', 'mean', 'max', 'min', 'attention', 'none'],
                        help='specify the pooling strategy to use at lower transformer i.e. TF1 layer')
    args  = parser.parse_args()
    
    overwrite_flag = False if args.inference else True
    #configure and add logger to arguments
    args.logger = MyLogger('', os.path.join(base_dir, "%s.log"%args.logger_exp_name), 
                            use_stdout=False, overwrite=overwrite_flag, log_level=LOG_LEVELS.DEBUG)

    #get the arguments passed to this program
    if not args.inference:
        args.logger.info('\ncommand line argument captured ..')
        args.logger.info('--'*30)
        
        for arg in vars(args):
            value = getattr(args, arg)
            args.logger.info('%s - %s' % (arg, value))
        args.logger.info('--'*30)
    
    # random_seed(args.seed)

    if args.inference:
        init_testing(args)
    else:
        start_training(args)
    
