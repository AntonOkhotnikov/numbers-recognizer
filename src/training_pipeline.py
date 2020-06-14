#!/usr/bin/env python3
import os
import math
import logging
import argparse

import torch
from torch.optim import AdamW

from model import NumbersRecognizer, DeepSpeechClone
from dataloader import get_train_dev_loaders
from data_utils import TextTransformer
from metrics import CERmetrics
from decoder import GreedyDecoder
from writer import CustomWriter


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train(model, train_loader, criterion, optimizer, text_transformer, blank_id, writer, epoch, logger):
    # monitor training loss
    train_loss = 0.0

    # prepare weighs and gradients for training
    model.train()
    
    for inputs, labels in train_loader:
        # print(inputs.shape)
        # here goes one mini-batch processing
        inputs = inputs.to(device)
        # convet labels to integers and stack in Tensor (with padding)
        labels = [torch.tensor(text_transformer.txt2int(label)) for label in labels]
        # pad long sequences with a blank symbol
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=blank_id).to(device)
        
        # all inputs are padded to the same length, so next is valid
        inputs_lens = [59] * inputs.shape[0]
        labels_lens = [len(label) for label in labels]


        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(inputs)  # (batch, time, n_class)
        output = output.transpose(0, 1) # (time, batch, n_class)

        # calculate the loss
        loss = criterion(output, labels, inputs_lens, labels_lens)
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()

        # average loss across minibatch
        # loss = loss / inputs.size(0)

        # training loss
        if loss.item() > 1e8 or math.isnan(loss.item()):
            # logger.error("Loss exploded to %.02f at step %d!" % (loss, step))
            raise Exception("Loss exploded")

        # update running training loss
        # train_loss += loss.item() * inputs.size(0)
        train_loss += loss.item()

    # write loss to tensorboard
    writer.log_training(train_loss, epoch + 1)
    logger.info(f"Wrote summary at epoch {epoch + 1}")

    # save checkpoint
    if (epoch + 1) % 10 == 0:
        os.makedirs('exps/baseline_enrich/chkpt', exist_ok=True)
        torch.save({'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch + 1},
                    f'exps/baseline_enrich/chkpt/chkpt_{epoch + 1}.pt')

    return train_loss


def test(model, val_loader, criterion, text_transformer, blank_id, cer_computer, writer, epoch):

    test_loss = 0.0

    model.eval()

    cer = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            # here goes one mini-batch processing
            inputs = inputs.to(device)
            # convert labels to integers and stack in Tensor (with padding)
            labels = [torch.tensor(text_transformer.txt2int(label)) for label in labels]
            # pad long sequences with a blank symbol
            labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=blank_id).to(device)
            # all inputs are padded to the same length, so next is valid
            inputs_lens = [59] * inputs.shape[0]
            labels_lens = [len(label) for label in labels]

            outputs = model(inputs)
            outputs = outputs.transpose(0, 1) # (time, batch, n_class)

            loss = criterion(outputs, labels, inputs_lens, labels_lens)

            # average loss across minibatch
            # loss = loss / inputs.size(0)
            test_loss += loss.item()

            prediction, reference = GreedyDecoder(outputs, labels, labels_lens, blank_id, text_transformer)

            # get batch CER
            cer.append(cer_computer.calculate_cer(prediction, reference))

        mean_cer = sum(cer) / len(cer)
        writer.log_evaluation(test_loss, mean_cer, epoch + 1)
    
    return test_loss, mean_cer


def main():

    loader_params = {'batch_size': 1024,
                     'shuffle': False,
                     'num_workers': 1 if torch.cuda.is_available() else 6,
                     'pin_memory': False}

    root_path='/home/anton/work/projects/myna/data/numbers'

    train_loader, val_loader = get_train_dev_loaders(root_path, loader_params, train_ratio=0.9, max_len=3.9, enrich_target=True)

    # 0. Training parameters
    n_epochs = 15

    # 1. NN hyperparams
    enrich_target = True  # this flag adds symbol '*' ~ 'тысяча' if a number is longer than 3 digits
    num_classes = 10
    net_params = {'n_cnn_layers': 3, 
                  'n_rnn_layers': 3,
                  'rnn_dim': 128, 
                  'n_class': num_classes + 1 if not enrich_target else num_classes + 2,  # add output for blank symbol
                  'n_feats': 64, 
                  'stride': (2, 1),
                  'dropout': 0.1}
    # model = NumbersRecognizer(num_classes)
    model = DeepSpeechClone(**net_params).to(device)
    print('Number of model params', sum([param.nelement() for param in model.parameters()]))

    # set optimizer
    optimizer = AdamW(model.parameters(), lr=0.001)

    # define blank_id
    blank_id = num_classes if not enrich_target else num_classes + 1
    criterion = torch.nn.CTCLoss(blank=blank_id).to(device)

    # postprocess targets and so on
    text_transformer = TextTransformer(enrich_target=enrich_target, length=num_classes)

    # CER metrics in train/test
    cer_computer = CERmetrics(blank_id, text_transformer, enrich_target=enrich_target)

    # Tensorboard
    writer = CustomWriter('exps/baseline_enrich/logs')

    # Logger
    logger = logging.getLogger()

    for epoch in range(n_epochs):
        train_loss = train(model, train_loader, criterion, optimizer, text_transformer, blank_id, writer, epoch, logger)
        print(f'Epoch: {epoch + 1}\nTraining Loss: {train_loss :.4f}')

        test_loss, test_cer = test(model, val_loader, criterion, text_transformer, blank_id, cer_computer, writer, epoch)
        print(f'Testing loss: {test_loss :.4f}\nTesting CER: {test_cer * 100 :.2f}\n')
    
    writer.close()


if __name__ == "__main__":
    main()
