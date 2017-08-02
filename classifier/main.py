#!/usr/bin/env python3
from __future__ import division, absolute_import, print_function

import os
import time
import argparse
import multiprocessing
import json
import random

import numpy as np
import pandas as pd
from utils.misc import AverageMeter, create_dir_if_not_exists

from model import CervixClassificationModel
import dataset

import torch
import torchvision

TYPE_MAP = {
    1 : 0,
    2 : 1,
    3 : 2
}

NUM_CLASSES = len(set(TYPE_MAP.values()))

def train_single_epoch(model, criterion, optimizer, train_loader, epoch, is_cuda):
    model.train() # switch to train mode
    avg_loss = AverageMeter()
    end = time.time()
    running_loss = 0.0

    for i, (inputs, labels) in enumerate(train_loader, 0):
        # wrap them in Variable
        X = inputs
        if is_cuda:
            labels = labels.cuda(async=True)
            X = [ x.cuda(async=True) for x in inputs]

        X_var = [ torch.autograd.Variable( x ) for x in X ]
        label_var = torch.autograd.Variable( labels )

        # zero the parameter gradients
        optimizer.zero_grad()
        #softmax = torch.nn.Softmax()
        # forward + backward + optimize
        output = model( X_var )
#        output = softmax(output)
        loss = criterion(output, label_var)
        loss.backward()
        optimizer.step()

        # print statistics
        avg_loss.update(loss.data[0], labels.size(0))

    return avg_loss.avg

def evaluate(model, eval_loader, is_cuda):
    # switch to evaluate mode
    model.eval()

    result = torch.FloatTensor(0, NUM_CLASSES)
    targets = []
    for i, (inputs, labels) in enumerate(eval_loader):
        X = inputs
        if is_cuda:
            X = [ x.cuda(async=True) for x in inputs]

        X_var = [ torch.autograd.Variable( x, volatile=True ) for x in X ]
        # compute output
        output = model(X_var)
        result = torch.cat( (result, output.data.float().cpu()) )
        if torch.is_tensor(labels):
            targets.append(labels)
    return result, torch.cat(targets) if len(targets) > 0 else torch.LongTensor()


def evaluate_avg(model, eval_loader, is_cuda, num_samples):
#    softmax = torch.nn.Softmax()
    output = None
    for i in range(num_samples):
        o1, targets = evaluate(model, eval_loader, is_cuda)
        if not output is None:
            output.add_(o1)
        else:
            output = o1
    output.div_(num_samples)
#    output = softmax( torch.autograd.Variable(output) ).data
    return output, targets

def validation_loss( criterion, output, targets):
    output_var = torch.autograd.Variable( output )
    target_var = torch.autograd.Variable( targets)
    loss = criterion(output_var, target_var)

    return loss.data[0]

def print_worst(epoch,k,images, output, targets, fp):
    rows = targets.size()[0]
    assert(len(images) == rows)
    res = torch.zeros(rows)
    for i  in range(rows):
        t = targets[i]
#        print(t, images[i].cervix_type, images[i].origpath)
        assert(t == images[i].cervix_type-1)
        res[i] = output[i,t]
    p_v, p_i = res.topk(k=k, dim=0,largest = False)
    print(";; epoch: %d" % epoch, file=fp)
    for i in range(k):
        ix = p_i[i]
        loss = p_v[i]
        ap =  os.path.abspath(images[ix].origpath)
        ip = os.path.abspath(images[ix].filepath)
        s = """%d,%.6f,"%s","%s",%.4f,%.4f,%.4f""" % ( ix, loss, ap, ip, output[ix,0], output[ix,1], output[ix,2])
        print( s, file=fp)


def accuracy(output, targets):
    vals, indices = output.topk(1)
    return torch.eq(targets, indices).float().mean()


def save_evaluation_results(evaluate_output_path, eval_loader, result):

    imlist = eval_loader.dataset.images
    result_np = result.numpy()
    rows,_ = result_np.shape
    assert(len(imlist) == rows)

    with open(evaluate_output_path, 'w') as fp:
        print("image_name,Type_1,Type_2,Type_3", file=fp)
        for i in range(rows):
            a1,a2,a3 = result_np[i]
            p = os.path.basename(imlist[i].origpath)
            print( "%s,%.10f,%.10f,%.10f" % ( p,  a1, a2, a3) , file=fp)

def load_clusters():
    root_dir = '/home/fedor/src/kaggle/cervix/'

    df1 = pd.read_csv(root_dir + "type1.csv", header=None, names=['filename', 'cluster'])
    df1['type'] = 0

    df2 = pd.read_csv(root_dir + "type2.csv", header=None, names=['filename', 'cluster'])
    df2['type'] = 1

    df3 = pd.read_csv(root_dir + "type3.csv", header=None, names=['filename', 'cluster'])
    df3['type'] = 2

    df = pd.concat([df1,df2,df3]  )
    df = df.sort_values(by='filename')

    res = []
    for row in df.itertuples():
        res.append( (row.filename, row.type*30 + row.cluster) )

    return res

def adjust_learning_rate(lr0, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr0 * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def run(options):
    work_dir = os.path.join(options.work_dir, "classifier")
    print(" + Random seed: %d" % options.random_seed)
    random.seed(options.random_seed)
    torch.manual_seed(options.random_seed)
    np.random.seed(options.random_seed)
    print(" + Classes: %d" % NUM_CLASSES)

    model = CervixClassificationModel(num_classes = NUM_CLASSES, batch_norm=True)
#    model = torchvision.models.inception_v3(num_classes = NUM_CLASSES, aux_logits=False)
#    model =  torchvision.models.vgg16_bn(num_classes = NUM_CLASSES)
    criterion = torch.nn.CrossEntropyLoss()
    is_cuda = not options.no_cuda and torch.cuda.is_available()
    print(" + CUDA enabled" if is_cuda else " - CUDA disabled")
    if is_cuda:
        torch.cuda.manual_seed(options.random_seed)
        model =  torch.nn.DataParallel( model ).cuda()
        criterion = criterion.cuda()

#    optimizer = torch.optim.Adagrad(model.parameters(),lr=options.lr, weight_decay=5e-2)
    optimizer = torch.optim.SGD(model.parameters(), lr=options.lr, momentum=0.9, weight_decay=0.03)

    if os.path.isfile(options.model_path):
        print( " + Loading model: %s" % options.model_path)
        model_state_dict = torch.load(options.model_path)
        model.load_state_dict(model_state_dict)


    print(" + Mini-batch size: %d" % options.batch_size)
    if options.train_epochs and options.train_input_path:
        print(" + Training epochs          : %d" % options.train_epochs)
        print(" +          input file      : %s" % options.train_input_path)
        validation_split =  options.validation_split if options.validation_split > 0 and options.validation_split < 1 else None

        train_loader, validate_loader = dataset.create_data_loader(
            annotations_path = options.train_input_path,
            batch_size = options.batch_size,
            num_workers = options.workers,
            validation_split = validation_split,
            is_train = True,
            type_map = TYPE_MAP,
            rseed = options.random_seed
        )

        if validation_split:
            print(" +          validation split: %.2f" % validation_split)

        print(" +          training samples: %.d" % len(train_loader.sampler) )
        if validation_split:
            print(" +          validate samples: %.d" % len(validate_loader.sampler) )
        with open(os.path.join(work_dir, "run.log"), 'w') as fp:
            for epoch in range(0, options.train_epochs):
                adjust_learning_rate(options.lr, optimizer, epoch)
                # train on single epoch
                train_loss = train_single_epoch(model, criterion, optimizer, train_loader, epoch, is_cuda)
                loss_str = "%d: train_loss: %.6f" % (epoch, train_loss)
                if validation_split and epoch % 5 == 0:
                    val_output, val_targets = evaluate_avg(model, validate_loader, is_cuda, 5)
                    val_loss = validation_loss(criterion, val_output, val_targets)
                    acc_val = accuracy(val_output, val_targets)
                    loss_str += ", val_loss: %.6f" % val_loss
                    loss_str += ", accuracy: %.1f" % (acc_val * 100)
                print(loss_str)
                    #print_worst(epoch, 500, val_images, val_output, val_targets, fp)
                    #fp.flush()


        print(" + Training completed")
        if not options.dry_run:
            print( " + Saving model: %s" % options.model_path)
            torch.save( model.state_dict(),  options.model_path)

    if options.evaluate_input_path:
        print(" + Evaluating input file: %s" % options.evaluate_input_path)
        eval_loader,_ = dataset.create_data_loader(
            annotations_path = options.evaluate_input_path,
            num_workers = options.workers,
            batch_size = options.batch_size,
            validation_split = None,
            is_train = False,
            type_map = TYPE_MAP
        )

        output, targets = evaluate_avg(model, eval_loader, is_cuda, 5)
        if targets.size():
            eval_loss = validation_loss(criterion, output, targets)
            print(" + eval_loss: %.6f" % eval_loss)

        # Save results
        if options.evaluate_output_path:
            print(" + Exporting results to: %s" % options.evaluate_output_path)
            softmax = torch.nn.Softmax()
            output = softmax(torch.autograd.Variable( output)).data
            save_evaluation_results(options.evaluate_output_path, eval_loader, output )

    print(" + DONE")

def main():

    default_workers = multiprocessing.cpu_count()

    parser = argparse.ArgumentParser()

    parser.add_argument("--train-input", dest="train_input_path", required=False,
                        help="Path to sloth annotations file with input train data")

    parser.add_argument("--workdir", dest="work_dir",
                        help="Work directory to store intermediate state and caches", required=True)

    parser.add_argument('--train-epochs', dest="train_epochs", type=int,
                        help='Number of training epochs')

    parser.add_argument('--model-path', dest="model_path", required=True,
                        help='Path to (pre)trained model. Model will be updated at the end of training')

    parser.add_argument('--batch-size', type=int, required=True,
                        help='Mini-batch size')

    parser.add_argument('--validation-split', default=0.8, type=float, dest="validation_split",
                        help='Train/Validation split')

    parser.add_argument('--lr', default=0.01, type=float, dest="lr",
                        help='Learning rate')

    parser.add_argument("--eval-input", dest="evaluate_input_path",
                        help="Path to sloth annotations file with data to evaluate", required=False)

    parser.add_argument("--eval-output", dest="evaluate_output_path", required=False,
                        help="Path to evaluation result output csv format")

    parser.add_argument("--no-cuda", dest="no_cuda",action='store_true',
                        help="Disable cuda")

    parser.add_argument("--dry-run", dest="dry_run", action='store_true',
                        help="Do not update model at the end of training")

    parser.add_argument('-j', '--workers', default=default_workers, type=int,
                        help='number of data loading workers (default: %d)' % default_workers)

    parser.add_argument("-r", "--rseed", dest="random_seed",type=int,
                        help="Random seed, will use current time if none specified",
                        required=False, default = int(time.time()))

    options = parser.parse_args()

    run(options)





if __name__ == '__main__':
    main()
