#!/usr/bin/env python3
from __future__ import division, absolute_import, print_function

import os
import time
import argparse
import multiprocessing
import json
import random

import numpy as np
from utils.misc import AverageMeter, create_dir_if_not_exists
from utils.transformation import Scale
from utils.annotations import AnnotatedCervixImage, save_annotations

from model import CervixLocalisationModel, alexnet
import dataset

import torch
import torchvision
import torchvision.transforms as transforms

TARGET_SIZE = 256 # target image size

NUM_CLASSES = 3

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

epochs_transform = [
#    transforms.RandomCrop(224),
#    transforms.RandomHorizontalFlip(),
#    transforms.Scale( size=224 ),
    transforms.ToTensor(),
    normalize
]

eval_transform = [
    Scale( size=[TARGET_SIZE, TARGET_SIZE] ),
    transforms.ToTensor(),
]


def train_single_epoch(model, criterion, optimizer, train_loader, epoch, is_cuda):
    model.train() # switch to train mode
    avg_loss = AverageMeter()
    end = time.time()
    running_loss = 0.0

    for i, (inputs, labels) in enumerate(train_loader, 0):
        # wrap them in Variable
        if is_cuda:
            labels = labels.cuda(async=True)
            inputs = inputs.cuda(async=True)

        input_var = torch.autograd.Variable( inputs )
        label_var = torch.autograd.Variable( labels )

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        output = model(input_var)
        loss = criterion(output, label_var)
        loss.backward()
        optimizer.step()

        # print statistics
        avg_loss.update(loss.data[0], labels.size(0))

    return avg_loss.avg

def evaluate(model, eval_loader, is_cuda):
    # switch to evaluate mode
    model.eval()
    result = torch.FloatTensor(0,NUM_CLASSES)
    targets = torch.FloatTensor(0,NUM_CLASSES)
    for i, (inputs, labels) in enumerate(eval_loader):
        if is_cuda:
            inputs = inputs.cuda(async=True)
        input_var = torch.autograd.Variable( inputs, volatile=True )
        # compute output
        output = model(input_var)
        result = torch.cat( (result, output.data.cpu()) )
        targets = torch.cat( (targets, labels) )
    return result, targets

def validation_loss( criterion, output, targets):
    output_var = torch.autograd.Variable( output )
    target_var = torch.autograd.Variable( targets)
    loss = criterion(output_var, target_var)
    return loss.data[0]

def extract_results(eval_loader, output):
    for i in range(len(eval_loader.dataset)):
        ra = eval_loader.dataset.images[i]
        x0,y0,w = output[i].tolist()
        yield ra.with_coords(x0,y0,x0+w,y0+w)


def run(options):
    work_dir = os.path.join(options.work_dir, "localizer")
    print(" + Random seed: %d" % options.random_seed)
    random.seed(options.random_seed)
    torch.manual_seed(options.random_seed)
    np.random.seed(options.random_seed)

#    model = CervixLocalisationModel(num_classes = NUM_CLASSES, batch_norm=True)
    model = torchvision.models.resnet18(num_classes = NUM_CLASSES)
    criterion = torch.nn.MSELoss()
#    criterion= torch.nn.CrossEntropyLoss()

    is_cuda = not options.no_cuda and torch.cuda.is_available()
    print(" + CUDA enabled" if is_cuda else " - CUDA disabled")
    if is_cuda:
        torch.cuda.manual_seed(options.random_seed)
        model =  torch.nn.DataParallel( model ).cuda()
        criterion = criterion.cuda()

#    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
    optimizer = torch.optim.Adadelta(model.parameters(), weight_decay=options.l2reg)

    if os.path.isfile(options.model_path):
        print( " + Loading model: %s" % options.model_path)
        model_state_dict = torch.load(options.model_path)
        model.load_state_dict(model_state_dict)

    print(" + Mini-batch size: %d" % options.batch_size)
    if options.train_epochs and options.train_input_path:
        print(" + Training epochs          : %d" % options.train_epochs)
        print(" +          input file      : %s" % options.train_input_path)
        validation_split =  options.validation_split if options.validation_split > 0 and options.validation_split < 1 else None
        if validation_split:
            print(" +          validation split: %.2f" % validation_split)

        train_loader, validate_loader = dataset.create_data_loader(
            annotations_path = options.train_input_path,
            transform = transforms.Compose (epochs_transform),
            batch_size = options.batch_size,
            num_workers = options.workers,
            validation_split = validation_split,
            is_train = True
        )
        print(" +          training samples: %.d" % len(train_loader.dataset) )

        for epoch in range(0, options.train_epochs):
            # train on single epoch
            train_loss = train_single_epoch(model, criterion, optimizer, train_loader, epoch, is_cuda)
            loss_str = "%d: train_loss: %.6f" % (epoch, train_loss)
            if validation_split:
                val_output, val_targets = evaluate(model, validate_loader, is_cuda)
                val_loss = validation_loss(criterion, val_output, val_targets)
                loss_str += ", val_loss: %.6f" % val_loss
            print(loss_str)

        print(" + Training completed")
        if not options.dry_run:
            print( " + Saving model: %s" % options.model_path)
            torch.save( model.state_dict(),  options.model_path)

    if options.evaluate_input_path:
        print(" + Evaluating input file: %s" % options.evaluate_input_path)
        eval_loader,_ = dataset.create_data_loader(
            annotations_path = options.evaluate_input_path,
            transform = transforms.Compose (eval_transform),
            batch_size = options.batch_size,
            num_workers = options.workers,
            validation_split = None,
            is_train = False
        )

        output, targets = evaluate(model, eval_loader, is_cuda)
        eval_loss = validation_loss(criterion, output, targets)
        print(" + eval_loss: %.6f" % eval_loss)

        # Save results
        if options.evaluate_output_path:
            print(" + Exporting results to: %s" % options.evaluate_output_path)
            res = extract_results( eval_loader, output )
            save_annotations( res , options.evaluate_output_path)

    print(" + DONE")

def main():

    default_workers = multiprocessing.cpu_count()

    parser = argparse.ArgumentParser()

    parser.add_argument("--train-input", dest="train_input_path", required=False,
                        help="Path to sloth annotations file with input train data")

    parser.add_argument("--workdir", dest="work_dir",
                        help="Work directory to store intermediate state and caches", required=True)

    parser.add_argument("--eval-input", dest="evaluate_input_path",
                        help="Path to sloth annotations file with data to evaluate", required=False)

    parser.add_argument('--train-epochs', dest="train_epochs", type=int,
                        help='Number of training epochs')

    parser.add_argument('--model-path', dest="model_path", required=True,
                        help='Path to (pre)trained model. Model will be updated at the end of training')

    parser.add_argument('--batch-size', type=int, required=True,
                        help='Mini-batch size')

    parser.add_argument('--l2reg', type=float, required=False, default=1e-3,
                        help='L2 regularization')

    parser.add_argument('--validation-split', default=0.8, type=float, dest="validation_split",
                        help='Train/Validation split')

    parser.add_argument("--eval-output", dest="evaluate_output_path", required=False,
                        help="Path to evaluation result output in sloth annotations format")

    parser.add_argument("--dry-run", dest="dry_run", action='store_true',
                        help="Do not update model at the end of training")

    parser.add_argument('-j', '--workers', default=default_workers, type=int,
                        help='number of data loading workers (default: %d)' % default_workers)

    parser.add_argument("--no-cuda", dest="no_cuda",action='store_true',
                        help="Disable cuda")

    parser.add_argument("-r", "--rseed", dest="random_seed",type=int,
                        help="Random seed, will use current time if none specified",
                        required=False, default = int(time.time()))

    options = parser.parse_args()

    run(options)


if __name__ == '__main__':
    main()
