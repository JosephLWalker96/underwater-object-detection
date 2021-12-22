import copy

from Averager import Averager
import numpy as np
import torch.nn as nn
import torch
import time
import datetime
from torch.utils.data import DataLoader
from QRDatasets import QRDatasets
from utils import get_train_transform, collate_fn, plotting, get_iou_score
from wbf_ensemble import make_ensemble_predictions, run_wbf
import os
import pandas as pd
import split_data
from tqdm import tqdm
from model import net
import argparse
import cv2

class train:
    def __init__(self, model: nn.Module, optimizer: torch.optim, num_epochs: int, train_datasets: list,
                 val_datasets: list, batch_size: int = 4, valid_ratio: float = 0.2, early_stop: int = 2,
                 lr_scheduler: torch.optim.lr_scheduler = None, model_dir: str = 'models', model_name:str = 'faster-cnn',
                 use_grayscale: bool = False):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model = model
        self.model.to(self.device)
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.lr_scheduler = lr_scheduler
        
        if len(train_datasets) == 1:
            self.train_dataset = train_datasets[0]
            self.val_dataset = val_datasets[0]
        else:
            self.train_datasets = train_datasets
            self.val_datasets = val_datasets

        # getting where to store model
#         model_path = model_path.split('/')
        self.model_dir_path = model_dir
        self.model_filename = model_name

        # This is required for early stopping, the number of epochs we will wait with no improvement before stopping
        self.early_stop = early_stop
        self.batch_size = batch_size
        self.valid_ratio = valid_ratio
        
        self.use_grayscale = use_grayscale

    def cross_val_training(self):
        patience = self.early_stop
        best_val = None
        best_loss = None
        best_model = None
        train_score_list = []
        train_loss_list = []
        val_score_list = []
        val_loss_list = []
        
        print("start training")
        for epoch in range(len(self.train_datasets)):
            print("Epoch "+str(epoch+1))
            train_loss_hist = Averager()
            valid_loss_hist = Averager()
            start_time = time.time()
            itr = 1
            train_image_precisions = []
            iou_thresholds = [x for x in np.arange(0.5, 0.76, 0.05)]
            train_loss_hist.reset()
            train_scores = []
            
            train_dataset = torch.utils.data.ConcatDataset([*self.train_datasets[:epoch], *self.train_datasets[epoch+1:]])
            valid_dataset = self.train_datasets[epoch]
            train_data_loader = DataLoader(train_dataset, shuffle=True, batch_size=self.batch_size, pin_memory=False, 
                                           collate_fn=collate_fn, num_workers=4)
            valid_data_loader = DataLoader(valid_dataset, shuffle=True, batch_size=self.batch_size, pin_memory=False, 
                                           collate_fn=collate_fn, num_workers=4)
            
            print("training")
            for images, targets, image_ids in tqdm(train_data_loader):
                self.model.train()
                images = list(image.to(self.device) for image in images)

                targets = [{k: v.to(self.device) if k == 'labels' else v.float().to(self.device) for k, v in t.items()}
                           for t in targets]
                # [{k: v.double().to(device) if k =='boxes' else v.to(device) for k, v in t.items()} for t in targets]
                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                loss_value = losses.item()

                train_loss_hist.send(loss_value)

                self.optimizer.zero_grad()
                losses.backward()
                self.optimizer.step()

                # if itr % 50 == 0:
                #     print(f"Iteration #{itr} loss: {loss_value}")

                itr += 1
#                 predictions = make_ensemble_predictions(images, self.device, [self.model])
                self.model.eval()
                predictions = self.model(images)
                train_image_precisions = self.gather_iou_scores(predictions, targets, images,
                                                                train_image_precisions, iou_thresholds)

            train_score_list.append(np.mean(train_image_precisions))
            train_loss_list.append(train_loss_hist.value)

            # update the learning rate
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            # At every epoch we will also calculate the validation IOU
            print("validation")
            validation_image_precisions = []
            for images, targets, imageids in tqdm(valid_data_loader):  # return image, target, image_id
                # model must be in train mode so that forward() would return losses
                self.model.train()
                images = list(image.to(self.device) for image in images)
                targets = [{k: v.to(self.device) if k == 'labels' else v.float().to(self.device) for k, v in t.items()}
                           for t in targets]
                # outputs = model(images)
                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                loss_value = losses.item()
                valid_loss_hist.send(loss_value)

                # it is simply the combination of every output list from the model lists
#                 predictions = make_ensemble_predictions(images, self.device, [self.model])
                self.model.eval()
                predictions = self.model(images)
                # gathering the iou scores into validation_image_precisions
                validation_image_precisions = self.gather_iou_scores(predictions, targets, images,
                                                                     validation_image_precisions, iou_thresholds)
#             print(validation_image_precisions)
            val_iou = np.mean(validation_image_precisions)
#             print(val_iou)
            val_score_list.append(val_iou)
            val_loss = valid_loss_hist.value
            val_loss_list.append(val_loss)
            if not os.path.exists(self.model_dir_path):
                os.mkdir(self.model_dir_path)
            plotting(train_score_list, train_loss_list, val_score_list, val_loss_list, self.model_dir_path)
            print(f"Epoch #{epoch + 1} Validation Loss: {val_loss}", "Validation Predicted Mean Score: {0:.4f}".format(val_iou),
                  "Time taken :",
                  str(datetime.timedelta(seconds=time.time() - start_time))[:7])
#             if not best_val:
            if not best_loss:
                # So any validation roc_auc we have is the best one for now
                best_val = val_iou
                best_loss = val_loss
                # Saving the model
                print("Saving model to "+self.model_dir_path + "/" + self.model_filename)
                with open(self.model_dir_path + "/" + self.model_filename, 'w') as f:
                    torch.save(self.model, self.model_dir_path + "/" + self.model_filename)
                best_model = copy.deepcopy(self.model)
                # continue
#           elif val_iou >= best_val:
            elif val_loss <= best_loss:
                print("Saving model")
                best_val = val_iou
                best_loss = val_loss
                # Resetting patience since we have new best validation accuracy
                patience = self.early_stop
                # Saving current best model
                print("Saving model to "+self.model_dir_path + "/" + self.model_filename)
                with open(self.model_dir_path + "/" + self.model_filename, 'w') as f:
                    torch.save(self.model, self.model_dir_path + "/" + self.model_filename)

            else:
                patience -= 1
                if patience == 0:
                    print('Early stopping.')
                    print('Best Validation Predicted Mean Score: {:.3f}'.format(best_val))
                    print('Best Validation Loss: {:.3f}'.format(best_loss))
                    break

        return best_model, train_score_list, train_loss_list, val_score_list, val_loss_list
    
    def mini_batch_training(self):
        # preparing dataloader
        train_data_loader = DataLoader(self.train_dataset, shuffle=True, batch_size=self.batch_size,
                                       pin_memory=True, collate_fn=collate_fn, num_workers=4)
        valid_data_loader = DataLoader(self.val_dataset, shuffle=True, batch_size=self.batch_size,
                                       pin_memory=True, collate_fn=collate_fn, num_workers=4)

        patience = self.early_stop
        best_val = None
        best_loss = None
        best_model = None
        train_score_list = []
        train_loss_list = []
        val_score_list = []
        val_loss_list = []

        print("start training")

        for epoch in range(self.num_epochs):
            print("Epoch "+str(epoch+1))
            train_loss_hist = Averager()
            valid_loss_hist = Averager()
            start_time = time.time()
            itr = 1
            train_image_precisions = []
            iou_thresholds = [x for x in np.arange(0.5, 0.76, 0.05)]
            train_loss_hist.reset()
            train_scores = []
            print("training")
            with torch.enable_grad():
                for images, targets, image_ids in tqdm(train_data_loader):
                    self.model.train()
                    images = list(image.to(self.device) for image in images)

                    targets = [{k: v.to(self.device) if k == 'labels' else v.float().to(self.device) for k, v in t.items()}
                               for t in targets]
                    # [{k: v.double().to(device) if k =='boxes' else v.to(device) for k, v in t.items()} for t in targets]
                    loss_dict = self.model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())
                    loss_value = losses.item()

                    train_loss_hist.send(loss_value)

                    self.optimizer.zero_grad()
                    losses.backward()
                    self.optimizer.step()

                    # if itr % 50 == 0:
                    #     print(f"Iteration #{itr} loss: {loss_value}")

                    itr += 1
    #                 predictions = make_ensemble_predictions(images, self.device, [self.model])
                    self.model.eval()
                    predictions = self.model(images)
                    train_image_precisions = self.gather_iou_scores(predictions, targets, images,
                                                                    train_image_precisions, iou_thresholds)

            train_score_list.append(np.mean(train_image_precisions))
            train_loss_list.append(train_loss_hist.value)

            # update the learning rate
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            # At every epoch we will also calculate the validation IOU
            print("validation")
            validation_image_precisions = []
            with torch.no_grad():
                for images, targets, imageids in tqdm(valid_data_loader):  # return image, target, image_id
                    # model must be in train mode so that forward() would return losses
                    self.model.train()
                    images = list(image.to(self.device) for image in images)
                    targets = [{k: v.to(self.device) if k == 'labels' else v.float().to(self.device) for k, v in t.items()}
                               for t in targets]
                    # outputs = model(images)
                    loss_dict = self.model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())
                    loss_value = losses.item()
                    valid_loss_hist.send(loss_value)

                    # it is simply the combination of every output list from the model lists
    #                 predictions = make_ensemble_predictions(images, self.device, [self.model])
                    self.model.eval()
                    predictions = self.model(images)
                    # gathering the iou scores into validation_image_precisions
                    validation_image_precisions = self.gather_iou_scores(predictions, targets, images,
                                                                         validation_image_precisions, iou_thresholds)
#             print(validation_image_precisions)
            val_iou = np.mean(validation_image_precisions)
#             print(val_iou)
            val_score_list.append(val_iou)
            val_loss = valid_loss_hist.value
            val_loss_list.append(val_loss)
            if not os.path.exists(self.model_dir_path):
                os.mkdir(self.model_dir_path)
            plotting(train_score_list, train_loss_list, val_score_list, val_loss_list, self.model_dir_path)
            print(f"Epoch #{epoch + 1} Validation Loss: {val_loss}", "Validation Predicted Mean Score: {0:.4f}".format(val_iou),
                  "Time taken :",
                  str(datetime.timedelta(seconds=time.time() - start_time))[:7])
#             if not best_val:
            if not best_loss:
                # So any validation roc_auc we have is the best one for now
                best_val = val_iou
                best_loss = val_loss
                print("Saving model")
                # Saving the model
                with open(self.model_dir_path + "/" + self.model_filename, 'w') as f:
                    torch.save(self.model, self.model_dir_path + "/" + self.model_filename)
                best_model = copy.deepcopy(self.model)
                # continue
#           elif val_iou >= best_val:
            elif val_loss <= best_loss:
                print("Saving model")
                best_val = val_iou
                best_loss = val_loss
                # Resetting patience since we have new best validation accuracy
                patience = self.early_stop
                # Saving current best model
                with open(self.model_dir_path + "/" + self.model_filename, 'w') as f:
                    torch.save(self.model, self.model_dir_path + "/" + self.model_filename)

            else:
                patience -= 1
                if patience == 0:
                    print('Early stopping.')
                    print('Best Validation Predicted Mean Score: {:.3f}'.format(best_val))
                    print('Best Validation Loss: {:.3f}'.format(best_loss))
                    break

        return best_model, train_score_list, train_loss_list, val_score_list, val_loss_list

    # return the array storing the scores of each image from the given images array
    def gather_iou_scores(self, predictions, targets, images: np.array, image_precisions, iou_thresholds):
        for i, image in enumerate(images):
            predictions[i]['boxes']=predictions[i]['boxes']/512
            
            target_n = len(targets[i]['labels'])
            for j in range(target_n):
                result_box, iou_score = get_iou_score(predictions[i], targets[i], 512, 512, j)
                image_precisions.append(max(0, iou_score))
        return image_precisions

    def predict(self):
        if (self.test_dataset):
            detection_threshold = 0.5
            results = []
            outputs = []
            test_images = []
            data_loader = DataLoader(self.test_dataset, shuffle=True, batch_size=self.test_dataset.__len__(),
                                     pin_memory=True, collate_fn=collate_fn, num_workers=4)
            self.model.eval()
            for images, targets, image_ids in data_loader:
                images = list(image.to(self.device) for image in images)
                predictions = make_ensemble_predictions(images, self.device, [self.model])

                for i, image in enumerate(images):
                    test_images.append(image)  # Saving image values
                    boxes, scores, labels = run_wbf(predictions, image_index=i)

                    boxes = boxes.astype(np.int32).clip(min=0, max=1023)

                    preds = boxes
                    preds_sorted_idx = np.argsort(scores)[::-1]
                    preds_sorted = preds[preds_sorted_idx]
                    boxes = preds

                    output = {
                        'boxes': boxes,
                        'scores': scores
                    }

                    outputs.append(output)  # Saving outputs and scores
                    image_id = image_ids[i]

            return outputs, test_images
        else:
            print("There is no defined dataset for testing")
            return None, None

def main(args):
    # execute only if run as a script
    path_to_images = args.image_path

    if path_to_images is None:
        print("Please specify image path with '--image_path path_to_images'")
        raise FileNotFoundError

    # check whether the path exists
    if not os.path.exists(path_to_images):
        print("Train Image path does not exist")
        raise FileNotFoundError
    
    if not os.path.exists(path_to_images+"/train_qr_labels.csv"):
        # CSVGenerator.run(args)
        os.system('python split_data.py --dataset_path '+args.dataset_path)
        # split_data.run(args)

    train_datasets = []
    val_datasets = []
    print("loading "+path_to_images+"/train_qr_labels.csv")
    train_df = pd.read_csv(path_to_images+"/train_qr_labels.csv")
    val_df = pd.read_csv(path_to_images+"/val_qr_labels.csv")
    train_tf = get_train_transform()
    train_dataset = QRDatasets(path_to_images+'/train', train_df, transforms=train_tf, use_grayscale=args.use_grayscale)
    val_dataset = QRDatasets(path_to_images+'/val', val_df, transforms=train_tf, use_grayscale=args.use_grayscale)
    train_datasets.append(train_dataset)
    val_datasets.append(val_dataset)

    model = net(num_classes=3, nn_type=args.model, use_grayscale=args.use_grayscale)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    my_trainer = train(model=model, optimizer=optimizer, num_epochs=args.num_epoch, early_stop=args.early_stop, 
                       train_datasets=train_datasets, val_datasets=val_datasets, model_dir = args.image_path+'/../models',
                       model_name=args.model,lr_scheduler=scheduler, batch_size=args.batch_size, 
                       use_grayscale=args.use_grayscale)
    
    my_trainer.mini_batch_training()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str)
    parser.add_argument('--label_path', type=str)
    parser.add_argument('--dataset_path', type=str)
    parser.add_argument('--model', default='faster-rcnn', type=str)
    parser.add_argument('--lr', default=0.002, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight_decay', default=0.0005, type=float)
    parser.add_argument('--step_size', default=8, type=int)
    parser.add_argument('--gamma', default=0.1, type=float)
    parser.add_argument('--num_epoch', default=10, type=int)
    parser.add_argument('--early_stop', default=2, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--valid_ratio', default=0.2, type=float)
    parser.add_argument('--use_grayscale', default=False, action='store_true')
    main(parser.parse_args())
