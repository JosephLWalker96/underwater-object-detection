import os

import numpy as np

from score_measure import ScoreMeasurer
import pandas as pd
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


class StatsCollector:
    def __init__(self, iou_threshold=0.5):
        self.epoch_records = None

        self.recall_precision = dict()  # recall -> precision
        self.iou_threshold = iou_threshold

        # total number of predictions
        self.total = 0
        self.total_SUIT = 0
        self.total_target = 0
        # elements for confusion matrix
        self.classified_SUIT = 0
        self.classified_target = 0
        self.misclassified_SUIT_as_target = 0
        self.misclassified_target_as_SUIT = 0
        self.misclassified_background_as_target = 0
        self.misclassified_background_as_SUIT = 0
        self.misclassified_SUIT_as_background = 0
        self.misclassified_target_as_background = 0
        self.contain_nothing = 0

    def save_matrix_as_csv(self, dir_path: str = None, filename: str = 'confusion'):
        matrix = [(self.classified_SUIT, self.misclassified_target_as_SUIT, self.misclassified_background_as_SUIT),
                  (self.misclassified_SUIT_as_target, self.classified_target, self.misclassified_background_as_target),
                  (
                      self.misclassified_SUIT_as_background, self.misclassified_target_as_background,
                      self.contain_nothing)]
        assert dir_path is not None
        matrix = pd.DataFrame(matrix, columns=['True SUIT', 'True target', 'True background'],
                              index=['Predicted SUIT', 'Predicted target', 'Predicted background'])
        matrix.to_csv(os.path.join(dir_path, filename + '.csv'))
        return matrix

    def update(self, prediction):
        measurement = ScoreMeasurer()
        num_pred = len(prediction['labels'])
        ious = prediction['IoU']
        labels = prediction['labels']
        for idx in range(num_pred):
            iou = ious[idx]
            label = labels[idx]
            if iou < 0:
                measurement.false_positive += 1
                if iou == -2:
                    target_label = 1 if label == 2 else 2
                    self.update_confusion_matrix(label, target_label)
                else:
                    self.update_confusion_matrix(label, 0)
            elif iou < self.iou_threshold:
                measurement.false_negative += 1
                self.update_confusion_matrix(0, label)
            else:
                measurement.true_positive += 1
                self.update_confusion_matrix(label, label)

        precision = measurement.get_precision()
        recall = measurement.get_recall()
        if self.recall_precision.__contains__(recall):
            self.recall_precision[recall] = max(precision, self.recall_precision[recall])
        else:
            self.recall_precision[recall] = precision

    def plot_PR_Curve(self, dir_path: str = None, filename: str = 'confusion'):
        pr_pairs = sorted(self.recall_precision.items())
        precision, recall = zip(*pr_pairs)
        plt.plot(recall, precision)
        plt.xlabel('recall')
        plt.ylabel('precision')
        plt.legend()
        plt.savefig(os.path.join(dir_path, filename + '.png'))

    def get_mAP(self):
        sum_precision = 0
        cnt = 0
        assert len(self.recall_precision.values()) > 0
        for precision in self.recall_precision.values():
            sum_precision += precision
            cnt += 1
        return sum_precision / cnt

    def get_interpolated_mAP(self):
        sum_precision = 0
        cnt = 0
        recalls = sorted(self.recall_precision.keys(), reverse=True)
        assert len(recalls) > 0
        increment_precision = self.recall_precision[recalls[0]]
        for recall in recalls:
            if self.recall_precision[recall] > increment_precision:
                increment_precision = self.recall_precision[recall]
            sum_precision += increment_precision
            cnt += 1
        return sum_precision / cnt

    def update_confusion_matrix(self, predicted_label, target_label):
        self.total += 1
        if predicted_label == target_label:
            if predicted_label == 0:
                self.contain_nothing += 1
            elif predicted_label == 1:
                self.total_SUIT += 1
                self.classified_SUIT += 1
            elif predicted_label == 2:
                self.total_target += 1
                self.classified_target += 1
        else:
            if target_label == 0:
                if predicted_label == 1:
                    self.misclassified_background_as_SUIT += 1
                elif predicted_label == 2:
                    self.misclassified_background_as_target += 1
            elif target_label == 1:
                self.total_SUIT += 1
                if predicted_label == 0:
                    self.misclassified_SUIT_as_background += 1
                elif predicted_label == 2:
                    self.misclassified_SUIT_as_target += 1
            elif target_label == 2:
                self.total_target += 1
                if predicted_label == 0:
                    self.misclassified_target_as_background += 1
                elif predicted_label == 1:
                    self.misclassified_SUIT_as_target += 1

    def append_new_record(self, epoch, train_loss, train_score, val_loss, val_score):
        if self.epoch_records is None:
            self.epoch_records = pd.DataFrame(
                columns=['epoch', 'train loss', 'train score', 'validation loss', 'validation score']
            )
        self.epoch_records.append({
            'epoch': epoch,
            'train loss': train_loss,
            'train score': train_score,
            'validation loss': val_loss,
            'validation score': val_score
        }, ignore_index=True)


    def plot_loss_curve(self, dir_path):
        plt.plot('epoch', 'train loss', data=self.epoch_records, marker='', color='skyblue', linewidth=2)
        plt.plot('epoch', 'validation loss', data=self.epoch_records, marker='', color='olive', linewidth=2)
        plt.xlabel('epoch\n(a) Training and Validation Loss')
        plt.ylabel('loss')
        plt.legend()
        plt.savefig(dir_path + '/loss.png')
        plt.close()

    def plot_acc_curve(self, dir_path):
        plt.plot('epoch', 'train score', data=self.epoch_records, marker='', color='skyblue', linewidth=2)
        plt.plot('epoch', 'validation score', data=self.epoch_records, marker='', color='olive', linewidth=2)
        plt.xlabel('epoch\n(a) Training and Validation Score')
        plt.ylabel('Score')
        plt.legend()
        plt.savefig(dir_path + '/score.png')
        plt.close()


def loss_and_acc_plot(train_score_list, train_loss_list, val_score_list, val_loss_list, model_path):
    df1 = pd.DataFrame(
        {'epoch': np.arange(len(train_loss_list)), 'train loss': train_loss_list, 'validation loss': val_loss_list}
    )
    plt.plot('epoch', 'train loss', data=df1, marker='', color='skyblue', linewidth=2)
    plt.plot('epoch', 'validation loss', data=df1, marker='', color='olive', linewidth=2)
    plt.xlabel('epoch\n(a) Training and Validation Loss')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig(model_path + '/loss.png')
    plt.clf()

    df2 = pd.DataFrame(
        {'epoch': np.arange(len(train_score_list)), 'train score': train_score_list,
         'validation score': val_score_list})
    plt.plot('epoch', 'train score', data=df2, marker='', color='skyblue', linewidth=2)
    plt.plot('epoch', 'validation score', data=df2, marker='', color='olive', linewidth=2)
    plt.xlabel('epoch\n(a) Training and Validation Score')
    plt.ylabel('Score')
    plt.legend()
    plt.savefig(model_path + '/score.png')
    plt.close()
