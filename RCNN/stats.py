import os

import numpy as np

from score_measure import ScoreMeasurer
import pandas as pd
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


class StatsCollector:
    def __init__(self, iou_threshold=0.5, dir_path: str = '../ExperimentRecords', transform_type=None, exp_num: str = None,
                 exp_env: str = None):
        if exp_num == 'exp3' or exp_num == 'exp4':
            assert exp_env is not None

        self.dir_path = dir_path
        if not os.path.exists(dir_path):
            os.system('mkdir ' + self.dir_path)
        self.exp_num = exp_num if exp_num is not None else 'exp1'
        self.exp_env = exp_env if (exp_num is not None) and (exp_num == 'exp3' or exp_num == 'exp4') else 'NAN'

        self.epoch_records = None
        if os.path.exists(os.path.join(self.dir_path, 'epoch_record.csv')):
            self.epoch_records = pd.read_csv(os.path.join(self.dir_path, 'epoch_record.csv'))

        self.test_records = None
        if os.path.exists(os.path.join(self.dir_path, 'test_record.csv')):
            self.test_records = pd.read_csv(os.path.join(self.dir_path, 'test_record.csv'))

        self.transform_type = 'NAN' if transform_type is None else transform_type

        # variable for mAP calculation\
        self.mAP = 0
        self.mAP_ready = False
        self.recall_precision = {0: dict(), 1: dict(), 2: dict()}  # recall -> precision
        self.iou_threshold = iou_threshold
        self.confidences = {0: [], 1: [], 2: []}
        self.tp = {0: [], 1: [], 2: []}
        self.fp = {0: [], 1: [], 2: []}
        self.GTBox = {0: 0, 1: 0, 2: 0}

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
        num_pred = len(prediction['labels'])
        ious = prediction['IoU']
        confidences = prediction['scores']
        labels = prediction['labels']
        match_targets = prediction['matched_target']
        num_target = prediction['num_target']
        is_matched = np.zeros(prediction['m'])
        for label in num_target.keys():
            assert label == 1 or label == 2 or label == 0
            if label == 0:  # background
                continue
            self.GTBox[label] += num_target[label]

        for idx in range(num_pred):
            measurement = ScoreMeasurer()
            iou = ious[idx]
            matched_target = match_targets[idx]

            label = labels[idx]
            confidence = confidences[idx]
            if iou < self.iou_threshold:
                measurement.false_positive = 1
                if iou == -2:
                    target_label = 1 if label == 2 else 2
                    self.update_confusion_matrix(label, target_label)
                else:
                    self.update_confusion_matrix(label, 0)
            else:
                if is_matched[matched_target] == 0:
                    measurement.true_positive = 1
                    is_matched[matched_target] = 1
                    self.update_confusion_matrix(label, label)
                else:
                    measurement.false_positive = 1
                    self.update_confusion_matrix(label, 0)

            self.confidences[label].append(confidence)
            self.tp[label].append(measurement.true_positive)
            self.fp[label].append(measurement.false_positive)

    def plot_PR_Curve(self, dir_path: str = None, filename: str = 'confusion'):
        pr_pairs = sorted(self.recall_precision.items())
        precision, recall = zip(*pr_pairs)
        plt.plot(recall, precision)
        plt.xlabel('recall')
        plt.ylabel('precision')
        plt.legend()
        plt.savefig(os.path.join(dir_path, filename + '.png'))

    def setup_mAP(self, label):
        # print(type(self.confidences[0]))
        indices = np.argsort(self.confidences[label])
        # print(indices)
        self.tp[label] = np.cumsum(np.array(self.tp[label])[indices])
        self.fp[label] = np.cumsum(np.array(self.fp[label])[indices])
        precisions = np.array(self.tp[label]) / (self.tp[label] + self.fp[label])
        recalls = np.array(self.tp[label]) / self.GTBox[label]

        for precision, recall in zip(precisions, recalls):
            if self.recall_precision[label].__contains__(recall):
                self.recall_precision[label][recall] = max(precision, self.recall_precision[label][recall])
            else:
                self.recall_precision[label][recall] = precision

    def get_mAP(self):
        if self.mAP_ready:
            return self.mAP
        for label in range(1, 3):
            self.setup_mAP(label)
            ap = 0
            recalls = sorted(self.recall_precision[label].keys(), reverse=False)
            if len(recalls) < 0:
                continue

            for idx in range(len(recalls) - 1, 0, -1):
                curr = recalls[idx]
                prev = recalls[idx - 1]
                self.recall_precision[label][prev] = \
                    max(self.recall_precision[label][curr], self.recall_precision[label][prev])

            for idx in range(0, len(recalls) - 1):
                curr = recalls[idx]
                next = recalls[idx + 1]
                ap += self.recall_precision[label][next] * (next - curr)

            self.mAP += ap
        self.mAP /= 2
        self.mAP_ready = True
        return self.mAP

    def clear_mAP(self):
        self.confidences = {0: [], 1: [], 2: []}
        self.tp = {0: [], 1: [], 2: []}
        self.fp = {0: [], 1: [], 2: []}
        self.GTBox = {0: 0, 1: 0, 2: 0}
        self.mAP = 0
        self.mAP_ready = False

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

    def append_new_train_record(self, epoch, train_loss, train_score, val_loss, val_score):
        if self.epoch_records is None:
            self.epoch_records = pd.DataFrame(
                columns=['experiment number', 'experiment environment',
                         'epoch', 'train loss', 'train score', 'validation loss', 'validation score']
            )
        self.epoch_records = self.epoch_records.append({
            'experiment number': self.exp_num,
            'experiment environment': self.exp_env,
            'epoch': epoch,
            'train loss': train_loss,
            'train iou': train_score,
            'validation loss': val_loss,
            'validation iou': val_score,
            'validation mAP': self.get_mAP()
        }, ignore_index=True)

    def append_new_test_record(self, mean_iou_score):
        if self.test_records is None:
            self.test_records = pd.DataFrame(
                columns=['experiment number', 'experiment environment', 'mean iou', 'mAP', 'interpolated mAP']
            )
        self.test_records = self.test_records.append({
            'experiment number': self.exp_num,
            'transform': self.transform_type,
            'experiment environment': self.exp_env,
            'mean iou': mean_iou_score,
            'mAP': self.get_mAP()
        }, ignore_index=True)

    def save_result(self, isTrain: bool = True):
        if isTrain and self.epoch_records is not None:
            self.epoch_records.to_csv(os.path.join(self.dir_path, 'epoch_record.csv'), index=False)
            # self.plot_loss_curve()
            # self.plot_acc_curve()
        if not isTrain and self.test_records is not None:
            self.test_records.to_csv(os.path.join(self.dir_path, 'test_record.csv'), index=False)

    def plot_loss_curve(self):
        filename = self.check_filename(filename='loss', ext='.png')
        plt.plot('epoch', 'train loss', data=self.epoch_records, marker='', color='skyblue', linewidth=2)
        plt.plot('epoch', 'validation loss', data=self.epoch_records, marker='', color='olive', linewidth=2)
        plt.xlabel('epoch\n(a) Training and Validation Loss')
        plt.ylabel('loss')
        plt.legend()
        plt.savefig(os.path.join(self.dir_path, filename))
        plt.close()

    def plot_acc_curve(self):
        filename = self.check_filename(filename='loss', ext='.png')
        plt.plot('epoch', 'train score', data=self.epoch_records, marker='', color='skyblue', linewidth=2)
        plt.plot('epoch', 'validation score', data=self.epoch_records, marker='', color='olive', linewidth=2)
        plt.xlabel('epoch\n(a) Training and Validation Score')
        plt.ylabel('Score')
        plt.legend()
        plt.savefig(os.path.join(self.dir_path, filename))
        plt.close()

    def check_filename(self, filename, ext):
        _filename = filename
        cnt = 2
        while os.path.exists(os.path.join(self.dir_path, _filename)):
            _filename = filename + str(cnt)
            cnt += 1
        return _filename + ext


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
