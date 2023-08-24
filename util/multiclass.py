from util.confusion_matrix import ConfusionMatrix
import numpy as np
from typing import Any, Optional, Tuple

class MultiClassMetric():
    def __init__(self, num_classes:int, epsilon: float = 1e-7) -> None:
        """
        Args:
            num_classes: the number of classes
            epsilon: prevent nan or zero.
        """
        super(MultiClassMetric, self).__init__()

        self.cm = ConfusionMatrix(num_classes=num_classes)
        self.num_classes = num_classes
        self.epsilon = epsilon

    def result(self, return_str=False) -> Tuple[float, float, dict, dict, dict, dict, list]:
        """ compute accuracy, precision, recall and specificity"""

        tp_dict, fp_dict, fn_dict, tn_dict = self.cm.result()
        total_samples = self.cm.num_samples

        # accuracy
        tp_count = 0
        for class_index in range(self.num_classes):
            tp_count += tp_dict[class_index]

        accuracy = tp_count / (total_samples + self.epsilon)

        # accuracy per class
        class_accuracy_dict = {}
        for class_index in range(self.num_classes):
            class_accuracy_dict[class_index] = \
                (tp_dict[class_index] + tn_dict[class_index])/(total_samples + self.epsilon)

        # precision per class
        class_precision_dict = {}
        for class_index in range(self.num_classes):
            class_precision_dict[class_index] = \
                tp_dict[class_index]/(tp_dict[class_index] + fp_dict[class_index] + self.epsilon)

        # recall per class
        class_recall_dict = {}
        for class_index in range(self.num_classes):
            class_recall_dict[class_index] = \
                tp_dict[class_index]/(tp_dict[class_index] + fn_dict[class_index] + self.epsilon)

        # f1 score pre class
        class_f1_dict = {}
        for class_index in range(self.num_classes):
            class_f1_dict[class_index] =\
                2 * class_precision_dict[class_index] * class_recall_dict[class_index]/(class_precision_dict[class_index] + class_recall_dict[class_index] + self.epsilon)

        # f1 score
        f1_score = 0.0
        for class_index in range(self.num_classes):
            f1_score += class_f1_dict[class_index]
        f1_score = f1_score/self.num_classes

        if return_str:
            accuracy = f'{accuracy:.3f}'
            f1_score = f'{f1_score:.3f}'

            for class_index in range(self.num_classes):
                class_accuracy_dict[class_index] = f'{class_accuracy_dict[class_index]:.3f}'
                class_precision_dict[class_index] = f'{class_precision_dict[class_index]:.3f}'
                class_recall_dict[class_index] = f'{class_recall_dict[class_index]:.3f}'
                class_f1_dict[class_index] = f'{class_f1_dict[class_index]:.3f}'

        return accuracy, f1_score, class_accuracy_dict, class_precision_dict,\
            class_recall_dict, class_f1_dict, self.cm.cm.tolist()

    def reset(self):
        """ clear cm matrix """
        self.cm.reset()

    def set_cm(self, cm:np.ndarray):

        w, h = cm.shape
        assert w == h

        cm = cm.astype(np.int32)

        self.num_classes = w
        self.cm = ConfusionMatrix(num_classes=self.num_classes)
        self.cm.cm = cm
        self.cm.num_samples = np.sum(cm)

    def get_samples(self):
        total_samples = self.cm.num_samples
        cm = self.cm.cm
        class_samples = np.sum(cm, axis=1)

        class_samples_dict = {}
        for i in range(self.num_classes):
            class_samples_dict[i] = class_samples[i]

        return total_samples, class_samples_dict


if __name__ == '__main__':

    valid_cm = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 217, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 2, 212, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 3, 196, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 13, 200, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 21, 190, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 2, 33, 160, 18, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 8, 160, 42, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 13, 180, 3, 17, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 2, 16, 181, 14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 2, 1, 0, 1, 9, 3, 203, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 211, 1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 205, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 191, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16, 193, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 22, 180, 7, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 26, 163, 21, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 12, 171, 20, 5, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 38, 131, 4, 36], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16, 25, 147, 21], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 2, 9, 1, 94]], dtype=np.int32)

    test_cm = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 215, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 213, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 2, 198, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 15, 199, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 28, 186, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 42, 163, 7, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 1, 6, 158, 43, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 11, 172, 4, 26, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 9, 19, 169, 17, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 1, 0, 0, 10, 3, 204, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 207, 5, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 208, 0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 192, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 22, 186, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 17, 179, 13, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 18, 162, 29, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18, 164, 20, 7, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 54, 110, 5, 37], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 20, 20, 133, 36], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 13, 0, 92]],dtype=np.int32)

    #cm = valid_cm
    cm = test_cm

    metric = MultiClassMetric(num_classes=cm.shape[0])
    metric.set_cm(cm)

    total_samples, class_samples_dict = metric.get_samples()

    print(f'total samples = {total_samples}')
    print(f'class samples = {class_samples_dict}')

    accuracy, f1_score, class_accuracy_dict, class_precision_dict, class_recall_dict, \
        class_f1_dict, cm_list = metric.result(True)

    print(f'accuracy = {float(accuracy)*100:.1f}%')
    print(f'f1 score = {float(f1_score)*100:.1f}%')
    print(f'class_accuracy = {class_accuracy_dict}')
    print(f'class_precision = {class_precision_dict}')
    print(f'class_recall = {class_recall_dict}')
    print(f'class_f1 = {class_f1_dict}')
    print(f'confusion matrix')
    print(cm)
    print(f'class samples')
    print(cm.sum(axis=1))

# if __name__ == '__main__':
#
#     valid_cm = np.array([[2,0,0], [1,0,1],[0,2,0]], dtype=np.int32)
#
#     cm = valid_cm
#
#     metric = MultiClassMetric(num_classes=cm.shape[0])
#     metric.set_cm(cm)
#
#     total_samples, class_samples_dict = metric.get_samples()
#
#     print(f'total samples = {total_samples}')
#     print(f'class samples = {class_samples_dict}')
#
#     accuracy, f1_score, class_accuracy_dict, class_precision_dict, class_recall_dict, \
#         class_f1_dict, cm_list = metric.result(True)
#
#     print(f'accuracy = {accuracy}')
#     print(f'f1 = {f1_score}')
#     print(f'class_accuracy = {class_accuracy_dict}')
#     print(f'class_precision = {class_precision_dict}')
#     print(f'class_recall = {class_recall_dict}')
#     print(f'class_f1 = {class_f1_dict}')
#     print(f'confusion matrix')
#     print(cm)
#     print(f'class samples')
#     print(cm.sum(axis=1))
