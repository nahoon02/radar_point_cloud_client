import numpy as np

class ConfusionMatrix():
    def __init__(self, num_classes: int) -> None:
        """ update confusion matrix

        input shape = (n,)

        Args:
            num_classes: the number of classes
            threshold: if model pixel value > 0.5, the slice is considered as foreground slice

        """
        super(ConfusionMatrix, self).__init__()

        self.cm = np.zeros((num_classes,num_classes), dtype=np.int32)
        self.num_samples = 0
        self.num_classes = num_classes

    def __call__(self, pred: np.ndarray, target: np.ndarray) -> None:
        """
        pred: shape = (N, type = np.int)
        target: shape = (N, type = np_int)
        """
        N = target.shape[0]

        for i in range(N):
            target_class = target[i]
            pred_class = pred[i]
            self.cm[target_class][pred_class] += 1

        self.num_samples += N

    def tp(self):
        tp_dict = {}
        for class_index in range(self.num_classes):
            tp_dict[class_index] = self.cm[class_index][class_index]

        return tp_dict

    def fp(self):
        fp_dict = {}
        for class_index in range(self.num_classes):
            count = 0
            for i in range(self.num_classes):
                if i == class_index: continue
                count += self.cm[i][class_index]

            fp_dict[class_index] = count

        return fp_dict

    def fn(self):
        fn_dict = {}
        for class_index in range(self.num_classes):
            count = 0
            for i in range(self.num_classes):
                if i == class_index: continue
                count += self.cm[class_index][i]

            fn_dict[class_index] = count

        return fn_dict

    def tn(self):
        tp_dict = self.tp()
        fp_dict = self.fp()
        fn_dict = self.fn()

        tn_dict = {}
        total_samples = self.num_samples
        for class_index in range(self.num_classes):
            tn_count = total_samples - (tp_dict[class_index] + fp_dict[class_index] + fn_dict[class_index])
            tn_dict[class_index] = tn_count

        return tn_dict

    def reset(self):
        self.cm[:] = 0
        self.num_samples = 0

    def result(self):
        return self.tp(), self.fp(), self.fn(), self.tn()
