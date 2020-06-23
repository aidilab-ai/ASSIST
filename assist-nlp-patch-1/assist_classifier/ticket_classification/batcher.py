import random

class Batcher(object):
    #
    def __init__(self,x,y,batch_size):
        self.x = x
        self.y = y
        self.seq_feature = None
        self.batch_size = batch_size
    #
    def batches(self):
        # To shuffle sequences
        sentences_ids = random.sample(range(len(self.x)), len(self.x))

        # Generator for batch
        batch_x, batch_y = [], []
        if self.batch_size is None:
            batch_size = len(self.x)
        for id in sentences_ids:
            batch_x.append(self.x[id])
            batch_y.append(self.y[id])

            if len(batch_x) % self.batch_size == 0:
                yield batch_x, batch_y
                batch_x, batch_y = [], []

    #
    def addSequenceFeatures(self, seqFeature):
        self.seq_feature = seqFeature
    #
    def batchesWithFeatures(self):
        # To shuffle sequences
        sentences_ids = random.sample(range(len(self.x)), len(self.x))

        # Generator for batch
        batch_x, batch_y, batch_x_feature = [], [], []
        if self.batch_size is None:
            batch_size = len(self.x)
        for id in sentences_ids:
            batch_x.append(self.x[id])
            batch_y.append(self.y[id])
            batch_x_feature.append(self.seq_feature[id])

            if len(batch_x) % self.batch_size == 0:
                yield batch_x, batch_y, batch_x_feature
                batch_x, batch_y, batch_x_feature= [], [], []

