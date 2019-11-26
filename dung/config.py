from easydict import EasyDict as edict

class RSNAConfig():
    def __init__(self):
        self.conf = edict()
        self.conf.version = 3
        self.conf.num_classes = 6
        self.conf.folds = 5
        self.conf.stage1_train_dir = '../dataset/stage_1_train_images'
        self.conf.stage1_test_dir = '../dataset/stage_1_test_images'
        self.conf.new_trainset = 'dataset/trainset_nhannt.csv'
        self.conf.testset_stage1 = 'dataset/testset_stage1.csv'
        self.conf.snapshot_epochs = [5,11,18,19,28,29]

        self.conf.testset_stage2 = 'dataset/testset_stage2.csv'
        self.conf.stage2_test_dir = '../data/dicom/stage_2_test_images'
        self.conf.stage2_metadata = 'dataset/stage_2_test_metadata.csv'

    def update(self, network = None, fp16 = False, dgx = False):
        self.conf.network = network
        if self.conf.network == 'EfficientnetB2':
            self.conf.size = 512
            self.conf.batch_size = 8
        elif self.conf.network == 'EfficientnetB3':
            self.conf.size = 512
            self.conf.batch_size = 6
        elif self.conf.network == 'EfficientnetB4':
            self.conf.size = 380
            self.conf.batch_size = 12
        elif self.conf.network == 'EfficientnetB5':
            self.conf.size = 456
            self.conf.batch_size = 4
        elif self.conf.network == 'SE-ResNeXt101_32x4d':
            self.conf.size = 224
            self.conf.batch_size = 32
        elif self.conf.network == 'Resnext101_32x8d_WSL':
            self.conf.size = 320
            self.conf.batch_size = 18
        elif self.conf.network == 'InceptionV4':
            self.conf.size = 299
            self.conf.batch_size = 48
        elif self.conf.network == 'SE-ResNeXt50':
            self.conf.size = 224
            self.conf.batch_size = 32
        elif self.conf.network == 'InceptionResNetV2':
            self.conf.size = 299
            self.conf.batch_size = 16
        else:
            pass

        if fp16:
            self.conf.batch_size *= 2
        if dgx:
            self.conf.batch_size *= 3
        
        self.conf.batch_size = min(self.conf.batch_size, 64)