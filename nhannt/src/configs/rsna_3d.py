configs = dict(
    SESS_NAME=None,
    FOLD=0,

    DATA_DIR="./data/",
    TRAIN_FOLDER="npy/stage_1_train_images/",
    TEST_FOLDER="npy/stage_2_test_images/",
    TRAIN_CSV="train_metadata.csv",
    TEST_CSV="stage_2_test_metadata.csv",

    LOGS_DIR="./logs/",
    MODELS_DIR='./rsna_weights/',
    OUTPUT_DIR='./rsna_outputs_stage2/',
    RESUME=None,

    CUDA=True,
    MULTI_GPU=False,
    LOCAL_RANK=0,

    AUX_W=1., # auxiliary classifiers' weights 
    USE_KD=False, # whether to distil from main branch to auxiliary branches
    TAU=1., # temperature
    ALPHA=.9, # auxiliary cross-entropy loss weight

    # Encoder configs.
    MODEL_NAME="se_resnext50_32x4d",
    ATTENTION=True,
    # Decoder configs.
    DROPOUT=0.1,
    RECUR_TYPE="bilstm",
    NUM_LAYERS=2,
    NUM_HEADS=8, # number of heads in the transformer's multiheadattn block
    DIM_FFW=2048,
    NUM_CLASSES=6,

    SPLIT="study",
    FILTER_NO_BRAIN=False,
    P_AUGMENT=0.5,
    CUTMIX=False,
    MIXUP=False,
    TTA=True, # test-time augmentation
    IMG_SIZE=512,
    NUM_INP_CHAN=4,

    EPOCHS=30,
    WARMUP_EPOCHS=3,
    BATCH_SIZE=16,
    NUM_SLICES=10,
    GD_STEPS=1,

    BCE_W = [2., 1., 1., 1., 1., 1.],

    OPTIM="adamw",
    BASE_LR=1e-3,
    WEIGHT_DECAY=1e-2,
    BIAS_LR_FACTOR=2,
    WEIGHT_DECAY_BIAS=0.,

    NUM_WORKERS=8,
    PRINT_FREQ=40,
)