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
    OUTPUT_DIR='./rsna_outputs_stage1/',
    RESUME=None,

    CUDA=True,
    MULTI_GPU=False,
    LOCAL_RANK=0,

    RESUME=None,

    SPLIT="study",
    ATTENTION=True,
    AUX_W=1.,
    MODEL_NAME="tf_efficientnet_b3",
    P_AUGMENT=0.5,
    CUTMIX=True,
    MIXUP=False,
    IMG_SIZE=448,
    NUM_INP_CHAN=4,
    RATIO=1.25,

    EPOCHS=20,
    WARMUP_EPOCHS=2,
    BATCH_SIZE=64,
    GD_STEPS=4,

    BCE_W = [2., 1., 1., 1., 1., 1.],

    OPTIM="adamw",
    BASE_LR=1.6e-2,
    WEIGHT_DECAY=1e-2,
    BIAS_LR_FACTOR=2,
    WEIGHT_DECAY_BIAS=0.,

    NUM_WORKERS=8,
    PRINT_FREQ=40,
)