#optimizer
LR = 0.000078
OPTIM = 'AdamW'
weight_decay=0

#


#model


# INPUT_DIM_GEX2ADT = {'GEX': 13953, 'ADT': 134}


# INPUT_DIM_GEX2ATAC = {'GEX': 13431, 'ATAC': 116490}
EMBEDDING_DIM = 64

DROPOUT_RATES_FIRST = [0.022173, 0.296919]
DROPOUT_RATES_GEX = [0.010712, 0.254689]




LAYERS_DIM_FIRST = [512, 2048]
LAYERS_DIM_GEX = [1024, 512]

#LOG_T = 3.031943
LOG_T = 3.463735


N_LSI_COMPONENTS_GEX = 134 # used to be 128 made 134 for partability's sake

N_EPOCHS = 7000


BATCH_SIZE = 2048

SWAP_RATE_FIRST = 0.
SWAP_RATE_GEX = 0.

NUM_WORKERS = 8
MODEL_DUMPS_STORAGE = '/storage1/nerusskikh/OML/dumps'
TENSORBOARD_DUMPS_STORAGE = '/storage1/nerusskikh/OML/experiments'

#data

PATH_FIRST_TRAIN = 'datasets/openproblems_bmmc_cite_phase2_mod2/openproblems_bmmc_cite_phase2_mod2.censor_dataset.output_train_mod1.h5ad'
PATH_FIRST_TEST = 'datasets/openproblems_bmmc_cite_phase2_mod2/openproblems_bmmc_cite_phase2_mod2.censor_dataset.output_test_mod1.h5ad'


PATH_GEX_TRAIN = 'datasets/openproblems_bmmc_cite_phase2_mod2/openproblems_bmmc_cite_phase2_mod2.censor_dataset.output_train_mod2.h5ad'
PATH_GEX_TEST = 'datasets/openproblems_bmmc_cite_phase2_mod2/openproblems_bmmc_cite_phase2_mod2.censor_dataset.output_test_mod2.h5ad'

PATH_TARGET_TRAIN = 'datasets/openproblems_bmmc_cite_phase2_mod2/openproblems_bmmc_cite_phase2_mod2.censor_dataset.output_train_sol.h5ad'
PATH_TARGET_TEST = 'datasets//openproblems_bmmc_cite_phase2_mod2/openproblems_bmmc_cite_phase2_mod2.censor_dataset.output_test_sol.h5ad'