#optimizer
LR = 0.000076
OPTIM = 'Adam'
weight_decay=0

#
LAMBDA_I = 1.
LAMBDA_E = 1.

#model


# INPUT_DIM_GEX2ADT = {'GEX': 13953, 'ADT': 134}


# INPUT_DIM_GEX2ATAC = {'GEX': 13431, 'ATAC': 116490}
EMBEDDING_DIM = 100

DROPOUT_RATES_ENCODER = [0.107283]
DROPOUT_RATES_DECODER = [0.209865, 0.130652]

LAYERS_DIM_ENCODER = [512]
LAYERS_DIM_DECODER = [1024,1024]

N_LSI_COMPONENTS_GEX = 128
N_LSI_COMPONENTS_ATAC = 128

N_EPOCHS = 1000


BATCH_SIZE = 128


NUM_WORKERS = 8
MODEL_DUMPS_STORAGE = '/storage1/nerusskikh/sc_RNA_competition/dumps/proto_joint'
TENSORBOARD_DUMPS_STORAGE = '/storage1/nerusskikh/sc_RNA_competition/experiments_joint'

#data

PATH_FIRST_TRAIN = '/storage1/ryazantsev/scRNAcompetition/data/phase1-data/joint_embedding/openproblems_bmmc_multiome_phase1/openproblems_bmmc_multiome_phase1.censor_dataset.output_mod2.h5ad'

PATH_GEX_TRAIN = '/storage1/ryazantsev/scRNAcompetition/data/phase1-data/joint_embedding/openproblems_bmmc_multiome_phase1/openproblems_bmmc_multiome_phase1.censor_dataset.output_mod1.h5ad'

PATH_TARGET_TRAIN = '/storage1/ryazantsev/scRNAcompetition/data/phase1-data/joint_embedding/openproblems_bmmc_multiome_phase1/openproblems_bmmc_multiome_phase1.censor_dataset.output_solution.h5ad'
