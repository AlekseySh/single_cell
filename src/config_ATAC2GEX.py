#optimizer
LR = 0.000099
OPTIM = 'Adam'
weight_decay=0

#


#model


# INPUT_DIM_GEX2ADT = {'GEX': 13953, 'ADT': 134}
# OUTPUT_DIM_GEX2ADT = 256

# INPUT_DIM_GEX2ATAC = {'GEX': 13431, 'ATAC': 116490}
EMBEDDING_DIM = 64

DROPOUT_RATES_FIRST = [0.114164, 0.579678, 0.29451, 0.445662]
DROPOUT_RATES_GEX = [ 0.115739, 0.265155, 0.359979]




LAYERS_DIM_FIRST = [2048, 512, 2048, 2048]
LAYERS_DIM_GEX = [1024, 512, 1024]

LOG_T = 3.440161	


N_LSI_COMPONENTS_FIRST= 128
DROP_FIRST_ATAC = True

N_LSI_COMPONENTS_GEX = 256
DROP_FIRST_GEX = True

N_EPOCHS = 1000
EPOCHS_GEX2ADT = 1000

BATCH_SIZE = 2048

SWAP_RATE_FIRST = 0.035
SWAP_RATE_GEX = 0.049

NUM_WORKERS = 8
MODEL_DUMPS_STORAGE = '/storage1/nerusskikh/OML/dumps'
TENSORBOARD_DUMPS_STORAGE = '/storage1/nerusskikh/OML/experiments'

#data

PATH_FIRST_TRAIN = 'datasets/phase2-data/match_modality/openproblems_bmmc_multiome_phase2_mod2/openproblems_bmmc_multiome_phase2_mod2.censor_dataset.output_train_mod1.h5ad'
PATH_FIRST_TEST = 'datasets/match_modality/openproblems_bmmc_multiome_phase1_mod2/openproblems_bmmc_multiome_phase1_mod2.censor_dataset.output_test_mod1.h5ad'


PATH_GEX_TRAIN = 'datasets/match_modality/openproblems_bmmc_multiome_phase2_mod2/openproblems_bmmc_multiome_phase2_mod2.censor_dataset.output_train_mod2.h5ad'
PATH_GEX_TEST = '/storage1/ryazantsev/scRNAcompetition/data/phase1-data/match_modality/openproblems_bmmc_multiome_phase1_mod2/openproblems_bmmc_multiome_phase1_mod2.censor_dataset.output_test_mod2.h5ad'

PATH_TARGET_TRAIN = '/storage1/ryazantsev/scRNAcompetition/data/phase2-data/match_modality/openproblems_bmmc_multiome_phase2_mod2/openproblems_bmmc_multiome_phase2_mod2.censor_dataset.output_train_sol.h5ad'
PATH_TARGET_TEST = '/storage1/ryazantsev/scRNAcompetition/data/phase1-data/match_modality/openproblems_bmmc_multiome_phase1_mod2/openproblems_bmmc_multiome_phase1_mod2.censor_dataset.output_test_sol.h5ad'