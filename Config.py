import os
import torch
import time
import ml_collections


## PARAMETERS OF THE MODEL
save_model = True
tensorboard = True
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
use_cuda = torch.cuda.is_available()
seed = 666
os.environ['PYTHONHASHSEED'] = str(seed)


cosineLR = True
# epochs = 300
img_size = 224
print_frequency = 1
save_frequency = 5000
vis_frequency = 30
kfold = 2

# task_name = 'Synapse'
# task_name = 'ISIC2018'
task_name = 'DDTI'
# task_name = 'Kvasir-Seg'




if task_name == 'ISIC2018':
    n_channels = 3
    n_labels = 1
    epochs = 500
    learning_rate = 1e-3
    batch_size = 4
    early_stopping_patience = 40
    print_frequency = 1

elif task_name == 'Kvasir-Seg':
    n_channels = 3
    n_labels = 1
    epochs = 500
    learning_rate = 1e-3
    batch_size = 2
    early_stopping_patience = 40
    print_frequency = 1

elif task_name == 'DDTI':
    n_channels = 3
    n_labels = 1
    epochs = 500
    learning_rate = 1e-3
    batch_size = 2
    early_stopping_patience = 40
    print_frequency = 1

elif task_name == "Synapse":
    epochs = 500
    learning_rate = 1e-3
    early_stopping_patience = 50
    batch_size = 12
    n_labels = 9
    n_channels = 1
    print_frequency = 1


model_name = 'DHR_Net'
# model_name = 'DHR_Net_S'
# model_name = 'DHR_Net_L'







# used in testing phase, copy the session name in training phase
# test_session = "Test_session_10.10_09h05"
test_session = "Test_session"

if task_name == 'Synapse':
    train_dataset = './datasets/Synapse/train_npz/'
    val_dataset = './datasets/Synapse/train_npz/'
    test_dataset = './datasets/Synapse/test_vol_h5/'
else:
    train_dataset = './datasets/'+ task_name+ '/Train_Folder/'
    val_dataset = './datasets/' + task_name + '/Val_Folder/'
    test_dataset = './datasets/'+ task_name+ '/Test_Folder/'



session_name       = 'Test_session' + '_' + time.strftime('%m.%d_%Hh%M')
save_path          = "./train_test_session/" + task_name +'_kfold/'+ model_name +'/' + session_name + '/'
model_path         = save_path + 'models/'
tensorboard_folder = save_path + 'tensorboard_logs/'
logger_path        = save_path + session_name + ".log"
visualize_path     = save_path + 'visualize_val/'







##########################################################################
# CTrans configs
##########################################################################
def get_CTranS_config():
    config = ml_collections.ConfigDict()
    config.transformer = ml_collections.ConfigDict()
    config.KV_size = 960  # KV_size = Q1 + Q2 + Q3 + Q4
    config.transformer.num_heads  = 4
    config.transformer.num_layers = 4
    config.expand_ratio           = 4  # MLP channel dimension expand ratio
    config.transformer.embeddings_dropout_rate = 0.1
    config.transformer.attention_dropout_rate = 0.1
    config.transformer.dropout_rate = 0
    config.patch_sizes = [16,8,4,2]
    config.base_channel = 64 # base channel of U-Net
    config.n_classes = 1
    return config



##########################################################################
# DTrans configs
##########################################################################
def get_DTranS_config():
    config = ml_collections.ConfigDict()
    config.transformer = ml_collections.ConfigDict()
    config.transformer.num_heads  = 4
    config.transformer.num_layers = 4
    config.expand_ratio           = 2
    config.transformer.embedding_channels = 32 * config.transformer.num_heads
    config.KV_size = config.transformer.embedding_channels * 4
    config.KV_size_S = config.transformer.embedding_channels
    config.transformer.attention_dropout_rate = 0.1
    config.transformer.dropout_rate = 0.1
    config.patch_sizes=[16,8,4,2]
    config.base_channel = 32
    config.decoder_channels = [32,64,128,256,512]
    return config