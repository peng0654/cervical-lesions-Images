##################################################
# Training Config
##################################################
GPU = '0'                   # GPU
workers = 2             # number of Dataloader workers
epochs =600            # number of epochs
batch_size =30# batch size
learning_rate = 0.001
# initial learning rate

##################################################
# Model Config
##################################################
image_size = (400, 400)     # size of training images
net = 'resnet34'  # feature extractor
num_attentions =40  # number of attention maps
               # param for update feature centers

##################################################
# Dataset/Path Config
##################################################
tag = 'Cer'                # 'aircraft', 'bird', 'car', or 'dog'

# saving directory of .ckpt models
save_dir = './save_path/ckpt/'
model_name = 'model.ckpt'
log_name = 'train.log'

# checkpoint model for resume training
ckpt = False
# ckpt = save_dir + model_name

##################################################
# Eval Config
##################################################
visualize = False
eval_ckpt = save_dir + model_name
eval_savepath = './FGVC/CUB-200-2011/visualize/'

num_image = 2000-500

beta = -1.0
cutmix_prob = 1.0