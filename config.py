IMG_SIZE = 224  # DINOv2 backbones were trained at 518×518
batch_size = 8
epochs = 10
lr = 2e-5
gradient_accumulation_steps = 4
max_grad_norm = 1.0
num_of_vis_data=96

ecg_length = 1000


PROJ_DIM = 512
NUM_MODALITIES = 2