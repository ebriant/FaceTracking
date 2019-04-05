import os
import logging

logging_level = logging.INFO
data_dir = "data"
video_path = os.path.join(data_dir, "videos/171214_1.MP4")
img_dir = os.path.join(data_dir, "img")
out_dir = os.path.join(data_dir, "output/171214_1_verif30")
label_dir = os.path.join(data_dir, "labels")
save_box = True

angle_proximity_treshhold = 5
checking_treshold = 30
roi_ratio = 2
roi_min_size = 80
min_bbox_size = 30
overlay_threshold = 0.4

tracking_data = "data/tracking"
BBOX_KEY = "bbox"
LANDMARKS_KEY= "landmarks"
BBOX_COLOR = (0, 1, 0)
SELECTED_COLOR = (0.8, 0, 0)

face_detection_trh = 0.95


##########################___DATASETS___#########################

wider = "C:/Users/Horio/Documents/Datasets/face_detection/WIDER"


##########################____MEMTRACK____#############################
#================= data preprocessing ==========================
home_path = 'C:/Users/Horio/Documents/PotatoNet/trackers/memtrack/Home'
root_path = home_path+'/Data/ILSVRC'
tfrecords_path = home_path+'/Data/ILSVRC-TF'
otb_data_dir = home_path+'/Data/Benchmark/OTB'
data_dir = home_path+'/Data'
data_path_t = os.path.join(root_path, 'Data/VID/train')
data_path_v = os.path.join(root_path, 'Data/VID/val')
anno_path_t = os.path.join(root_path, 'Annotations/VID/train/')
anno_path_v = os.path.join(root_path, 'Annotations/VID/val/')

vid_info_t = './VID_Info/vid_info_train.txt'
vid_info_v = './VID_Info/vid_info_val.txt'
vidb_t = './VID_Info/vidb_train.pk'
vidb_v = './VID_Info/vidb_val.pk'

max_trackid = 50
min_frames = 50

num_threads_t = 16
num_threads_v = 2

patch_size = 255+2*8

fix_aspect = True
enlarge_patch = True
if fix_aspect:
    context_amount = 0.5
else:
    z_scale = 2

#========================== data input ============================
min_queue_examples = 500
num_readers = 2
num_preprocess_threads = 8

z_exemplar_size = 127
x_instance_size = 255

is_limit_search = False
max_search_range = 200

is_augment = True
max_strech_x = 0.05
max_translate_x = 4
max_strech_z = 0.1
max_translate_z = 8

label_type= 0 # 0: overlap: 1 dist
overlap_thres = 0.7
dist_thre = 2

#========================== Memnet ===============================
hidden_size = 512
memory_size = 8
slot_size = [6, 6, 256]
usage_decay = 0.99

clip_gradients = 20.0
keep_prob = 0.8
weight_decay = 0.0001
use_attention_read = False
use_fc_key = False
key_dim = 256


#========================== train =================================
batch_size = 8
time_step = 16

decay_circles = 10000
lr_decay = 0.8
learning_rate = 0.0001
use_focal_loss = False

summaries_dir = 'MemTrack/output/summary/'
checkpoint_dir = 'MemTrack/output/models/'
pretrained_model_checkpoint_path = 'MemTrack/output/pre_models/'

summary_save_step = 500
model_save_step = 5000
validate_step = 5000
max_iterations = 100000
summary_display_step = 8

#========================== evaluation ==================================
batch_size_eval = 2
time_step_eval = 48
num_example_epoch_eval = 1073
max_iterations_eval = num_example_epoch_eval//batch_size_eval

#========================== tracking ====================================
num_scale = 3
scale_multipler = 1.05
scale_penalty = 0.97
scale_damp = 0.6

response_up = 16
response_size = 17
window = 'cosine'
win_weights = 0.15
stride = 8
avg_num = 1

is_save = False
save_path = home_path+'/snapshots'
