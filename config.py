import os
import logging

logging_level = logging.INFO
data_dir = "data"
video_path = os.path.join(data_dir, "videos/171218_2_1.MP4")
out_dir = os.path.join(data_dir, "output/171218_2_1")
img_dir = os.path.join(data_dir, "img")
label_dir = os.path.join(data_dir, "labels")
save_img = True
# img_scale = 1
max_frame = 12000
start_frame = 1

if video_path[-12:] == "171214_1.MP4":
    init = {'a': {'bbox': [300, 183, 57, 49]}, 'b': {'bbox': [139, 201, 53, 45]},
            'c': {'bbox': [94, 296, 77, 98]}, 'd': {'bbox': [317, 472, 48, 63]},
            'e': {'bbox': [427, 443, 61, 43]}, 'f': {'bbox': [421, 230, 63, 39]}}
elif video_path[-12:] == "171214_2.MP4":
    init = {'g': {'bbox': [124, 245, 63, 49]}, 'h': {'bbox': [176, 431, 63, 60]},
            'i': {'bbox': [315, 493, 17, 48]}, 'j': {'bbox': [403, 439, 47, 51]},
            'k': {'bbox': [361, 240, 100, 77]}}
elif video_path[-14:] == "171218_1_1.MP4":
    init = {'l': {'bbox': [376, 230, 65, 54]}, 'b': {'bbox': [281, 120, 43, 55]},
            'c': {'bbox': [176, 220, 60, 54]}, 'e': {'bbox': [158, 379, 55, 48]},
            'f': {'bbox': [376, 515, 111, 94]}}
    # Frame 540
    # init = {'l': {"bbox": [384, 229, 69, 45]},
    #         'b': {"bbox": [290, 149, 46, 53]},
    #         'c': {"bbox": [140, 259, 82, 50]},
    #         'e': {"bbox": [161, 385, 57, 45]},
    #         'f': {"bbox": [367, 462, 95, 70]}}

elif video_path[-14:] == "171218_2_1.MP4":
    init = {'a': {'bbox': [415, 260, 67, 54]}, 'k': {'bbox': [335, 130, 58, 63]},
            'g': {'bbox': [160, 193, 66, 75]}, 'd': {'bbox': [152, 406, 55, 50]},
            'i': {'bbox': [290, 522, 37, 40]}, 'j': {'bbox': [470, 327, 110, 151]}}

else:
    init = None

angle_proximity_treshhold = 5
checking_rate = 30
roi_ratio = 1.8
roi_min_size = 80
min_bbox_size = 30
max_bbox_size = 150
correction_overlay_threshold = 0.4
tracking_overlay_threshold = 0.25

confidence_update_rate = 0.5
conf_ud_rt = confidence_update_rate

tracking_data = "data/tracking"
BBOX_KEY = "bbox"
LANDMARKS_KEY = "landmarks"
PITCH_KEY = "pitch"
YAW_KEY = "yaw"
ROLL_KEY = "roll"
BBOX_COLOR = (0, 1, 0)
SELECTED_COLOR = (0.8, 0, 0)

face_detection_iou_trh = 0.6
face_detection_roi_trh = 0.8
face_detection_angle_trh = 0.9

label_frame_step = 10

face_orientation_max_grad = 3

##########################___DATASETS___#########################

wider = "C:/Users/Horio/Documents/Datasets/face_detection/WIDER"

##########################____MEMTRACK____#############################
# ================= data preprocessing ==========================
home_path = 'C:/Users/Horio/Documents/PotatoNet/trackers/memtrack/Home'
root_path = home_path + '/Data/ILSVRC'
tfrecords_path = home_path + '/Data/ILSVRC-TF'
otb_data_dir = home_path + '/Data/Benchmark/OTB'
data_dir = home_path + '/Data'
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

patch_size = 255 + 2 * 8

fix_aspect = True
enlarge_patch = True
if fix_aspect:
    context_amount = 0.5
else:
    z_scale = 2

# ========================== data input ============================
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

label_type = 0  # 0: overlap: 1 dist
overlap_thres = 0.7
dist_thre = 2

# ========================== Memnet ===============================
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

# ========================== train =================================
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

# ========================== evaluation ==================================
batch_size_eval = 2
time_step_eval = 48
num_example_epoch_eval = 1073
max_iterations_eval = num_example_epoch_eval // batch_size_eval

# ========================== tracking ====================================
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
save_path = home_path + '/snapshots'
