import random
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Lambda
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.optimizers import Adam

from model.model_functional import YOLOv3
from loss.loss_functional import yolo_loss
from dataloader.dataloader import YoloDataGenerator

from utils.callbacks import ExponentDecayScheduler, LossHistory, ModelCheckpoint
from utils.utils import *
from configs import *
import datetime

# =======================================================
# Set a seed value
# =======================================================
seed_value = 121

# =======================================================
# 1. Set `PYTHONHASHSEED` environment variable at a fixed value
# =======================================================
os.environ['PYTHONHASHSEED'] = str(seed_value)

# =======================================================
# 2. Set `python` built-in pseudo-random generator at a fixed value
# =======================================================
random.seed(seed_value)

# =======================================================
# 3. Set `numpy` pseudo-random generator at a fixed value
# =======================================================
np.random.seed(seed_value)

# =======================================================
# 4. Set `tensorflow` pseudo-random generator at a fixed value
# =======================================================
tf.random.set_seed(seed_value)
print(tf.__version__)

"""""
When training your own target detection model, you must pay attention to the following points: 
1.  Before training, carefully check whether your format meets the requirements. The library requires the data set
    format to be VOC format, and the content to be prepared includes input pictures and labels. The input image is a
    .jpg image, no fixed size is required, and it will be automatically resized before being passed into training. 
    Grayscale images will be automatically converted to RGB images for training, no need to modify them yourself. 
    If the suffix of the input image is not jpg, you need to convert it into jpg in batches before starting training. 

    The tag is in .xml format, and the file contains target information to be detected. The tag file corresponds to
    the input image file.
    
2.  The trained weight file is saved in the logs folder, and each epoch will be saved once. If only a few steps are 
    trained, it will not be saved. The concepts of epoch and step should be clarified. During the training process, 
    the code does not set to save only the lowest loss, so after training with the default parameters, there will be
    100 weights. If the space is not enough, you can delete it yourself. This is not to save as little as possible, 
    nor to save as much as possible. Some people want to save them all, and some people want to save only a little bit. 
    In order to meet most needs, it is still highly optional to save them. 

3.  The size of the loss value is used to judge whether to converge or not. The more important thing is that there is a 
    trend of convergence, that is, the loss of the validation set continues to decrease. If the loss of the validation
    set basically does not change, the model basically converges. The specific size of the loss value does not make much
    sense. The big and small only depend on the calculation method of the loss, and it is not good to be close to 0. If
    you want to make the loss look better, you can directly divide 10000 into the corresponding loss function. The loss
    value during training will be saved in the loss_%Y_%m_%d_%H_%M_%S folder under the logs folder.

4.  Parameter tuning is a very important knowledge. No parameter is necessarily good. The existing parameters are the 
    parameters that I have tested and can be trained normally, so I would recommend using the existing parameters.
    But the parameters themselves are not absolute. For example, as the batch increases, the learning rate can also be
    increased, and the effect will be better; too deep networks should not use too large a learning rate, etc. These
    are all based on experience, you can only rely on the students to inquire more information and try it yourself.
"""


def _main():
    # =======================================================
    #   Be sure to modify classes_path before training so that it corresponds to your own dataset
    # =======================================================
    classes_path = PATH_CLASSES

    # =======================================================
    #   Anchors_path represents the txt file corresponding to the a priori box, which is generally not modified
    #   Anchors_mask is used to help the code find the corresponding a priori box and is generally not modified
    # =======================================================
    anchors_path = PATH_ANCHORS
    anchors_mask = YOLO_ANCHORS_MASK

    # =======================================================
    #   Please refer to the README for the download of the weight file, which can be downloaded from the network disk
    #   The pretrained weights of the model are common to different datasets because the features are common
    #   The more important part of the pre-training weight of the model is the weight part of the backbone feature
    #       extraction network, which is used for feature extraction.
    #
    #   Pre-training weights must be used in 99% of cases. If they are not used, the weights of the main part are too
    #       random, the feature extraction effect is not obvious, and the results of network training will not be good
    #   If there is an operation that interrupts the training during the training process, you can set the weight_path
    #       to the weights file in the logs folder, and reload the weights that have been trained
    #   At the same time, modify the parameters of the freeze phase or thaw phase below to ensure the continuity
    #       of the model epoch
    #
    #   When weight_path = '', the weights of the entire model are not loaded.
    #
    #   The weights of the entire model are used here, so they are loaded in train0.py
    #   If you want the model to start training from 0, set weight_path = '', the following freeze_body = False,
    #       then start training from 0, and there is no process of freezing the backbone
    #   Generally speaking, starting from 0 will have a poor training effect, because the weights are too random,
    #       and the feature extraction effect is not obvious
    #
    #   The network generally does not start training from 0, at least the weights of the backbone part are used
    #   Some papers mention that pre-training is not necessary
    #   The main reason is that their data set is large and their parameter adjustment ability is excellent
    #   If you must train the backbone part of the network, you can learn about the imagenet data set
    #   First, train the classification model. The backbone part of the classification model is common to the model,
    #       and training is based on this
    # =======================================================
    weight_path = PATH_WEIGHT

    # Check if weights file exists
    if weight_path and not os.path.exists(weight_path):
        print(f"\nError: Weights file not found at {weight_path}")
        print("\nPlease follow these steps:")
        print("1. Download YOLOv3 weights:")
        print("   - Visit https://pjreddie.com/media/files/yolov3.weights")
        print("   - Save the file to model_data/yolov3.weights")
        print("\n2. Convert the weights to Keras format:")
        print("   python convert.py model_data/yolov3.weights model_data/yolov3_ps.h5")
        print("\n3. Then run this training script again")
        return

    # =======================================================
    #   Directory to store the loss tracking and model weights
    # =======================================================
    log_dir = LOG_DIR
    log_dir2 = LOG_DIR2
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(log_dir2):
        os.makedirs(log_dir2)

    # =======================================================
    #   The size of the input shape must be a multiple of 32
    # =======================================================
    image_shape = IMAGE_SIZE

    # =======================================================
    #   Whether to freeze training, the default is to freeze the main training first and then unfreeze the training
    # =======================================================
    freeze_body = TRAIN_FREEZE_BODY

    # =======================================================
    #   The training is divided into two phases, the freezing phase and the thawing phase (when freeze_body is True)
    #   Insufficient video memory has nothing to do with the size of the data set
    #   If it indicates that the video memory is insufficient, please reduce the batch_size
    #   Affected by the BatchNorm layer, the minimum batch_size is 2 and cannot be 1
    # =======================================================
    # =======================================================
    #   Freeze phase training parameters
    #   At this time, the backbone of the model is frozen, and the feature extraction network does not change
    #   Occupy less memory, only fine-tune the network
    # =======================================================
    train_annot_paths = TRAIN_ANNOT_PATHS
    train_data_paths = DIR_TRAIN
    init_epoch = TRAIN_FREEZE_INIT_EPOCH
    freeze_end_epoch = TRAIN_FREEZE_END_EPOCH
    train_freeze_batch_size = TRAIN_FREEZE_BATCH_SIZE
    freeze_lr = TRAIN_FREEZE_LR

    # =======================================================
    #   Unfreeze phase training parameters
    #   At this time, the backbone of the model is not frozen, and the feature extraction network will change
    #   The occupied video memory is large, and all the parameters of the network will be changed
    # =======================================================
    unfreeze_end_epoch = TRAIN_UNFREEZE_END_EPOCH
    train_unfreeze_batch_size = TRAIN_UNFREEZE_BATCH_SIZE
    unfreeze_lr = TRAIN_UNFREEZE_LR

    # =======================================================
    # Validation parameters
    # =======================================================
    val_annot_paths = VAL_ANNOT_PATHS
    val_data_paths = DIR_VAL
    val_batch_size = VAL_BATCH_SIZE
    val_using = VAL_VALIDATION_USING
    val_split = VAL_VALIDATION_SPLIT
    assert os.path.exists(val_annot_paths[0]) or \
           (not os.path.exists(val_annot_paths[0]) and (val_using == "TRAIN" or val_using is None)), \
        'VAL_VALIDATION_USING can not be VAL on absence of validation data.'

    # =======================================================
    #   Get classes, anchors and threshold
    # =======================================================
    class_names, num_classes = get_classes(classes_path)
    anchors, num_anchors = get_anchors(anchors_path)
    loss_iou_thresh = YOLO_LOSS_IOU_THRESH

    # =======================================================
    #   Create a yolo model
    # =======================================================
    model_body = YOLOv3((None, None, 3), num_classes)
    print('Create YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    # =======================================================
    #   Load pretrained weights
    # =======================================================
    if weight_path != '':
        assert weight_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'
        print('Load weights {}.'.format(weight_path))
        model_body.load_weights(weight_path, by_name=True, skip_mismatch=True)

    # =======================================================
    #   Construct model with loss layer
    # =======================================================
    # Create input tensors for the model
    input_image = Input(shape=(None, None, 3))
    y_true = [Input(shape=(None, None, len(anchors_mask[l]), num_classes + 5)) for l in range(len(anchors_mask))]
    
    # Get model outputs
    model_outputs = model_body(input_image)
    
    # Create loss layer
    model_loss = Lambda(
        yolo_loss,
        output_shape=(1,),
        name='yolo_loss',
        arguments={
            'input_shape': image_shape,
            'anchors': anchors,
            'anchors_mask': anchors_mask,
            'num_classes': num_classes,
            'loss_iou_thresh': loss_iou_thresh
        }
    )([*model_outputs, *y_true])

    # Create the complete model
    model = Model([input_image, *y_true], model_loss)
    model.summary()

    # =======================================================
    #   Callbacks
    # =======================================================
    time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
    log_dir = os.path.join(log_dir, "loss_" + str(time_str))
    logging = TensorBoard(log_dir=log_dir)
    loss_history = LossHistory(log_dir)
    checkpoint_period = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                     monitor='val_loss', save_weights_only=False, period=1)
    reduce_lr = ExponentDecayScheduler(decay_rate=0.94, decay_steps=5, min_lr=1e-6, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)

    # =======================================================
    #   Freeze body
    # =======================================================
    if freeze_body:
        print('Freeze the first {} layers of total {} layers.'.format(184, len(model_body.layers)))
        for i in range(184): model_body.layers[i].trainable = False
        print('Freeze the first {} layers of total {} layers.'.format(184, len(model_body.layers)))
    else:
        print('Do not freeze any layers.')

    for i in range(len(model.layers)):
        model.layers[i].trainable = True
    model.compile(optimizer=Adam(learning_rate=freeze_lr), loss={
        'yolo_loss': lambda y_true, y_pred: y_pred
    })
    print('Unfreeze all layers.')

    # =======================================================
    #   Data loaders
    # =======================================================
    train_lines = read_lines(train_annot_paths, train_data_paths)
    if val_using == "VAL":
        val_lines = read_lines(val_annot_paths, val_data_paths)
    if val_using == "TRAIN":
        val_lines = random.sample(train_lines, int(len(train_lines) * val_split))
        train_lines = [line for line in train_lines if line not in val_lines]

    # Create data generators
    train_dataloader = YoloDataGenerator(train_lines, image_shape, anchors, train_freeze_batch_size,
                                       num_classes, anchors_mask, do_aug=False)
    if val_using == "VAL" or val_using == "TRAIN":
        val_dataloader = YoloDataGenerator(val_lines, image_shape, anchors, val_batch_size,
                                         num_classes, anchors_mask, do_aug=False)

    # Get number of samples
    num_train = len(train_lines)
    num_val = len(val_lines) if val_using in ["VAL", "TRAIN"] else 0

    batch_size = train_freeze_batch_size
    print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
    if val_using == "VAL":
        model.fit(
            train_dataloader, steps_per_epoch=train_dataloader.__len__(),
            validation_data=val_dataloader, validation_steps=val_dataloader.__len__(),
            initial_epoch=init_epoch, epochs=freeze_end_epoch,
            callbacks=[logging, checkpoint_period, reduce_lr, early_stopping, loss_history]
        )
    else:
        model.fit(
            train_dataloader, steps_per_epoch=train_dataloader.__len__(),
            initial_epoch=init_epoch, epochs=freeze_end_epoch,
            callbacks=[logging, checkpoint_period, reduce_lr, early_stopping, loss_history]
        )
    model.save(log_dir2 + 'trained_weights_stage_1.keras')

    # =======================================================
    #   Unfreeze phase training
    # =======================================================
    if freeze_body:
        for i in range(len(model_body.layers)):
            model_body.layers[i].trainable = True
        print('Unfreeze all layers.')

        for i in range(len(model.layers)):
            model.layers[i].trainable = True
        model.compile(optimizer=Adam(learning_rate=unfreeze_lr), loss={
            'yolo_loss': lambda y_true, y_pred: y_pred
        })
        print('Unfreeze and continue to train {} epochs.'.format(unfreeze_end_epoch))
        
        # Create data generators with new batch size
        train_dataloader = YoloDataGenerator(train_lines, image_shape, anchors, train_unfreeze_batch_size,
                                           num_classes, anchors_mask, do_aug=False)
        if val_using == "VAL" or val_using == "TRAIN":
            val_dataloader = YoloDataGenerator(val_lines, image_shape, anchors, val_batch_size,
                                             num_classes, anchors_mask, do_aug=False)

        batch_size = train_unfreeze_batch_size
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        if val_using == "VAL":
            model.fit(
                train_dataloader, steps_per_epoch=train_dataloader.__len__(),
                validation_data=val_dataloader, validation_steps=val_dataloader.__len__(),
                initial_epoch=freeze_end_epoch, epochs=unfreeze_end_epoch,
                callbacks=[logging, checkpoint_period, reduce_lr, early_stopping, loss_history]
            )
        else:
            model.fit(
                train_dataloader, steps_per_epoch=train_dataloader.__len__(),
                initial_epoch=freeze_end_epoch, epochs=unfreeze_end_epoch,
                callbacks=[logging, checkpoint_period, reduce_lr, early_stopping, loss_history]
            )
        model.save(log_dir2 + 'trained_weights_final.keras')


if __name__ == '__main__':
    _main()
