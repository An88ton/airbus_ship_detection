import tensorflow as tf

import os
import tensorflow as tf

from tensorflow_examples.models.pix2pix import pix2pix
import matplotlib.pyplot as plt
from IPython.display import clear_output


BATCH_SIZE = 32
BUFFER_SIZE = 1000
STEPS_PER_EPOCH = 120
RESERVED_FOR_TRAIN = 2000


image_folder_path  = './airbus-ship-detection/train_v2'
mask_folder_path  = './airbus-ship-detection/masks_v2'

def normalize(input_image, input_mask):
  input_image = tf.cast(input_image, tf.float32) / 255.0
  input_mask = tf.cast(input_mask, tf.float32) / 255.0
  return input_image, input_mask

def load_image(img, mask):
  input_image = tf.image.resize(img, (256, 256))
  input_mask = tf.image.resize(
    mask,
    (256, 256),
    method = tf.image.ResizeMethod.NEAREST_NEIGHBOR,
  )
  input_image, input_mask = normalize(input_image, input_mask)
  return input_image, input_mask

class Augment(tf.keras.layers.Layer):
    def __init__(self, seed=42):
        super().__init__()
        # both use the same seed, so they'll make the same random changes.
        self.augment_inputs = tf.keras.layers.RandomFlip(mode="horizontal", seed=seed)
        self.augment_labels = tf.keras.layers.RandomFlip(mode="horizontal", seed=seed)

    def call(self, inputs, labels):
        inputs = self.augment_inputs(inputs)
        labels = self.augment_labels(labels)
        return inputs, labels

def display(display_list):
  plt.figure(figsize=(15, 15))

  title = ['Input Image', 'True Mask', 'Predicted Mask']

  for i in range(len(display_list)):
    plt.subplot(1, len(display_list), i+1)
    plt.title(title[i])
    plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
    plt.axis('off')
  plt.show()

# Loading images and masks
img_ds = tf.keras.utils.image_dataset_from_directory(image_folder_path, labels=None, shuffle=None)
segmentation_mask_ds = tf.keras.utils.image_dataset_from_directory(mask_folder_path, labels=None, shuffle=None, color_mode='grayscale')  # Assumes the filenames are in the same order
# Combine to one dataset, data will be in tuples (img, mask)
ds = tf.data.Dataset.zip((img_ds, segmentation_mask_ds))

ds.batch(BATCH_SIZE)
train_images = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)

train_batches = (
    train_images
    .cache()
    # .shuffle(BUFFER_SIZE)
    # .batch(BATCH_SIZE) batch was setted above, if we call it here it will breake shape of dataset
    .repeat()
    .map(Augment())
    .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

# Split dataset because original data for test was corrupted
test_batches = train_batches.take(RESERVED_FOR_TRAIN).cache()
train_batches = train_batches.skip(RESERVED_FOR_TRAIN)



for images, masks in train_batches.take(2):
  sample_image, sample_mask = images[0], masks[0]
  display([sample_image, sample_mask])

# Train model
base_model = tf.keras.applications.MobileNetV2(input_shape=[256, 256, 3], include_top=False)


layer_names = [
    'block_1_expand_relu',   # 64x64
    'block_3_expand_relu',   # 32x32
    'block_6_expand_relu',   # 16x16
    'block_13_expand_relu',  # 8x8
    'block_16_project',      # 4x4
]
base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

# Create the feature extraction model
down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)

down_stack.trainable = False

up_stack = [
    pix2pix.upsample(512, 3),  # 4x4 -> 8x8
    pix2pix.upsample(256, 3),  # 8x8 -> 16x16
    pix2pix.upsample(128, 3),  # 16x16 -> 32x32
    pix2pix.upsample(64, 3),   # 32x32 -> 64x64
]

def unet_model(output_channels:int):
  inputs = tf.keras.layers.Input(shape=[256, 256, 3])

  # Downsampling through the model
  skips = down_stack(inputs)
  x = skips[-1]
  skips = reversed(skips[:-1])

  # Upsampling and establishing the skip connections
  for up, skip in zip(up_stack, skips):
    x = up(x)
    concat = tf.keras.layers.Concatenate()
    x = concat([x, skip])

  # This is the last layer of the model
  last = tf.keras.layers.Conv2DTranspose(
      filters=output_channels, kernel_size=3, strides=2,
      padding='same')  #64x64 -> 128x128

  x = last(x)

  return tf.keras.Model(inputs=inputs, outputs=x)

OUTPUT_CLASSES = 3

model = unet_model(output_channels=OUTPUT_CLASSES)
model.compile(optimizer= 'adam', #tf.keras.optimizers.Adam(learning_rate=0.01),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

tf.keras.utils.plot_model(model, show_shapes=True)

def create_mask(pred_mask):
  pred_mask = tf.math.argmax(pred_mask, axis=-1)
  pred_mask = pred_mask[..., tf.newaxis]
  return pred_mask[0]

def show_predictions(dataset=None, num=1):
  if dataset:
    for image, mask in dataset.take(num):
      pred_mask = model.predict(image)
      display([image[0], mask[0], create_mask(pred_mask)])
  else:
    display([sample_image, sample_mask,
             create_mask(model.predict(sample_image[tf.newaxis, ...]))])


class DisplayCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    clear_output(wait=True)
    show_predictions()
    print ('\nSample Prediction after epoch {}\n'.format(epoch+1))

EPOCHS = 20

model_history = model.fit(train_batches, epochs=EPOCHS,
                          steps_per_epoch=STEPS_PER_EPOCH,
                          validation_data=test_batches,
                          callbacks=[DisplayCallback()])

# label = [0,0]
# prediction = [[-3., 0], [-3, 0]]
# sample_weight = [1, 10]
#
# loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
#                                                reduction=tf.keras.losses.Reduction.NONE)
# loss(label, prediction, sample_weight).numpy()
#
# def add_sample_weights(image, label):
#   # The weights for each class, with the constraint that:
#
#   class_weights = tf.constant([3.0, 1.0, 1.0])
#   # print(sum(class_weights) == 1.0)
#   class_weights = class_weights/tf.reduce_sum(class_weights)
#
#   # Create an image of `sample_weights` by using the label at each pixel as an
#   # index into the `class weights` .
#   sample_weights = tf.gather(class_weights, indices=tf.cast(label, tf.int32))
#
#   return image, label, sample_weights
#
# train_batches.map(add_sample_weights).element_spec
#
# weighted_model = unet_model(OUTPUT_CLASSES)
# weighted_model.compile(
#     optimizer='adam',
#     loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#     metrics=['accuracy'])
#
# weighted_model.fit(
#     train_batches.map(add_sample_weights),
#     epochs=10,
#     steps_per_epoch=10,
#     validation_data=test_batches,
#     callbacks=[DisplayCallback()]
# )
