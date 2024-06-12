import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
import os
import numpy as np
import cv2

def u2net(input_shape=(256, 256, 3)):
    def RSU7(x, mid_channels, out_channels):
        # RSU7: Residual U-Block 7
        hx = x
        hxin = layers.Conv2D(out_channels, 3, padding='same')(x)

        hx1 = layers.Conv2D(mid_channels, 3, padding='same')(hxin)
        hx1 = layers.ReLU()(hx1)
        hx1 = layers.Conv2D(mid_channels, 3, padding='same')(hx1)
        hx1 = layers.ReLU()(hx1)

        # Match the number of channels
        if hxin.shape[-1] != hx1.shape[-1]:
            hxin = layers.Conv2D(mid_channels, 1, padding='same')(hxin)

        hx = hx1 + hxin
        return hx

    inputs = layers.Input(shape=input_shape)

    # Stage 1
    hx1 = RSU7(inputs, 32, 64)

    # Downsample
    hx = layers.MaxPooling2D(pool_size=(2, 2))(hx1)

    # Stage 2
    hx2 = RSU7(hx, 64, 128)

    # Downsample
    hx = layers.MaxPooling2D(pool_size=(2, 2))(hx2)

    # Stage 3
    hx3 = RSU7(hx, 128, 256)

    # Downsample
    hx = layers.MaxPooling2D(pool_size=(2, 2))(hx3)

    # Bottleneck
    hx4 = RSU7(hx, 256, 512)

    # Upsample
    hx = layers.UpSampling2D(size=(2, 2))(hx4)

    # Stage 3d
    hx = RSU7(hx, 128, 256)
    hx = hx + hx3

    # Upsample
    hx = layers.UpSampling2D(size=(2, 2))(hx)

    # Stage 2d
    hx = RSU7(hx, 64, 128)
    hx = hx + hx2

    # Upsample
    hx = layers.UpSampling2D(size=(2, 2))(hx)

    # Stage 1d
    hx = RSU7(hx, 32, 64)
    hx = hx + hx1

    # Output
    outputs = layers.Conv2D(2, 1, activation='softmax')(hx)

    model = models.Model(inputs, outputs)
    return model

def load_data(image_dir, mask_dir, img_size=(256, 256)):
    images = []
    masks = []

    for img_file in os.listdir(image_dir):
        if img_file.endswith(".png"):
            img_path = os.path.join(image_dir, img_file)
            mask_path = os.path.join(mask_dir, img_file.replace(".png", "_label.png"))

            image = cv2.imread(img_path)
            image = cv2.resize(image, img_size)
            image = image / 255.0
            images.append(image)

            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, img_size)
            # Ensure mask values are in the range [0, 1]
            mask = mask // 255
            mask = to_categorical(mask, num_classes=2)
            masks.append(mask)

    return np.array(images), np.array(masks)

# Directory paths
image_dir = 'synthetic_frog_eggs/processed/images'
mask_dir = 'synthetic_frog_eggs/processed/masks'
images, masks = load_data(image_dir, mask_dir)

# Split data into training and validation sets
split_idx = int(0.8 * len(images))
train_images, train_masks = images[:split_idx], masks[:split_idx]
val_images, val_masks = images[split_idx:], masks[split_idx:]

# Instantiate and compile the model
model = u2net(input_shape=(256, 256, 3))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Define the callback to save the best model
checkpoint_callback = callbacks.ModelCheckpoint(
    'best_model.keras',
    monitor='val_loss',
    save_best_only=True,
    save_weights_only=False,
    mode='min',
    verbose=1
)

# Train the model with the callback
history = model.fit(
    train_images, train_masks,
    epochs=50,
    batch_size=32,
    validation_data=(val_images, val_masks),
    callbacks=[checkpoint_callback]
)

# Print training and validation accuracies and losses
import matplotlib.pyplot as plt

def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))

    plt.plot(epochs, acc, 'r', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend(loc=0)
    plt.figure()

    plt.plot(epochs, loss, 'r', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend(loc=0)
    plt.figure()

    plt.show()

plot_history(history)

