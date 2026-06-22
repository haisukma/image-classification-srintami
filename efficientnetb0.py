import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.applications.efficientnet import preprocess_input
import zipfile
import os
import seaborn as sns
from PIL import Image
import os
import zipfile

def training_model(data_path):

    img_size = 256
    batch_size = 16
    epochs = 100
    learning_rate = 0.00005
    seed = 123

    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_path,
        validation_split=0.3,
        subset="training",
        seed=seed,
        image_size=(img_size, img_size),
        batch_size=batch_size
    )

    temp_ds = tf.keras.utils.image_dataset_from_directory(
        data_path,
        validation_split=0.3,
        subset="validation",
        seed=seed,
        image_size=(img_size, img_size),
        batch_size=batch_size
    )

    class_names = train_ds.class_names

    print(class_names)

    temp_batches = tf.data.experimental.cardinality(temp_ds)
    val_size = temp_batches // 2
    val_ds = temp_ds.take(val_size)
    test_ds = temp_ds.skip(val_size)

    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.03),
        tf.keras.layers.RandomZoom(0.05),
        tf.keras.layers.RandomBrightness(0.1),
        # tf.keras.layers.RandomTranslation(0.1, 0.1)
    ])

    train_ds = train_ds.map(
        lambda x, y: (
            data_augmentation(x, training=True),
            y
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    Autotune = tf.data.AUTOTUNE

    train_ds = (
        train_ds
        .cache()
        .shuffle(1000)
        .prefetch(buffer_size=Autotune)
    )

    val_ds = (
        val_ds
        .cache()
        .prefetch(buffer_size=Autotune)
    )

    test_ds = (
        test_ds
        .cache()
        .prefetch(buffer_size=Autotune)
    )

    base_model = tf.keras.applications.EfficientNetB0(input_shape=(img_size, img_size, 3),
                                                    include_top=False,
                                                    weights='imagenet')

    base_model.trainable = False

    for layer in base_model.layers[-20:]:
        layer.trainable = True

    model = tf.keras.Sequential([
        tf.keras.Input(shape=(img_size, img_size, 3)),
        # data_augmentation,
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(8, activation='softmax')
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile (
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )

    model.summary()

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[early_stopping]
    )

    test_loss, test_accuracy = model.evaluate(test_ds)
    print(f'\nTest Loss: {test_loss:.4f}')
    print(f'Test Accuracy: {test_accuracy:.4f}')

    y_pred_raw = []
    y_true = []

    for images, labels in test_ds:
        predictions = model.predict(images)
        y_pred_raw.extend(np.argmax(predictions, axis=1))
        y_true.extend(labels.numpy())

    y_pred = np.array(y_pred_raw)
    y_true = np.array(y_true)

    print('\nClassification Report:')
    print(classification_report(y_true, y_pred, target_names=class_names))

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

    model.export("saved_model")

    return{
        "accuracy": test_accuracy,
        "loss": test_loss
    }

if __name__ == "__main__":
    training_model("new-dataset6")