import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import tf_keras as keras
from tf_keras import layers, models

def train_and_save_model():
    dataset_path = 'dataset' 
    ordered_classes = ['a', 'ae', 'e', 'u', 's', 'r', 'v']
    
    print("Loading dataset...")
    dataset = keras.utils.image_dataset_from_directory(
        dataset_path,
        labels='inferred',
        label_mode='int',
        class_names=ordered_classes, 
        color_mode='grayscale',
        batch_size=32,
        image_size=(28, 28),
        shuffle=True,
        validation_split=0.2,
        subset='both',
        seed=123
    )
    
    train_ds = dataset[0]
    val_ds = dataset[1]

    normalization_layer = layers.Rescaling(1./255)
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

    data_augmentation = keras.Sequential([
        layers.RandomRotation(factor=0.08),       
        layers.RandomTranslation(0.1, 0.1),       
        layers.RandomZoom(height_factor=0.1)      
    ])

    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),
        data_augmentation,
        
        layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5), 
        layers.Dense(len(ordered_classes), activation='softmax')
    ])

   
    optimizer = keras.optimizers.Adam(learning_rate=0.0005)
    
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    print("Training upgraded deep model...")
    
    model.fit(train_ds, epochs=100, validation_data=val_ds) 

    model.save('drawing_model.keras')
    print("Upgraded model saved successfully!")

if __name__ == "__main__":
    train_and_save_model()