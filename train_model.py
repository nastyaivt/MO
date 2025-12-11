# train_model.py - Обучение и сохранение модели
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import os


def main():
    print("Обучение модели CNN для классификации рукописных цифр (MNIST)")

    # 1. Загрузка данных
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # 2. Подготовка данных
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    # 3. Создание модели CNN
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # 4. Обучение
    print("Обучение модели...")
    model.fit(x_train, y_train,
              epochs=10,
              batch_size=64,
              validation_split=0.2,
              verbose=1)

    # 5. Сохранение
    if not os.path.exists('models'):
        os.makedirs('models')
    model.save('models/mnist_cnn_model.h5')
    print("Модель сохранена в models/mnist_cnn_model.h5")

    # 6. Оценка
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f"Точность на тестовых данных: {test_acc:.4f}")


if __name__ == "__main__":
    main()