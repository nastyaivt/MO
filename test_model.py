# test_model.py - Тестирование загруженной модели
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from PIL import Image
import os


def load_and_test_image(model, image_path):
    """Загрузка и тестирование пользовательского изображения"""
    try:
        img = Image.open(image_path).convert('L')
        img = img.resize((28, 28))
        img_array = np.array(img)

        # Инверсия при необходимости
        if np.mean(img_array) < 128:
            img_array = 255 - img_array

        img_array = img_array.astype("float32") / 255
        img_array = np.expand_dims(img_array, -1)
        img_array = np.expand_dims(img_array, 0)

        predictions = model.predict(img_array, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = np.max(predictions[0])

        return predicted_class, confidence, img_array[0, :, :, 0]
    except Exception as e:
        print(f"Ошибка: {e}")
        return None, None, None


def main():
    print("Тестирование модели CNN для классификации рукописных цифр")

    # Проверка существования модели
    if not os.path.exists('models/mnist_cnn_model.h5'):
        print("Ошибка: Модель не найдена!")
        print("Сначала выполните train_model.py для обучения модели")
        return

    # Загрузка модели
    print("Загрузка модели...")
    model = keras.models.load_model('models/mnist_cnn_model.h5')
    print("Модель загружена успешно!")

    while True:
        print("\n--- МЕНЮ ТЕСТИРОВАНИЯ ---")
        print("1. Протестировать на случайной цифре из MNIST")
        print("2. Загрузить свое изображение")
        print("3. Оценить точность на тестовом наборе")
        print("4. Выход")

        choice = input("Выберите действие (1-4): ")

        if choice == '1':
            # Тест на случайном изображении из MNIST
            (_, _), (x_test, y_test) = keras.datasets.mnist.load_data()
            idx = np.random.randint(0, len(x_test))
            test_image = x_test[idx]
            test_label = y_test[idx]

            # Подготовка и предсказание
            img_array = test_image.astype("float32") / 255
            img_array = np.expand_dims(img_array, -1)
            img_array = np.expand_dims(img_array, 0)

            predictions = model.predict(img_array, verbose=0)
            predicted_class = np.argmax(predictions[0])
            confidence = np.max(predictions[0])

            print(f"\nРезультат:")
            print(f"Истинная цифра: {test_label}")
            print(f"Предсказанная цифра: {predicted_class}")
            print(f"Уверенность: {confidence:.2%}")

            # Визуализация
            plt.figure(figsize=(6, 3))
            plt.subplot(1, 2, 1)
            plt.imshow(test_image, cmap='gray')
            plt.title(f'True: {test_label}')
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.bar(range(10), predictions[0])
            plt.title(f'Pred: {predicted_class}')
            plt.xlabel('Цифра')
            plt.ylabel('Вероятность')
            plt.show()

        elif choice == '2':
            # Тестирование пользовательского изображения
            image_path = input("Введите путь к изображению: ")

            if os.path.exists(image_path):
                predicted_class, confidence, processed_image = load_and_test_image(model, image_path)

                if predicted_class is not None:
                    print(f"\nРезультат:")
                    print(f"Предсказанная цифра: {predicted_class}")
                    print(f"Уверенность: {confidence:.2%}")

                    plt.figure(figsize=(6, 6))
                    plt.imshow(processed_image, cmap='gray')
                    plt.title(f'Predicted: {predicted_class}, Confidence: {confidence:.2%}')
                    plt.axis('off')
                    plt.show()
            else:
                print("Файл не найден!")

        elif choice == '3':
            # Полная оценка точности
            (_, _), (x_test, y_test) = keras.datasets.mnist.load_data()
            x_test = x_test.astype("float32") / 255
            x_test = np.expand_dims(x_test, -1)
            y_test = keras.utils.to_categorical(y_test, 10)

            test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
            print(f"\nТочность на тестовом наборе: {test_acc:.4f}")
            print(f"Потери на тестовом наборе: {test_loss:.4f}")

        elif choice == '4':
            print("Выход из программы тестирования.")
            break

        else:
            print("Неверный выбор!")


if __name__ == "__main__":
    main()