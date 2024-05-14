import io
import os
import requests

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import telebot
import tensorflow as tf
from PIL import Image
from keras.preprocessing.image import img_to_array
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset

from tools import download_image
from tools import change_working_directory_to_current_file


def plot_results(img, prefix, title, z1=200, z2=300, z3=150, z4=250):
    """
    Функція для побудови графіків зображень і збереження їх у файл.
    
    Параметри:
    img (np.array): Вхідне зображення.
    prefix (str): Префікс для імені файлу зображення.
    title (str): Заголовок графіка.
    z1, z2, z3, z4 (int, optional): Координати для області збільшення. За замовчуванням 200, 300, 150, 250.
    """

    # Зображення готуються до побудови графіків шляхом перетворення їх у масиви та масштабування їхніх значень
    img_array = img_to_array(img)
    img_array = img_array.astype('float32') / 255.0

    # Для вихідного зображення та збільшеного зображення створюється підзображення
    _figure, parent = plt.subplots()

    # Причина, чому ми хочемо побудувати малюнок у зворотному порядку, полягає в тому, що коли ми використовуємо
    # img_to_array, розташування висоти і ширини інвертується. Параметр «origin» задає розташування початкової 
    # точки (0, 0). У цьому випадку - лівий нижній кут
    parent.imshow(img_array[::-1], origin='lower')
    # plt.yticks(visible=False)
    # plt.xticks(visible=False)     
    plt.title(title)

    # Визначте осі вставки на основі батьківських осей і вкажіть значення масштабування (2x, 3x і т.д.)
    # Ми також вказуємо місце для зображення зі зміненим масштабом, у цьому випадку - верхній лівий кут
    inset = zoomed_inset_axes(parent, 2, loc='upper left')
    inset.imshow(img_array[::-1], origin='lower')

    x1, x2, y1, y2 = z1, z2, z3, z4
    inset.set_xlim(x1, x2)  #  Вкажіть координати по осі X для масштабування зображення
    inset.set_ylim(y1, y2)  #  Вкажіть координати по осі Y для масштабування зображення
    plt.yticks(visible=False)
    plt.xticks(visible=False)

    # Намалюйте додаткові лінії, що вказують на місце розташування збільшеного зображення
    # «loc1» і «loc2» - кути осей вставки, «fc» і «ec» - кольори ліній
    mark_inset(parent, inset, loc1=1, loc2=3, fc='none', ec='blue')
    plt.savefig(f'images\\{prefix}.jpg')  # Збереження графіка
    plt.close()


def comparison(bot, model, button_message_id, message):
    """
    Візуальне порівняння зображення до та після покращення
    """

    change_working_directory_to_current_file()
    try:
        HR_image = download_image(bot, message)

        # delete 'Скасувати' button after image is downloaded
        bot.delete_message(chat_id=message.chat.id, message_id=button_message_id)

        # Обробка зображення
        upscale_factor = 3
        LR_image = HR_image.resize((HR_image.size[0] // upscale_factor,
                                    HR_image.size[1] // upscale_factor),
                                    Image.BICUBIC)

        ycbcr = LR_image.convert('YCbCr')
        y, cb, cr = ycbcr.split()
        y = img_to_array(y)
        y = y.astype('float32') / 255.0

        model_input = y.reshape(1, y.shape[0], y.shape[1], y.shape[2])

        output = model.predict(model_input)
        output = output[0]
        output *= 255.0
        output = output.clip(0, 255)
        output = output.reshape((output.shape[0], output.shape[1]))
        output = Image.fromarray(np.uint8(output))
        output = output.resize(HR_image.size, Image.Resampling.NEAREST)

        cb = cb.resize(output.size, Image.Resampling.BICUBIC)
        cr = cr.resize(output.size, Image.Resampling.BICUBIC)

        ER_image = Image.merge('YCbCr', (output, cb, cr))
        ER_image = ER_image.convert('RGB')

        LR_image = LR_image.resize(ER_image.size, Image.Resampling.BICUBIC)

        # Візуалізація результатів
        z = {1: [150, 300, 230, 330], 3: [280, 380, 150, 250], 4: [200, 300, 180, 280],  5: [200, 330, 175, 300]}
        index = 0  # Виберіть індекс зі словника 'z'

        if index in z.keys():
            z1, z2, z3, z4 = z[index]
        else:
            z1, z2, z3, z4 = 200, 300, 150, 250

        plot_results(LR_image, 'LR_image', 'Low Resolution', z1, z2, z3, z4)
        plot_results(ER_image, 'ER_image', 'Enhanced Resolution', z1, z2, z3, z4)
        plot_results(HR_image, 'HR_image', 'Original Image', z1, z2, z3, z4)

        # Розрахунок точності
        LR_arr = img_to_array(LR_image)
        HR_arr = img_to_array(HR_image)
        ER_arr = img_to_array(ER_image)

        bicubic_psnr = tf.image.psnr(LR_arr, HR_arr, max_val=255)
        test_psnr = tf.image.psnr(ER_arr, HR_arr, max_val=255)

        LR_HR = f'PSNR між LR та HR зображеннями = {bicubic_psnr:.3f}\n'
        ER_HR = f'PSNR між ER та HR зображеннями = {test_psnr:.3f}\n'
        difference = f'Різниця PSNR = {(test_psnr - bicubic_psnr):.3}'
        caption = LR_HR + ER_HR + difference

        # Відправлення зображень
        with open('images\\LR_image.jpg', 'rb') as LR_image, \
             open('images\\ER_image.jpg', 'rb') as ER_image, \
             open('images\\HR_image.jpg', 'rb') as HR_image:
            media = [telebot.types.InputMediaPhoto(LR_image, caption=caption),
                     telebot.types.InputMediaPhoto(ER_image),
                     telebot.types.InputMediaPhoto(HR_image)]
            bot.send_media_group(message.chat.id, media, reply_to_message_id=message.message_id)
    except Exception as e:
        msg = 'На жаль, сталася помилка \nСпробуйте викликати команду /comparison ще раз'
        bot.send_message(message.chat.id, msg)
