from PIL import Image

import numpy as np
from skimage.metrics import peak_signal_noise_ratio
import telebot

from tools import change_working_directory_to_current_file
from tools import download_image


def degrade_image(bot, button_message_id, message):
    """
    Функція погіршення якості зображення
    """

    change_working_directory_to_current_file()
    try:
        # Завантаження зображення
        HR_image = download_image(bot, message)

        # delete 'Скасувати' button after image is downloaded
        bot.delete_message(chat_id=message.chat.id, message_id=button_message_id)

        # Зменшення розміру зображення
        small_image = HR_image.resize((HR_image.size[0]//3, HR_image.size[1]//3), Image.Resampling.BICUBIC)

        # Збільшення розміру зображення назад до оригінального розміру
        LR_image = small_image.resize(HR_image.size, Image.Resampling.BICUBIC)

        # Збереження зображень
        HR_image.save('images\\HR_image.jpg')
        LR_image.save('images\\LR_image.jpg')

        # Конвертація зображень в масиви NumPy
        HR_image = np.array(HR_image)
        LR_image = np.array(LR_image)

        psnr = peak_signal_noise_ratio(HR_image, LR_image)

        # Відправлення зображень
        caption = f'Peak Signal-to-Noise Ratio = {psnr:.3}'
        with open('images\\HR_image.jpg', 'rb') as HR_image, open('images\\LR_image.jpg', 'rb') as LR_image:
            media = [telebot.types.InputMediaPhoto(HR_image, caption=caption),
                     telebot.types.InputMediaPhoto(LR_image),]
            bot.send_media_group(message.chat.id, media, reply_to_message_id=message.message_id)
    except Exception as e:
        msg = 'На жаль, не вдалось погіршити якість Вашого зображення \nСпробуйте викликати команду /degrade ще раз'
        bot.send_message(message.chat.id, msg)
