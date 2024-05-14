import numpy as np

from PIL import Image
from skimage.metrics import peak_signal_noise_ratio
import telebot

from tools import change_working_directory_to_current_file
from tools import download_image


def upgrade_image(bot, model, button_message_id, message):
    """
    Функція покращення якості зображення
    """

    change_working_directory_to_current_file()
    try:
        # Завантаження зображення
        LR_image = download_image(bot, message)

        # delete 'Скасувати' button after image is downloaded
        bot.delete_message(chat_id=message.chat.id, message_id=button_message_id)

        # Конвертація зображення в YCbCr
        ycbcr = LR_image.convert('YCbCr')
        y, cb, cr = ycbcr.split()

        # Підготовка каналу Y для моделі
        y = np.array(y).astype('float32') / 255.0
        model_input = np.expand_dims(y, axis=[0, 3])

        output = model.predict(model_input)  #  Передбачення моделі

        # Пост-обробка вихідних даних
        output *= 255.0
        output = output.clip(0, 255)
        output = output.reshape((output.shape[1], output.shape[2]))
        output = Image.fromarray(np.uint8(output))

        # Змінюємо розмір каналів Cb і Cr до розміру ER
        cb = cb.resize(output.size, Image.Resampling.BICUBIC)
        cr = cr.resize(output.size, Image.Resampling.BICUBIC)
        ER_image = Image.merge('YCbCr', (output, cb, cr)).convert('RGB')  #  Об'єднання каналів Y, Cb, Cr

        # Зменшення розміру покращеного зображення до розміру оригінального зображення
        ER_image = ER_image.resize(LR_image.size, Image.Resampling.BICUBIC)

        # Збереження зображень
        LR_image.save('images\\LR_image.jpg')
        ER_image.save('images\\ER_image.jpg')

        # Конвертація зображень в масиви NumPy
        LR_image = np.array(LR_image)
        ER_image = np.array(ER_image)

        psnr = peak_signal_noise_ratio(LR_image, ER_image)

        caption = f'Peak Signal-to-Noise Ratio = {psnr:.3}'
        with open('images\\LR_image.jpg', 'rb') as LR_image, open('images\\ER_image.jpg', 'rb') as ER_image:
            media = [telebot.types.InputMediaPhoto(LR_image, caption=caption),
                     telebot.types.InputMediaPhoto(ER_image),]
            bot.send_media_group(message.chat.id, media, reply_to_message_id=message.message_id)
    except:
        msg = 'На жаль, не вдалось покращити якість Вашого зображення \nСпробуйте викликати команду /upgrade ще раз'
        bot.send_message(message.chat.id, msg)
