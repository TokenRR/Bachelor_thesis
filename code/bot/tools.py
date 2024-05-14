import os
import io
import requests
from PIL import Image
from typing import Dict

import telebot

from config import TOKEN


def change_working_directory_to_current_file():
    """
    Змінює поточну директорію на директорію файлу, який виконується
    """

    current_file_path = os.path.realpath(__file__)  # Отримуємо шлях до файлу, який виконується
    current_directory = os.path.dirname(current_file_path)  # Отримуємо директорію файлу
    os.chdir(current_directory)  # Змінюємо поточну директорію на директорію файлу


def download_image(bot, message):
    """
    Отримання зображення як файлу або стисненого фото
    """

    try:
        # Спробуємо отримати зображення як файл
        file_info = bot.get_file(message.document.file_id)
        file = requests.get('https://api.telegram.org/file/bot{0}/{1}'.format(TOKEN, file_info.file_path))
        file_format = file_info.file_path.split('.')[-1].lower()  # Визначаємо формат з розширення файлу
        # print(f'[INFO]  File format: {file_format}')
                
        valid_formats = ['png', 'jpg', 'jpeg']
        if file_format in valid_formats:
            HR_image = Image.open(io.BytesIO(file.content)).convert('RGB')
            return HR_image
        else:
            bot.send_message(message.chat.id, 'Будь ласка, надішліть зображення у форматі PNG, JPG або JPEG')
    except Exception as e:
        pass

    try:
        # Якщо не вдалося отримати зображення як файл, спробуємо отримати його як стиснене фото
        file_info = bot.get_file(message.photo[-1].file_id)
        file = requests.get('https://api.telegram.org/file/bot{0}/{1}'.format(TOKEN, file_info.file_path))
        HR_image = Image.open(io.BytesIO(file.content)).convert('RGB')
        return HR_image
    except Exception as e:
        pass


def add_cancel_button(bot, message, callback_data):
    """
    Додавання кнопки для повернення назад
    """

    markup = telebot.types.InlineKeyboardMarkup()
    itembtn1 = telebot.types.InlineKeyboardButton('Скасувати', callback_data=callback_data)
    markup.add(itembtn1)
    button_caption = 'Натисніть кнопку \'Скасувати\', щоб скасувати надсилання зображення',
    button = bot.send_message(message.chat.id, button_caption, reply_markup=markup)
    return button


def get_messages(filename: str) -> Dict[str, str]:
    """
    A function to read text messages from a file and convert them to a dictionary.

    Parameters:
    filename (str): The path to the file containing the messages. Each message in the file must be separated by '==='.

    Returns:
    dict: A dictionary where the keys are the names of the messages and the values are the corresponding text strings. 
          All newline characters are replaced with spaces in each message.

    Example:
    ===help_message===
    Це інструкція до бота
    Help func

    ===theory_message===
    Це теоретичні відомості

    ===empty_message===

    Then the function will return:
    {'help_message': 'This is an instruction for the Help func bot', 
     'theory_message': 'This is theory information', 
     'empty_message': ''}
    """

    change_working_directory_to_current_file()
    with open(filename, 'r', encoding='utf-8') as file:
        content = file.read()
        messages = content.split('===')[1:]
        messages_dict = {messages[i].strip(): messages[i + 1].replace('\n', '').replace('  ', '\n').strip() 
                         for i in range(0, len(messages) - 1, 2)}
    return messages_dict

