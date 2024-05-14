from functools import partial

import telebot
from keras.models import load_model

from config import TOKEN
from zooming import comparison
from image_enhancement import upgrade_image
from image_degradement import degrade_image
from tools import add_cancel_button, get_messages, change_working_directory_to_current_file


bot = telebot.TeleBot(TOKEN)
bot.set_my_commands([
    telebot.types.BotCommand('/start', 'Привітання'),
    telebot.types.BotCommand('/help', 'Допомога (документація)'),
    telebot.types.BotCommand('/theory', 'Теоретичні відомості'),
    telebot.types.BotCommand('/comparison', 'Візуальне порівняння роботи моделі'),
    telebot.types.BotCommand('/degrade', 'Погіршити якість фотографії'),
    telebot.types.BotCommand('/upgrade', 'Покращити якість фотографії'),
])

change_working_directory_to_current_file()
model = load_model('models\\model.h5')


@bot.message_handler(commands=['start'])
def handle_start(message):
    bot.send_message(message.chat.id, 'Привіт! Я частина навчального проекту студента КПІ, КМ-01, Романецького Микити')


@bot.message_handler(commands=['help'])
def handle_help(message):
    help_message = get_messages('messages.txt')['help_message']
    bot.reply_to(message, help_message)


@bot.message_handler(commands=['theory'])
def handle_theory(message):
    theory_message = get_messages('messages.txt')['theory_message']
    bot.reply_to(message, theory_message)


@bot.message_handler(commands=['comparison'])
def handle_comparison(message):
    msg = bot.reply_to(message, 'Надішліть зображення, яке я маю покращити \nБажано без стиснення (файлом)')
    button = add_cancel_button(bot, message, 'cancel_comparison')
    bot.register_next_step_handler(msg, partial(comparison, bot, model, button.message_id))


@bot.message_handler(commands=['upgrade'])
def handle_upgrade(message):
    msg = bot.reply_to(message, 'Надішліть зображення, яке я маю покращити \nБажано без стиснення (файлом)')
    button = add_cancel_button(bot, message, 'cancel_upgrade')
    bot.register_next_step_handler(msg, partial(upgrade_image, bot, model, button.message_id))


@bot.message_handler(commands=['degrade'])
def handle_degrade(message):
    msg = bot.reply_to(message, 'Надішліть зображення, яке я маю погіршити \nБажано без стиснення (файлом)')
    button = add_cancel_button(bot, message, 'cancel_degrade')
    bot.register_next_step_handler(msg, partial(degrade_image, bot, button.message_id))


@bot.callback_query_handler(func=lambda call: True)
def callback_query(call):
    if call.data in ['cancel_comparison', 'cancel_upgrade', 'cancel_degrade']:
        bot.answer_callback_query(call.id, 'Ви скасували надсилання зображення')
        bot.delete_message(chat_id=call.message.chat.id, message_id=call.message.message_id)
        bot.clear_step_handler_by_chat_id(call.message.chat.id)


@bot.message_handler(func=lambda message: True)
def echo_all(message):
    bot.reply_to(message, 'Я розумію лише команди')


if __name__ == '__main__':
    bot.polling(none_stop=True)
