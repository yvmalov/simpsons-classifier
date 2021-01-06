#!/usr/bin/python
# -*- coding: utf8 -*-

#
# Author: Yuriy Malov. https://t.me/ymalov
# Source skeleton: https://towardsdatascience.com/how-to-deploy-a-telegram-bot-using-heroku-for-free-9436f89575d2
#

# import packages
import os, io, logging, requests, time, json, random, wget, torch
import torch.optim as optim
import torchvision.models
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters

PORT = int(os.environ.get('PORT', 80))

# Enable logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)

logger = logging.getLogger(__name__)

# Tokens is saved on the Heroku. HerokuApp -> Settings ->  Config Vars
TOKEN = os.environ.get('TOKEN_TG')
HEROKU_URL = os.environ.get('HEROKU_URL')

# Define a few command handlers. These usually take the two arguments update and
# context. Error handlers also receive the raised TelegramError object in error.
def start(update, context):
	"""Send a message when the command /start is issued."""
	update.message.reply_text('Система приведена в рабочее положение. Если интересно как она работает внутри, почитай раздел /info. Для инструкции обратись к /help')

def help(update, context):
	"""Send a message when the command /help is issued."""
	update.message.reply_text('Отправляй картинку в чат и в ответ придёт предсказание какой из героев Симпсонов изображён на картинке.')

def info(update, context):
	"""Layer info block"""
	update.message.reply_text('Система построена на базе свёрточной нейронной сети (CNN), за основу взята архитектура Resnet34 обученная на датасете ImageNet. На базе этой сети замораживались первые 70% слоёв и обучались последние 30% на этом датасете: https://www.kaggle.com/ymalov/simpsons. Исходный код процесса обучения находится здесь: https://www.kaggle.com/ymalov/simpsons-baseline. Автор Юрий Малов: @ymalov')

def error(update, context):
    """Log Errors caused by Updates."""
    logger.warning('Update "%s" caused error "%s"', update, context.error)

# main function
def prediction(update, context):
	# this commands print only for console Heroku App
	print('start prediction')
	random_text = random.choice(('Картинка получена. Достаём монокль ...', \
                             	'Фотокарточка получена. Будим деда, вместе с ним посмотрим ...', \
                             	'Фото на базе. Аналитики подняты по тревоге ...', \
                             	'Доставлено. Авторы Симпсонов встрепенулись ...', \
                             	'Иллюстрация легла на стол аналитика ...', \
                             	'Фреска загружена. Вся королевская рать идёт искать ...'))

	file_block = update.message.photo[0].get_file()
	file_path = file_block['file_path']
	filename = str(file_block['file_unique_id']) + "_" + str(time.time()) + '.jpg'
	wget.download(file_path, str(filename))
	# message - file downloaded
	update.message.reply_text(random_text)
	
	# set empty model
	model = torchvision.models.resnet34(pretrained=False)
	
	# set FC layer with out 42. 42 = list persons from Simpsons cartoon
	model.fc = nn.Linear(512, 42)
	
	# load my weights and set eval
	model.load_state_dict(torch.load('resnet34_e30.pth', map_location=torch.device('cpu')))
	model.eval()

	# preprocessing
	print('start load_sample')
	file = Image.open(filename)
	file = file.convert("RGB")
	file.load()
	file = file.resize((224, 224))
	print('finish load_sample')

	# get tensor
	print('start transform_image')
	input_transforms = [transforms.Resize(255),
		transforms.CenterCrop(224),
		transforms.ToTensor(),
		# normalize for example ImageNet dataset
		transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])]
	my_transforms = transforms.Compose(input_transforms)
	file = my_transforms(file)
	file = file.unsqueeze_(0)
	print('finish transform_image')

	# run get_prediction
	print('start get_prediction')
	outputs = model.forward(file)
	prob, y_hat = outputs.max(1)
	prediction = y_hat.item()
	prob = prob.item()
	# get probability in %
	prob = "{:10.1f}".format(prob)
	print('finish get_prediction')

	print('start render_prediction')
	print('start img_class_map')
	# get list persons
	img_class_map = None
	mapping_file_path = 'index_to_name.json'
	if os.path.isfile(mapping_file_path):
		with open (mapping_file_path) as f:
			img_class_map = json.load(f)
	print('finish img_class_map')

	stridx = str(prediction)
	class_name = 'Unknown'
	if img_class_map is not None:
		if stridx in img_class_map is not None:
			class_name = img_class_map[stridx][1]
	print('finish render_prediction')

	# output = ('Это ' + class_name + ' с вероятностью:' + prob + '%')
	output = ('Результат: это ' + class_name)
	# send pred. message to Telegram
	update.message.reply_text(output)
	# remove file
	os.remove(file)
	print('finish prediction')


def main():
	print('start main')
	"""Start the bot."""
	# Create the Updater and pass it your bot's token.
	# Make sure to set use_context=True to use the new context based callbacks
	# Post version 12 this will no longer be necessary
	updater = Updater(TOKEN, use_context=True)

	# Get the dispatcher to register handlers
	dp = updater.dispatcher

	# on different commands - answer in Telegram
	dp.add_handler(CommandHandler("start", start))
	dp.add_handler(CommandHandler("help", help))
	dp.add_handler(CommandHandler("info", info))
	dp.add_handler(MessageHandler(Filters.photo, prediction))

	# log all errors
	dp.add_error_handler(error)

	# Start the Bot
	updater.start_webhook(listen="0.0.0.0",
				port=int(PORT),
                        	url_path=TOKEN)
	updater.bot.setWebhook(HEROKU_URL + TOKEN)

	# Run the bot until you press Ctrl-C or the process receives SIGINT,
	# SIGTERM or SIGABRT. This should be used most of the time, since
	# start_polling() is non-blocking and will stop the bot gracefully.
	updater.idle()

if __name__ == '__main__':
	main()

