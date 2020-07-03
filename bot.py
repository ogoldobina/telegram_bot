import logging
from aiogram import Bot, Dispatcher, executor, types
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.dispatcher import FSMContext
from aiogram.dispatcher.filters.state import State, StatesGroup
import aiogram.utils.markdown as md
from aiogram.dispatcher.filters import Text
from aiogram.types import ParseMode
from PIL import Image
import io
import style_transfer as st
from bot_token import API_TOKEN

logging.basicConfig(level=logging.INFO)

ms = MemoryStorage()
bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot, storage=ms)

class Form(StatesGroup):
    style_pic = State()
    content_pic = State()

@dp.message_handler(commands=['start', 'help'])
async def help(message: types.Message):
    markup = types.ReplyKeyboardRemove()
    msg = md.text(
        md.text('Hi! This bot can transfer style from one picture to another. '
            'To stylize your picture use command /style\_transfer.\n'),
        md.text('You will be asked to send two pictures:\n'),
        md.bold('style picture'),
        md.text(' - the one to take style patterns from,\n'),
        md.bold('content picture'),
        md.text(' - the one to stylize.')
    )
    await message.answer(msg, parse_mode=ParseMode.MARKDOWN, reply_markup=markup)

@dp.message_handler(commands='style_transfer')
async def style_transfer(message: types.Message):
    await Form.style_pic.set()
    await message.answer('Send style picture, please.')

@dp.message_handler(content_types=['photo'], state=Form.style_pic)
async def get_style_pic(message: types.Message, state: FSMContext):
    await message.photo[-1].download()
    path = (await message.photo[-1].get_file())['file_path']
    async with state.proxy() as data:
        data['style_pic_path'] = path

    await Form.content_pic.set()
    await message.answer('Style picture saved.')
    await message.answer('Send content picture, please.')

@dp.message_handler(content_types=['photo'], state=Form.content_pic)
async def get_content_pic(message: types.Message, state: FSMContext):
    await message.photo[-1].download()
    path = (await message.photo[-1].get_file())['file_path']
    async with state.proxy() as data:
        data['content_pic_path'] = path

        await message.answer('Content picture saved.')

        content_img = st.load_img(data['content_pic_path'])
        style_img = st.load_img(data['style_pic_path'])

    model = st.ST_model()
    text = 'Processing your picture...'
    ans_msg = (await bot.send_message(message.chat.id, text))
    msg = st.MessageRW(bot, ans_msg)
    output_img = await model.run_style_transfer(content_img, style_img, msg, num_steps=60)  # 300

    outputByteArr = io.BytesIO()
    output_img.save(outputByteArr, format='JPEG')
    outputByteArr = outputByteArr.getvalue()

    await message.answer_photo(outputByteArr, 'Here is your stylized picture!')
    await state.finish()

@dp.message_handler(commands='stop', state=[Form.style_pic, Form.content_pic])
async def stop_st(message: types.Message, state: FSMContext):
    await state.finish()
    await message.answer('The style transfer algorithm stopped.')

@dp.message_handler(content_types='any', state=[Form.style_pic, Form.content_pic])
async def wrong_content(message: types.Message, state: FSMContext):
    await message.answer('You are supposed to send a picture now.\nTo stop the style transfer algorythm use command /stop.')

@dp.message_handler(content_types='any')
async def echo(message: types.Message):
    await message.answer('Use /help to see what this bot can do.')


if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)
