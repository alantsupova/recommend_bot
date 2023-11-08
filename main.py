from aiogram import Bot, Dispatcher, executor, types
from aiogram.types import ReplyKeyboardRemove, \
    ReplyKeyboardMarkup, KeyboardButton, \
    InlineKeyboardMarkup, InlineKeyboardButton, Location
from aiogram.dispatcher import FSMContext
from aiogram.contrib.fsm_storage.memory import MemoryStorage
import logging
from model_Word2Vec import get_embeddings, get_similar_books


bot = Bot(token="6481647679:AAH7ApLFgeuBYPkuNM4UnnUYniSLORhkC68", parse_mode=types.ParseMode.HTML)
dp = Dispatcher(bot, storage=MemoryStorage())
logging.basicConfig(level=logging.INFO)

button_places = KeyboardButton('Помощь')
greet_kb = ReplyKeyboardMarkup(resize_keyboard=True)
greet_kb.add(button_places)


@dp.message_handler(commands='start')
async def start(message: types.Message):
    """Обработка команды /start"""
    await message.answer(f"Привет! я - бот для рекомендаций книг :) \n"
                         f"Напиши, о чем ты бы хотел почитать, а я посоветую тебе книжку! ",
                         reply_markup=greet_kb)


@dp.message_handler(content_types=['text'])
async def location(message: types.Message, state: FSMContext):
    if message.text == 'Помощь':
        await message.answer(f"Тебе нужно просто написать мне о чем ты хотел бы прочитать, например\n"
                             f"'Что-то о магии в духе Гарри Поттера'",
                             reply_markup=greet_kb)
    else:
        recommended_books = get_similar_books(message.text, MODEL_WORD2VEC)
        final_recommendation = "\n".join(recommended_books)
        await message.answer(f'Список рекомендованных книг: \n'
                             f'{final_recommendation }')


if __name__ == '__main__':
    print('Начало')
    MODEL_WORD2VEC = get_embeddings()
    print('Эмбеддинги получены')

    executor.start_polling(dp, skip_updates=True)
