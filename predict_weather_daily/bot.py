import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
import discord
from discord.ext import commands
from tensorflow.keras.models import load_model

token = "token"
df = pd.read_csv("new_datasets.csv")
intents = discord.Intents.all()
bot = commands.Bot(command_prefix=".", intents=intents)
model = load_model("my_model.h5")

X = df["CET"]
y_temp = df["Mean TemperatureC"]
y_hum = df[" Mean Humidity"]
y_ws = df[" Max Wind SpeedKm/h"]

min_date = pd.to_datetime(X.min())
max_date = pd.to_datetime(X.max())

@bot.event
async def on_ready():
    print("Bot Hazır")
    print(f"Bot id: {bot.user.id}")
    print(f"Bot ismi: {bot.user.name}")

@bot.command()
async def yardim(ctx):
    embed=discord.Embed(title="AI Bot Help Command")
    embed.add_field(name=".generate", value="use .generate predict daily weather status", inline=True)
    await ctx.send(embed=embed)

@bot.command()
async def generate(ctx, year)
    id = []
    predict_ = predict_weather(year)
    print(f".generate komudunu kullanan kişi: {ctx.author.id}")
    id.append(ctx.author.id)
    await ctx.send(ctx.author.mention)

    embed=discord.Embed(title="Daily Weather Prediction")
    embed.add_field(name="Status", value=predict_, inline=True)
    await ctx.send(embed=embed)

def predict_weather(year):
    tarih_obj = pd.to_datetime(year)

    zaman_damgasi = (tarih_obj - min_date) / (max_date - min_date)

    tarih_tensor = tf.constant([zaman_damgasi], dtype=tf.float32)
    temp_pred, hum_pred, ws_pred = model.predict(tarih_tensor)

    temp_pred = temp_pred * (y_temp.max() - y_temp.min()) + y_temp.min()
    hum_pred = hum_pred * (y_hum.max() - y_hum.min()) + y_hum.min()
    ws_pred = ws_pred * (y_ws.max() - y_ws.min()) + y_ws.min()
    return temp_pred[0][0], hum_pred[0][0], ws_pred[0][0]


"""     print(f"Tarih: {tarih_obj}")
    print(f"Sıcaklık Tahmini: {temp_pred[0][0]:.2f}°C")
    print(f"Nem Tahmini: {hum_pred[0][0]:.2f}%")
    print(f"Rüzgar Hızı Tahmini: {ws_pred[0][0]:.2f} km/h") """

bot.run(token)
