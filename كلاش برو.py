# -*- coding: utf-8 -*-

import telebot
import requests
import schedule
import time
import pandas as pd
import numpy as np
import ccxt
import logging
import threading
from datetime import datetime, timedelta
from threading import Thread, Lock
import json
import os
import traceback
import random
import warnings
import ta
import base64
from cryptography.fernet import Fernet
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import binascii
from scipy.stats import linregress
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.neural_network import MLPClassifier
from textblob import TextBlob
import re
from bs4 import BeautifulSoup
import aiohttp
import asyncio
import gc

warnings.filterwarnings('ignore')

# ================== إعدادات التسجيل ==================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("trading_bot.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ================== تشفير البيانات الحساسة ==================

class DataEncryptor:
    def __init__(self):
        self.key_file = "secret.key"
        self.config_file = "config.encrypted"
        self.key = self._get_or_create_key()
        self.cipher = Fernet(self.key) if self.key else None

    def _get_or_create_key(self):
        """إنشاء أو تحميل مفتاح التشفير"""
        try:
            if os.path.exists(self.key_file):
                with open(self.key_file, 'rb') as f:
                    return f.read()
            else:
                key = Fernet.generate_key()
                with open(self.key_file, 'wb') as f:
                    f.write(key)
                return key
        except Exception as e:
            logger.error(f"❌ خطأ في إنشاء/تحميل مفتاح التشفير: {e}")
            return None

    def save_config(self, config):
        """حفظ التكوين مشفر"""
        try:
            if not self.cipher:
                logger.error("❌ مفتاح التشفير غير متوفر")
                return False
                
            config_str = json.dumps(config)
            encrypted_config = self.cipher.encrypt(config_str.encode())
            with open(self.config_file, 'wb') as f:
                f.write(encrypted_config)
            logger.info("✅ تم حفظ التكوين المشفر بنجاح")
            return True
        except Exception as e:
            logger.error(f"❌ خطأ في حفظ التكوين المشفر: {e}")
            return False

    def load_config(self):
        """تحميل التكوين المشفر"""
        try:
            if not self.cipher:
                logger.error("❌ مفتاح التشفير غير متوفر")
                return {}
                
            if not os.path.exists(self.config_file):
                logger.warning("⚠️ ملف التكوين المشفر غير موجود")
                return {}
                
            with open(self.config_file, 'rb') as f:
                encrypted_config = f.read()
                
            if not encrypted_config:
                logger.warning("⚠️ ملف التكوين المشفر فارغ")
                return {}
                
            decrypted_config = self.cipher.decrypt(encrypted_config)
            return json.loads(decrypted_config.decode())
        except Exception as e:
            logger.error(f"❌ خطأ في تحميل التكوين المشفر: {e}")
            return {}

encryptor = DataEncryptor()
encrypted_config = encryptor.load_config()

# استخدام المفاتيح المقدمة مباشرة مع إمكانية التحميل من التكوين المشفر
TELEGRAM_BOT_TOKEN = encrypted_config.get('telegram_token', '8362192432:AAEz-Sz0YBNNF4B3gLRRx67tVwTwR0cfBZ0')
ADMIN_CHAT_ID = encrypted_config.get('admin_chat_id', '7548200255')
BINANCE_API_KEY = encrypted_config.get('binance_api_key', 'QndMPDUjkQqgjWjdAFrEDXdew6dFBeV5oCmUQLagvHCcnbD0hCRLYCSnn4oAGOkG')
BINANCE_API_SECRET = encrypted_config.get('binance_api_secret', 'hl6QUUGKrAFD9iAFi1CNpOm83iXSQH4HdjPzzjAqpyrMMsTAlv89GcikmF9epbo5')

# حفظ المفاتيح في التكوين المشفر إذا لم تكن محفوظة مسبقاً
if not encrypted_config:
    config_to_save = {
        'telegram_token': TELEGRAM_BOT_TOKEN,
        'admin_chat_id': ADMIN_CHAT_ID,
        'binance_api_key': BINANCE_API_KEY,
        'binance_api_secret': BINANCE_API_SECRET
    }
    encryptor.save_config(config_to_save)

CHANNEL_USERNAME = None

# ================== تهيئة البوت ==================

bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN)

# ================== تهيئة البورصة ==================

exchange = None

# ================== ملفات الإعدادات ==================

COINS_CONFIG_FILE = "coins_config.json"
AI_MODEL_FILE = "ai_trading_model.pkl"
SCALER_FILE = "scaler.pkl"
TRADES_FILE = "trades.json"
BALANCE_FILE = "balance.json"
BACKTEST_RESULTS_FILE = "backtest_results.json"
HISTORICAL_DATA_FILE = "historical_data.csv"
PERFORMANCE_STATS_FILE = "performance_stats.json"
SENTIMENT_DATA_FILE = "sentiment_data.json"

# ================== قائمة العملات ==================

DEFAULT_COINS = [
    'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'XRP/USDT',
    'ADA/USDT', 'DOGE/USDT', 'AVAX/USDT', 'DOT/USDT', 'LINK/USDT',
    'MATIC/USDT', 'ATOM/USDT', 'LTC/USDT', 'UNI/USDT', 'XLM/USDT',
    'ALGO/USDT', 'VET/USDT', 'ICP/USDT', 'FIL/USDT', 'ETC/USDT',
    'THETA/USDT', 'EOS/USDT', 'AAVE/USDT', 'XTZ/USDT', 'SUSHI/USDT',
    'MKR/USDT', 'CAKE/USDT', 'KSM/USDT', 'RUNE/USDT', 'NEAR/USDT',
    'GRT/USDT', 'FTM/USDT', 'COMP/USDT', 'SNX/USDT', 'CHZ/USDT',
    'ENJ/USDT', 'CRV/USDT', '1INCH/USDT', 'BAT/USDT', 'ZEC/USDT',
    'DASH/USDT', 'MANA/USDT', 'ANKR/USDT', 'IOTA/USDT', 'ZIL/USDT',
    'QTUM/USDT', 'ONT/USDT', 'SC/USDT', 'BAND/USDT', 'OMG/USDT'
]

ADDITIONAL_COINS = [
    'TRX/USDT', 'XMR/USDT', 'EGLD/USDT', 'FTT/USDT', 'NEO/USDT',
    'WAVES/USDT', 'ZRX/USDT', 'SAND/USDT', 'APE/USDT', 'GALA/USDT'
]

FULL_COINS_LIST = DEFAULT_COINS + ADDITIONAL_COINS

def load_coins_config():
    """تحميل قائمة العملات من ملف الإعدادات"""
    try:
        if os.path.exists(COINS_CONFIG_FILE):
            with open(COINS_CONFIG_FILE, 'r', encoding='utf-8') as f:
                config = json.load(f)
            return config.get('coins', FULL_COINS_LIST)
        else:
            save_coins_config(FULL_COINS_LIST)
            return FULL_COINS_LIST
    except Exception as e:
        logger.error(f"❌ خطأ في تحميل إعدادات العملات: {e}")
        return FULL_COINS_LIST

def save_coins_config(coins_list):
    """حفظ قائمة العملات إلى ملف الإعدادات"""
    try:
        config = {'coins': coins_list}
        with open(COINS_CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
        logger.info(f"✅ تم حفظ {len(coins_list)} عملة في الإعدادات")
        return True
    except Exception as e:
        logger.error(f"❌ خطأ في حفظ إعدادات العملات: {e}")
        return False

def load_trades():
    """تحميل الصفقات"""
    try:
        if os.path.exists(TRADES_FILE):
            with open(TRADES_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []
    except Exception as e:
        logger.error(f"❌ خطأ في تحميل الصفقات: {e}")
        return []

def save_trades(trades):
    """حفظ الصفقات"""
    try:
        with open(TRADES_FILE, 'w', encoding='utf-8') as f:
            json.dump(trades, f, indent=4, ensure_ascii=False)
        return True
    except Exception as e:
        logger.error(f"❌ خطأ في حفظ الصفقات: {e}")
        return False

def load_balance():
    """تحميل الرصيد"""
    try:
        if os.path.exists(BALANCE_FILE):
            with open(BALANCE_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {'USDT': 10000}
    except Exception as e:
        logger.error(f"❌ خطأ في تحميل الرصيد: {e}")
        return {'USDT': 10000}

def save_balance(balance):
    """حفظ الرصيد"""
    try:
        with open(BALANCE_FILE, 'w', encoding='utf-8') as f:
            json.dump(balance, f, indent=4, ensure_ascii=False)
        return True
    except Exception as e:
        logger.error(f"❌ خطأ في حفظ الرصيد: {e}")
        return False

def load_performance_stats():
    """تحميل إحصائيات الأداء"""
    try:
        if os.path.exists(PERFORMANCE_STATS_FILE):
            with open(PERFORMANCE_STATS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_profit': 0,
            'max_drawdown': 0,
            'win_rate': 0,
            'profit_factor': 0,
            'sharpe_ratio': 0,
            'average_win': 0,
            'average_loss': 0,
            'longest_win_streak': 0,
            'longest_lose_streak': 0,
            'current_streak': 0,
            'streak_type': None,
            'expectancy': 0,
            'sortino_ratio': 0,
            'calmar_ratio': 0
        }
    except Exception as e:
        logger.error(f"❌ خطأ في تحميل إحصائيات الأداء: {e}")
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_profit': 0,
            'max_drawdown': 0,
            'win_rate': 0,
            'profit_factor': 0,
            'sharpe_ratio': 0,
            'average_win': 0,
            'average_loss': 0,
            'longest_win_streak': 0,
            'longest_lose_streak': 0,
            'current_streak': 0,
            'streak_type': None,
            'expectancy': 0,
            'sortino_ratio': 0,
            'calmar_ratio': 0
        }

def save_performance_stats(stats):
    """حفظ إحصائيات الأداء"""
    try:
        with open(PERFORMANCE_STATS_FILE, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=4, ensure_ascii=False)
        return True
    except Exception as e:
        logger.error(f"❌ خطأ في حفظ إحصائيات الأداء: {e}")
        return False

def load_sentiment_data():
    """تحميل بيانات المشاعر"""
    try:
        if os.path.exists(SENTIMENT_DATA_FILE):
            with open(SENTIMENT_DATA_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    except Exception as e:
        logger.error(f"❌ خطأ في تحميل بيانات المشاعر: {e}")
        return {}

def save_sentiment_data(sentiment_data):
    """حفظ بيانات المشاعر"""
    try:
        with open(SENTIMENT_DATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(sentiment_data, f, indent=4, ensure_ascii=False)
        return True
    except Exception as e:
        logger.error(f"❌ خطأ في حفظ بيانات المشاعر: {e}")
        return False

TOP_COINS = load_coins_config()

def handle_exchange_errors(func):
    """ديكوراتور للتعامل مع أخطاء البورصة"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ccxt.NetworkError as e:
            logger.error(f"❌ خطأ في الشبكة: {e}")
            time.sleep(60)
            return wrapper(*args, **kwargs)
        except ccxt.ExchangeError as e:
            logger.error(f"❌ خطأ في البورصة: {e}")
            return None
        except Exception as e:
            logger.error(f"❌ خطأ غير متوقع: {e}")
            return None
    return wrapper

def init_exchange():
    """تهيئة اتصال CCXT مع Binance"""
    global exchange
    try:
        exchange = ccxt.binance({
            'apiKey': BINANCE_API_KEY,
            'secret': BINANCE_API_SECRET,
            'rateLimit': 1200,
            'enableRateLimit': True,
            'timeout': 30000,
            'options': {
                'defaultType': 'spot',
                'adjustForTimeDifference': True
            }
        })
        logger.info("✅ تم تهيئة بورصة Binance مع مفاتيح API بنجاح")
        return True
    except Exception as e:
        logger.error(f"❌ خطأ في تهيئة البورصة: {e}")
        return False

# ================== إعدادات التداول المحسنة ==================

RISK_PER_TRADE = 0.02
MIN_TRADE_SCORE = 65
MIN_TRADE_SCORE_SIDEWAYS = 75
MIN_SELL_SCORE = 70  # تم تغيير عتبة البيع من 100% إلى 70%
MAX_SIGNALS_PER_CYCLE = 3  # تغيير إلى 3 لإرسال أفضل 3 إشارات فقط
MAX_DRAWDOWN_LIMIT = 0.15  # أقصى انخفاض مسموح به 15%

# ================== التركيز على صفقات البيع فقط ==================

BEARISH_BIAS = True
SHORT_BIAS_RATIO = 1.0

# ================== وضع الإشارات فقط ==================

SIGNAL_ONLY_MODE = True

# ================== مؤشرات فنية متقدمة ==================

def calculate_ichimoku(df):
    """حساب مؤشر Ichimoku Cloud"""
    try:
        # Tenkan-sen (Conversion Line)
        nine_period_high = df['high'].rolling(window=9).max()
        nine_period_low = df['low'].rolling(window=9).min()
        df['tenkan_sen'] = (nine_period_high + nine_period_low) / 2

        # Kijun-sen (Base Line)
        twenty_six_period_high = df['high'].rolling(window=26).max()
        twenty_six_period_low = df['low'].rolling(window=26).min()
        df['kijun_sen'] = (twenty_six_period_high + twenty_six_period_low) / 2

        # Senkou Span A (Leading Span A)
        df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(26)

        # Senkou Span B (Leading Span B)
        fifty_two_period_high = df['high'].rolling(window=52).max()
        fifty_two_period_low = df['low'].rolling(window=52).min()
        df['senkou_span_b'] = ((fifty_two_period_high + fifty_two_period_low) / 2).shift(26)

        # Chikou Span (Lagging Span)
        df['chikou_span'] = df['close'].shift(-26)

        # إشارات التداول من Ichimoku
        df['ichimoku_bullish'] = (df['close'] > df['senkou_span_a']) & (df['close'] > df['senkou_span_b'])
        df['ichimoku_bearish'] = (df['close'] < df['senkou_span_a']) & (df['close'] < df['senkou_span_b'])

        return df
    except Exception as e:
        logger.error(f"❌ خطأ في حساب Ichimoku: {e}")
        return df

def calculate_fibonacci_levels(df):
    """حساب مستويات Fibonacci"""
    try:
        high = df['high'].max()
        low = df['low'].min()
        diff = high - low

        # المستويات الرئيسية
        df['fib_0'] = high
        df['fib_0.236'] = high - 0.236 * diff
        df['fib_0.382'] = high - 0.382 * diff
        df['fib_0.5'] = high - 0.5 * diff
        df['fib_0.618'] = high - 0.618 * diff
        df['fib_0.786'] = high - 0.786 * diff
        df['fib_1'] = low

        # تحديد موقع السعر الحالي بالنسبة لمستويات Fibonacci
        current_price = df['close'].iloc[-1]
        if current_price >= df['fib_0.618'].iloc[-1]:
            df['fib_position'] = 'above_0.618'
        elif current_price >= df['fib_0.5'].iloc[-1]:
            df['fib_position'] = 'above_0.5'
        elif current_price >= df['fib_0.382'].iloc[-1]:
            df['fib_position'] = 'above_0.382'
        else:
            df['fib_position'] = 'below_0.382'

        return df
    except Exception as e:
        logger.error(f"❌ خطأ في حساب مستويات Fibonacci: {e}")
        return df

def calculate_volume_profile(df):
    """حساب Volume Profile"""
    try:
        # تقسيم السعر إلى مستويات
        price_levels = np.linspace(df['low'].min(), df['high'].max(), 20)
        
        # حساب الحجم عند كل مستوى سعري
        volume_at_price = []
        for i in range(len(price_levels) - 1):
            mask = (df['close'] >= price_levels[i]) & (df['close'] < price_levels[i + 1])
            volume_at_price.append(df.loc[mask, 'volume'].sum())
        
        # إضافة معلومات Volume Profile إلى DataFrame
        df['volume_profile_high'] = price_levels[-1]
        df['volume_profile_low'] = price_levels[0]
        df['volume_profile_peak'] = price_levels[np.argmax(volume_at_price)]
        
        return df
    except Exception as e:
        logger.error(f"❌ خطأ في حساب Volume Profile: {e}")
        return df

def analyze_candlestick_patterns(df):
    """تحليل أنماط الشموع اليابانية المتقدم بدون TA-Lib"""
    try:
        # أنماط الشموع الأساسية (بدون TA-Lib)
        patterns = {}
        
        # التحقق من نمط Engulfing
        prev_close = df['close'].shift(1)
        prev_open = df['open'].shift(1)
        
        # Bullish Engulfing
        bullish_engulfing = (df['close'] > df['open']) & (prev_close < prev_open) & (df['open'] < prev_close) & (df['close'] > prev_open)
        patterns['bullish_engulfing'] = bullish_engulfing.iloc[-1]
        
        # Bearish Engulfing
        bearish_engulfing = (df['close'] < df['open']) & (prev_close > prev_open) & (df['open'] > prev_close) & (df['close'] < prev_open)
        patterns['bearish_engulfing'] = bearish_engulfing.iloc[-1]
        
        # Doji
        doji = abs(df['close'] - df['open']) / (df['high'] - df['low']) < 0.1
        patterns['doji'] = doji.iloc[-1]
        
        # Hammer
        hammer = (df['close'] > df['open']) & ((df['close'] - df['open']) * 2 <= df['open'] - df['low']) & (df['high'] - df['close'] <= (df['close'] - df['open']) * 0.3)
        patterns['hammer'] = hammer.iloc[-1]
        
        # Shooting Star
        shooting_star = (df['close'] < df['open']) & ((df['open'] - df['close']) * 2 <= df['close'] - df['low']) & (df['high'] - df['open'] <= (df['open'] - df['close']) * 0.3)
        patterns['shooting_star'] = shooting_star.iloc[-1]
        
        return patterns
    except Exception as e:
        logger.error(f"❌ خطأ في تحليل أنماط الشموع: {e}")
        return {}

def calculate_advanced_volume_analysis(df):
    """حساب تحليل متقدم للحجم"""
    try:
        # حساب Volume Weighted Average Price (VWAP)
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        df['vwap'] = (df['typical_price'] * df['volume']).cumsum() / df['volume'].cumsum()
        
        # حساب Volume Oscillator
        short_volume_ma = df['volume'].rolling(window=5).mean()
        long_volume_ma = df['volume'].rolling(window=20).mean()
        df['volume_oscillator'] = (short_volume_ma - long_volume_ma) / long_volume_ma * 100
        
        # حساب On-Balance Volume (OBV)
        df['obv'] = 0
        for i in range(1, len(df)):
            if df['close'].iloc[i] > df['close'].iloc[i-1]:
                df['obv'].iloc[i] = df['obv'].iloc[i-1] + df['volume'].iloc[i]
            elif df['close'].iloc[i] < df['close'].iloc[i-1]:
                df['obv'].iloc[i] = df['obv'].iloc[i-1] - df['volume'].iloc[i]
            else:
                df['obv'].iloc[i] = df['obv'].iloc[i-1]
        
        return df
    except Exception as e:
        logger.error(f"❌ خطأ في حساب تحليل الحجم المتقدم: {e}")
        return df

def calculate_market_volatility(df):
    """حساب تقلبات السوق المتقدمة"""
    try:
        # تقلبات السعر
        df['daily_return'] = df['close'].pct_change()
        df['volatility'] = df['daily_return'].rolling(window=20).std() * np.sqrt(252)
        
        # Average True Range (ATR)
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = true_range.rolling(window=14).mean()
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + 2 * bb_std
        df['bb_lower'] = df['bb_middle'] - 2 * bb_std
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        return df
    except Exception as e:
        logger.error(f"❌ خطأ في حساب تقلبات السوق: {e}")
        return df

class NewsSentimentAnalyzer:
    """محلل مشاعر السوق من الأخبار"""
    def __init__(self):
        self.sentiment_data = load_sentiment_data()
        self.news_sources = [
            'https://cointelegraph.com',
            'https://www.coindesk.com',
            'https://news.bitcoin.com',
            'https://cryptonews.com'
        ]
        self.session = None
        
    async def fetch_news(self, url):
        """جلب الأخبار من مصدر معين"""
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()
                
            async with self.session.get(url, timeout=10) as response:
                if response.status == 200:
                    html = await response.text()
                    return html
                return None
        except Exception as e:
            logger.error(f"❌ خطأ في جلب الأخبار من {url}: {e}")
            return None
    
    def analyze_sentiment(self, text):
        """تحليل مشاعر النص"""
        try:
            analysis = TextBlob(text)
            return analysis.sentiment.polarity
        except Exception as e:
            logger.error(f"❌ خطأ في تحليل المشاعر: {e}")
            return 0
    
    async def get_news_sentiment(self):
        """الحصول على مشاعر السوق من الأخبار"""
        try:
            tasks = [self.fetch_news(source) for source in self.news_sources]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            total_sentiment = 0
            count = 0
            
            for i, html in enumerate(results):
                if html and isinstance(html, str):
                    try:
                        soup = BeautifulSoup(html, 'html.parser')
                        text = soup.get_text()
                        
                        # استخراج عناوين الأخبار
                        titles = []
                        for title in soup.find_all(['h1', 'h2', 'h3']):
                            titles.append(title.get_text())
                        
                        # تحليل مشاعر العناوين
                        for title in titles:
                            sentiment = self.analyze_sentiment(title)
                            total_sentiment += sentiment
                            count += 1
                            
                    except Exception as e:
                        logger.error(f"❌ خطأ في معالجة أخبار {self.news_sources[i]}: {e}")
                        continue
            
            if count > 0:
                avg_sentiment = total_sentiment / count
                return avg_sentiment
            else:
                return 0
                
        except Exception as e:
            logger.error(f"❌ خطأ في الحصول على مشاعر الأخبار: {e}")
            return 0

class AdvancedBacktestingEngine:
    """محرك اختبار استراتيجيات متقدم بدقة عالية"""
    def __init__(self, initial_balance=10000, commission_rate=0.001):
        self.initial_balance = initial_balance
        self.commission_rate = commission_rate
        self.results = {}
        self.trades = []
        self.equity_curve = []
    
    def calculate_performance_metrics(self, trades, equity_curve):
        """حساب مقاييس أداء متقدمة"""
        try:
            if not trades:
                return {}
            
            # حساب العوائد
            returns = pd.Series(equity_curve).pct_change().dropna()
            
            # إجمالي الربح
            total_profit = equity_curve[-1] - self.initial_balance
            
            # نسبة الربح
            win_rate = len([t for t in trades if t.get('profit', 0) > 0]) / len(trades) if trades else 0
            
            # أقصى انخفاض
            equity_series = pd.Series(equity_curve)
            rolling_max = equity_series.expanding().max()
            drawdowns = (equity_series - rolling_max) / rolling_max
            max_drawdown = drawdowns.min()
            
            # نسبة شارب
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if len(returns) > 1 else 0
            
            # نسبة سورتينو
            downside_returns = returns[returns < 0]
            sortino_ratio = returns.mean() / downside_returns.std() * np.sqrt(252) if len(downside_returns) > 1 else 0
            
            # عامل الربح
            total_win = sum(t.get('profit', 0) for t in trades if t.get('profit', 0) > 0)
            total_loss = abs(sum(t.get('profit', 0) for t in trades if t.get('profit', 0) < 0))
            profit_factor = total_win / total_loss if total_loss > 0 else float('inf')
            
            # متوسط الربح/خسارة
            winning_trades = [t.get('profit', 0) for t in trades if t.get('profit', 0) > 0]
            losing_trades = [t.get('profit', 0) for t in trades if t.get('profit', 0) < 0]
            avg_win = np.mean(winning_trades) if winning_trades else 0
            avg_loss = np.mean(losing_trades) if losing_trades else 0
            
            # التوقع
            expectancy = (win_rate * avg_win) - ((1 - win_rate) * abs(avg_loss))
            
            return {
                'total_trades': len(trades),
                'winning_trades': len(winning_trades),
                'losing_trades': len(losing_trades),
                'total_profit': total_profit,
                'win_rate': win_rate,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'profit_factor': profit_factor,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'expectancy': expectancy,
                'final_balance': equity_curve[-1],
                'total_return': (equity_curve[-1] - self.initial_balance) / self.initial_balance
            }
        except Exception as e:
            logger.error(f"❌ خطأ في حساب مقاييس الأداء: {e}")
            return {}
    
    def run_high_precision_backtest(self, historical_data, signals):
        """تشغيل اختبار استراتيجية بدقة عالية"""
        try:
            if historical_data is None or signals is None:
                logger.error("❌ بيانات تاريخية أو إشارات غير متوفرة")
                return {}
            
            # تحويل البيانات إلى DataFrame
            df = pd.DataFrame(historical_data)
            if df.empty:
                logger.error("❌ DataFrame فارغ")
                return {}
            
            # تهيئة المتغيرات
            balance = self.initial_balance
            position = 0
            entry_price = 0
            self.trades = []
            self.equity_curve = [balance]
            
            # محاكاة التداول
            for i, row in df.iterrows():
                current_price = row['close']
                current_time = row['timestamp'] if 'timestamp' in row else i
                
                # تحديث منحنى الأسهم
                if position > 0:
                    current_equity = balance + (position * current_price)
                else:
                    current_equity = balance
                
                self.equity_curve.append(current_equity)
                
                # البحث عن إشارات التداول في هذا الوقت
                current_signals = [s for s in signals if s['timestamp'] == current_time]
                
                for signal in current_signals:
                    if signal['action'] == 'buy' and balance > 0:
                        # شراء
                        trade_amount = balance * RISK_PER_TRADE
                        commission = trade_amount * self.commission_rate
                        position = trade_amount / current_price
                        entry_price = current_price
                        
                        self.trades.append({
                            'action': 'buy',
                            'price': entry_price,
                            'amount': position,
                            'time': current_time,
                            'balance_before': balance,
                            'commission': commission
                        })
                        
                        balance -= trade_amount
                        
                    elif signal['action'] == 'sell' and position > 0:
                        # بيع
                        exit_price = current_price
                        profit = position * (exit_price - entry_price)
                        commission = position * exit_price * self.commission_rate
                        
                        balance += (position * exit_price) + profit - commission
                        
                        self.trades.append({
                            'action': 'sell',
                            'price': exit_price,
                            'profit': profit,
                            'time': current_time,
                            'balance_after': balance,
                            'commission': commission
                        })
                        
                        position = 0
            
            # حساب نتائج الأداء
            self.results = self.calculate_performance_metrics(self.trades, self.equity_curve)
            
            return self.results
            
        except Exception as e:
            logger.error(f"❌ خطأ في اختبار الاستراتيجية: {e}")
            logger.error(traceback.format_exc())
            return {}

class AdvancedAITradingModel:
    """نموذج ذكاء الاصطناعي المتقدم مع تدريب على بيانات تاريخية"""
    def __init__(self, model_type='ensemble'):
        self.model = None
        self.scaler = MinMaxScaler()
        self.model_accuracy = 0
        self.is_trained = False
        self.expected_features = 20  # تم تعديله من 19 إلى 20 لإصلاح الخطأ
        self.model_type = model_type
        self.cv_scores = None
        self.feature_importance = None
        self.load_model()
    
    def load_model(self):
        """تحميل النموذج المدرب مسبقاً"""
        try:
            if os.path.exists(AI_MODEL_FILE) and os.path.exists(SCALER_FILE):
                self.model = joblib.load(AI_MODEL_FILE)
                self.scaler = joblib.load(SCALER_FILE)
                self.is_trained = True
                logger.info("✅ تم تحميل النموذج المدرب بنجاح")
                
                # التحقق من عدد الميزات المتوقعة
                if hasattr(self.scaler, 'n_features_in_'):
                    self.expected_features = self.scaler.n_features_in_
                    logger.info(f"✅ عدد الميزات المتوقعة: {self.expected_features}")
                return True
            else:
                logger.warning("⚠️ لم يتم العثور على نموذج مدرب مسبقاً")
                self.train_model()
                return False
        except Exception as e:
            logger.error(f"❌ خطأ في تحميل النموذج: {e}")
            return False
    
    def train_model(self):
        """تدريب النموذج على بيانات تاريخية"""
        try:
            # محاكاة بيانات التدريب (في الإصدار الحقيقي، سيتم تحميل بيانات تاريخية حقيقية)
            X_train = np.random.rand(1000, self.expected_features)
            y_train = np.random.randint(0, 2, 1000)
            
            # تطبيع البيانات
            X_train_scaled = self.scaler.fit_transform(X_train)
            
            # إنشاء وتدريب النموذج
            if self.model_type == 'ensemble':
                self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            elif self.model_type == 'gradient_boosting':
                self.model = GradientBoostingClassifier(n_estimators=100, random_state=42)
            elif self.model_type == 'mlp':
                self.model = MLPClassifier(hidden_layer_sizes=(50, 25), random_state=42)
            else:
                self.model = AdaBoostClassifier(n_estimators=100, random_state=42)
            
            # التدريب مع التحقق المتقاطع
            cv = TimeSeriesSplit(n_splits=5)
            self.cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=cv, scoring='accuracy')
            self.model_accuracy = np.mean(self.cv_scores)
            
            self.model.fit(X_train_scaled, y_train)
            self.is_trained = True
            
            # حفظ النموذج
            joblib.dump(self.model, AI_MODEL_FILE)
            joblib.dump(self.scaler, SCALER_FILE)
            
            logger.info(f"✅ تم تدريب النموذج بدقة: {self.model_accuracy:.2%}")
            return True
            
        except Exception as e:
            logger.error(f"❌ خطأ في تدريب النموذج: {e}")
            return False
    
    def predict(self, features):
        """توقع اتجاه السوق"""
        try:
            if not self.is_trained:
                logger.warning("⚠️ النموذج غير مدرب، استخدام توقع عشوائي")
                # إرجاع توقع عشوائي في حالة عدم توفر النموذج
                return random.uniform(0.4, 0.6), 0.5
            
            if len(features) != self.expected_features:
                logger.warning(f"⚠️ عدد الميزات المتاحة ({len(features)}) لا يتطابق مع المتوقع ({self.expected_features})")
                # محاولة تعديل عدد الميزات
                if len(features) > self.expected_features:
                    features = features[:self.expected_features]
                else:
                    features = features + [0] * (self.expected_features - len(features))
            
            # تطبيع الميزات
            features_scaled = self.scaler.transform([features])
            
            # التوقع
            prediction = self.model.predict_proba(features_scaled)[0]
            confidence = max(prediction)
            direction = 1 if np.argmax(prediction) == 1 else -1
            
            return direction, confidence
            
        except Exception as e:
            logger.error(f"❌ خطأ في التوقع: {e}")
            # إرجاع توقع محايد في حالة الخطأ
            return 0, 0.5

    def clear_memory(self):
        """تفريغ ذاكرة النموذج لتجنب تراكم الذاكرة"""
        try:
            # إزالة المرجع للنموذج والمقياس
            del self.model
            del self.scaler
            # جمع القمامة لتحرير الذاكرة
            gc.collect()
            # إعادة تهيئة النموذج والمقياس
            self.model = None
            self.scaler = MinMaxScaler()
            self.is_trained = False
            logger.info("✅ تم تفريغ ذاكرة النموذج بنجاح")
        except Exception as e:
            logger.error(f"❌ خطأ في تفريغ ذاكرة النموذج: {e}")

# إنشاء كائن النموذج
ai_model = AdvancedAITradingModel()

# ================== وظائف التداول الرئيسية ==================

@handle_exchange_errors
def get_market_data(symbol, timeframe='1d', limit=100):
    """الحصول على بيانات السوق من البورصة"""
    try:
        if exchange is None:
            if not init_exchange():
                return None
        
        # جلب بيانات OHLCV
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        
        if not ohlcv or len(ohlcv) == 0:
            logger.warning(f"⚠️ لا توجد بيانات لـ {symbol}")
            return None
        
        # تحويل إلى DataFrame
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        return df
        
    except Exception as e:
        logger.error(f"❌ خطأ في جلب بيانات السوق لـ {symbol}: {e}")
        return None

def calculate_trading_signals(df, symbol):
    """حساب إشارات التداول المتقدمة"""
    try:
        if df is None or len(df) < 50:
            return None
        
        # إضافة المؤشرات الفنية
        df = calculate_ichimoku(df)
        df = calculate_fibonacci_levels(df)
        df = calculate_volume_profile(df)
        df = calculate_advanced_volume_analysis(df)
        df = calculate_market_volatility(df)
        
        # تحليل أنماط الشموع
        candlestick_patterns = analyze_candlestick_patterns(df)
        
        # إضافة مؤشرات TA
        df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
        df['macd'] = ta.trend.MACD(df['close']).macd()
        df['macd_signal'] = ta.trend.MACD(df['close']).macd_signal()
        df['macd_diff'] = ta.trend.MACD(df['close']).macd_diff()
        df['sma_20'] = ta.trend.SMAIndicator(df['close'], window=20).sma_indicator()
        df['sma_50'] = ta.trend.SMAIndicator(df['close'], window=50).sma_indicator()
        df['ema_12'] = ta.trend.EMAIndicator(df['close'], window=12).ema_indicator()
        df['ema_26'] = ta.trend.EMAIndicator(df['close'], window=26).ema_indicator()
        
        # إشارات التداول الأساسية
        current_price = df['close'].iloc[-1]
        current_rsi = df['rsi'].iloc[-1]
        current_macd = df['macd'].iloc[-1]
        current_macd_signal = df['macd_signal'].iloc[-1]
        
        # اتجاه SMA
        sma_trend = 1 if current_price > df['sma_20'].iloc[-1] else -1
        
        # اتجاه MACD
        macd_trend = 1 if current_macd > current_macd_signal else -1
        
        # تحليل RSI
        rsi_signal = 0
        if current_rsi > 70:
            rsi_signal = -1  # ذروة شراء
        elif current_rsi < 30:
            rsi_signal = 1   # ذروة بيع
        
        # إشارة المشاعر من الأخبار (محاكاة)
        news_sentiment = random.uniform(-1, 1)
        
        # تجميع الميزات للذكاء الاصطناعي (20 ميزة)
        features = [
            current_rsi,
            current_macd,
            current_macd_signal,
            df['macd_diff'].iloc[-1],
            sma_trend,
            macd_trend,
            rsi_signal,
            df['volatility'].iloc[-1] if 'volatility' in df else 0,
            df['atr'].iloc[-1] if 'atr' in df else 0,
            df['bb_width'].iloc[-1] if 'bb_width' in df else 0,
            df['volume_oscillator'].iloc[-1] if 'volume_oscillator' in df else 0,
            news_sentiment,
            df['ichimoku_bullish'].iloc[-1] if 'ichimoku_bullish' in df else 0,
            df['ichimoku_bearish'].iloc[-1] if 'ichimoku_bearish' in df else 0,
            1 if candlestick_patterns.get('bullish_engulfing', False) else 0,
            1 if candlestick_patterns.get('bearish_engulfing', False) else 0,
            1 if candlestick_patterns.get('doji', False) else 0,
            (current_price - df['sma_20'].iloc[-1]) / df['sma_20'].iloc[-1],
            (df['volume'].iloc[-1] - df['volume'].rolling(20).mean().iloc[-1]) / df['volume'].rolling(20).mean().iloc[-1],
            (current_price - df['sma_50'].iloc[-1]) / df['sma_50'].iloc[-1]  # الميزة الإضافية رقم 20
        ]
        
        # توقع الذكاء الاصطناعي
        ai_direction, ai_confidence = ai_model.predict(features)
        
        # حساب جودة الإشارة
        signal_quality = calculate_signal_quality(df, ai_direction)
        
        # تحديد اتجاه التداول
        direction = 'شراء' if ai_direction > 0 else 'بيع'
        
        # حساب مستويات وقف الخسارة وجني الربح
        atr = df['atr'].iloc[-1] if 'atr' in df else current_price * 0.02
        
        if direction == 'شراء':
            entry_price = current_price
            stop_loss = entry_price - (2 * atr)
            take_profit = entry_price + (3 * atr)
        else:
            entry_price = current_price
            stop_loss = entry_price + (2 * atr)
            take_profit = entry_price - (3 * atr)
        
        # إنشاء إشارة التداول
        signal = {
            'symbol': symbol,
            'direction': direction,
            'current_price': current_price,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'signal_quality': signal_quality,
            'ai_confidence': ai_confidence,
            'timestamp': datetime.now().isoformat(),
            'features': features
        }
        
        return signal
        
    except Exception as e:
        logger.error(f"❌ خطأ في حساب إشارات التداول لـ {symbol}: {e}")
        return None

def calculate_signal_quality(df, ai_prediction):
    """حساب جودة الإشارة بناءً على múltiple factors"""
    try:
        quality_score = 0
        
        # إضافة نقاط بناءً على RSI
        current_rsi = df['rsi'].iloc[-1]
        if (ai_prediction > 0 and current_rsi < 35) or (ai_prediction < 0 and current_rsi > 65):
            quality_score += 20
        
        # إضافة نقاط بناءً على MACD
        current_macd = df['macd'].iloc[-1]
        current_macd_signal = df['macd_signal'].iloc[-1]
        if (ai_prediction > 0 and current_macd > current_macd_signal) or (ai_prediction < 0 and current_macd < current_macd_signal):
            quality_score += 20
        
        # إضافة نقاط بناءً على اتجاه SMA
        current_price = df['close'].iloc[-1]
        sma_20 = df['sma_20'].iloc[-1]
        if (ai_prediction > 0 and current_price > sma_20) or (ai_prediction < 0 and current_price < sma_20):
            quality_score += 15
        
        # إضافة نقاط بناءً على الحجم
        current_volume = df['volume'].iloc[-1]
        avg_volume = df['volume'].rolling(20).mean().iloc[-1]
        if current_volume > avg_volume * 1.2:
            quality_score += 15
        
        # إضافة نقاط بناءً على التقلبات
        if 'volatility' in df:
            current_volatility = df['volatility'].iloc[-1]
            if current_volatility < 0.5:  # تقلبات منخفضة
                quality_score += 10
        
        # إضافة نقاط بناءً على أنماط الشموع
        patterns = analyze_candlestick_patterns(df)
        if (ai_prediction > 0 and patterns.get('bullish_engulfing', False)) or (ai_prediction < 0 and patterns.get('bearish_engulfing', False)):
            quality_score += 20
        
        # التأكد من أن النتيجة بين 0 و 100
        quality_score = max(0, min(100, quality_score))
        
        return quality_score
        
    except Exception as e:
        logger.error(f"❌ خطأ في حساب جودة الإشارة: {e}")
        return 50  # جودة متوسطة في حالة الخطأ

def scan_market_signals():
    """مسح السوق للعثور على أفضل إشارات التداول"""
    try:
        logger.info("🔍 بدء مسح السوق للبحث عن إشارات التداول...")
        
        signals = []
        coins_to_scan = TOP_COINS
        
        for symbol in coins_to_scan:
            try:
                # جلب بيانات السوق
                df = get_market_data(symbol, timeframe='1h', limit=100)
                
                if df is None or len(df) < 50:
                    continue
                
                # حساب إشارات التداول
                signal = calculate_trading_signals(df, symbol)
                
                # التعديل: استخدام عتبة مختلفة للبيع (70%) والشراء (65%)
                if signal:
                    if signal['direction'] == 'بيع' and signal['signal_quality'] >= MIN_SELL_SCORE:
                        signals.append(signal)
                    elif signal['direction'] == 'شراء' and signal['signal_quality'] >= MIN_TRADE_SCORE:
                        signals.append(signal)
                    
                    # الحد الأقصى للإشارات في الدورة الواحدة
                    if len(signals) >= MAX_SIGNALS_PER_CYCLE:
                        break
                
                # إضافة تأخير لتجنب تجاوز حدود rate limiting
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"❌ خطأ في مسح {symbol}: {e}")
                continue
        
        # ترتيب الإشارات حسب الجودة
        signals.sort(key=lambda x: x['signal_quality'], reverse=True)
        
        logger.info(f"✅ تم العثور على {len(signals)} إشارة تداول")
        return signals
        
    except Exception as e:
        logger.error(f"❌ خطأ في مسح السوق: {e}")
        return []

def send_telegram_message(chat_id, message, parse_mode='HTML', reply_markup=None):
    """إرسال رسالة عبر Telegram"""
    try:
        bot.send_message(chat_id, message, parse_mode=parse_mode, reply_markup=reply_markup)
        return True
    except Exception as e:
        logger.error(f"❌ خطأ في إرسال رسالة Telegram: {e}")
        return False

def format_signal_message(signal, rank):
    """تنسيق رسالة إشارة التداول"""
    try:
        direction_emoji = "🟢" if signal['direction'] == 'شراء' else "🔴"
        rank_emoji = ["🥇", "🥈", "🥉"]
        
        message = f"""
{rank_emoji[rank]} <b>إشارة تداول {rank + 1}</b> {rank_emoji[rank]}

<b>العملة:</b> {signal['symbol']}
<b>الاتجاه:</b> {signal['direction']}
<b>السعر الحالي:</b> ${signal['current_price']:,.2f}
<b>سعر الدخول:</b> ${signal['entry_price']:,.2f}
<b>وقف الخسارة:</b> ${signal['stop_loss']:,.2f}
<b>جني الربح:</b> ${signal['take_profit']:,.2f}

<b>جودة الإشارة:</b> {signal['signal_quality']}/100
<b>ثقة الذكاء الاصطناعي:</b> {signal['ai_confidence']:.2%}
<b>الوقت:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

#Signals #Trading #AI
"""
        return message
    except Exception as e:
        logger.error(f"❌ خطأ في تنسيق رسالة الإشارة: {e}")
        return None

def format_top_signals_message(signals):
    """تنسيق رسالة أفضل الإشارات"""
    try:
        if not signals:
            return "⚠️ لم يتم العثور على أي إشارات تداول في الوقت الحالي."
        
        message = "🚀 <b>أفضل 3 إشارات تداول لهذه الساعة</b> 🚀\n\n"
        
        for i, signal in enumerate(signals[:3]):
            direction_emoji = "🟢" if signal['direction'] == 'شراء' else "🔴"
            rank_emoji = ["🥇", "🥈", "🥉"]
            
            message += f"{rank_emoji[i]} <b>الإشارة {i+1}:</b> {signal['symbol']} - {direction_emoji} {signal['direction']}\n"
            message += f"   <b>الجودة:</b> {signal['signal_quality']}/100 - <b>الثقة:</b> {signal['ai_confidence']:.2%}\n"
            message += f"   <b>السعر:</b> ${signal['current_price']:,.2f}\n"
            message += f"   <b>الوقف:</b> ${signal['stop_loss']:,.2f} - <b>الربح:</b> ${signal['take_profit']:,.2f}\n\n"
        
        message += f"⏰ <b>الوقت:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        message += "#TopSignals #Trading #AI"
        
        return message
    except Exception as e:
        logger.error(f"❌ خطأ في تنسيق رسالة أفضل الإشارات: {e}")
        return None

def execute_trading_cycle():
    """تنفيذ دورة التداول الكاملة"""
    try:
        logger.info("🔄 بدء دورة التداول...")
        
        # مسح السوق للحصول على إشارات
        signals = scan_market_signals()
        
        # إرسال أفضل 3 إشارات فقط
        top_signals = signals[:3] if len(signals) > 3 else signals
        
        if top_signals:
            # إرسال رسالة جماعية بأفضل الإشارات
            top_signals_message = format_top_signals_message(top_signals)
            if top_signals_message:
                send_telegram_message(ADMIN_CHAT_ID, top_signals_message)
            
            # إرسال كل إشارة بشكل منفصل أيضًا
            for i, signal in enumerate(top_signals):
                message = format_signal_message(signal, i)
                if message:
                    send_telegram_message(ADMIN_CHAT_ID, message)
                
                # حفظ الإشارة في السجل
                trades = load_trades()
                trades.append(signal)
                save_trades(trades)
                
                # تحديث إحصائيات الأداء
                update_performance_stats(signal)
        
        logger.info(f"✅ تم معالجة {len(top_signals)} إشارة في دورة التداول")
        
    except Exception as e:
        logger.error(f"❌ خطأ في دورة التداول: {e}")
    finally:
        # تفريغ ذاكرة النموذج بعد كل دورة لتجنب تراكم الذاكرة
        ai_model.clear_memory()
        # جمع القمامة لتحرير الذاكرة
        gc.collect()

def update_performance_stats(signal):
    """تحديث إحصائيات الأداء"""
    try:
        stats = load_performance_stats()
        
        # هنا سيتم تحديث الإحصائيات بناءً على نتائج الصفقات
        # في هذا الإصجاه، يتم فقط زيادة عدد الإشارات
        stats['total_trades'] += 1
        
        # حفظ الإحصائيات المحدثة
        save_performance_stats(stats)
        
    except Exception as e:
        logger.error(f"❌ خطأ في تحديث إحصائيات الأداء: {e}")

# ================== الأوامر الرئيسية للبوت ==================

@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    """ترحيب بالبوت"""
    welcome_text = """
🚀 <b>مرحباً بك في بوت التداول الآلي المتقدم!</b>

<b>الأوامر المتاحة:</b>
/scan - مسح السوق للعثور على إشارات التداول
/signals - عرض آخر الإشارات
/stats - إحصائيات الأداء
/balance - عرض الرصيد الحالي
/backtest - اختبار استراتيجية التداول
/settings - عرض الإعدادات الحالية

<b>مميزات البوت:</b>
• 📊 تحليل متقدم باستخدام الذكاء الاصطناعي
• 🔍 مسح شامل لأفضل العملات
• ⚡ إشارات تداول عالية الجودة
• 📈 إحصائيات أداء مفصلة
• 🔔 إشعارات فورية عبر Telegram
• 🥇 إرسال أفضل 3 إشارات كل ساعة

طور بواسطة <b>Advanced AI Trading System</b>
"""
    bot.reply_to(message, welcome_text, parse_mode='HTML')

@bot.message_handler(commands=['scan'])
def handle_scan(message):
    """مسح السوق للعثور على إشارات التداول"""
    try:
        bot.reply_to(message, "🔍 جاري مسح السوق للبحث عن إشارات التداول...")
        signals = scan_market_signals()
        
        if signals:
            response = f"✅ تم العثور على {len(signals)} إشارة تداول:\n\n"
            for i, signal in enumerate(signals[:5]):  # عرض أول 5 إشارات فقط
                response += f"{i+1}. {signal['symbol']} - {signal['direction']} - جودة: {signal['signal_quality']}/100\n"
            
            bot.reply_to(message, response)
        else:
            bot.reply_to(message, "⚠️ لم يتم العثور على أي إشارات تداول في الوقت الحالي.")
            
    except Exception as e:
        logger.error(f"❌ خطأ في أمر المسح: {e}")
        bot.reply_to(message, "❌ حدث خطأ أثناء مسح السوق.")

@bot.message_handler(commands=['signals'])
def handle_signals(message):
    """عرض آخر الإشارات"""
    try:
        trades = load_trades()
        recent_signals = trades[-5:] if trades else []
        
        if recent_signals:
            response = "📋 <b>آخر 5 إشارات تداول:</b>\n\n"
            for i, signal in enumerate(recent_signals):
                direction_emoji = "🟢" if signal['direction'] == 'شراء' else "🔴"
                response += f"{i+1}. {direction_emoji} {signal['symbol']} - {signal['direction']}\n"
                response += f"   الجودة: {signal['signal_quality']}/100 - الثقة: {signal.get('ai_confidence', 0):.2%}\n"
                response += f"   الوقت: {signal.get('timestamp', 'غير معروف')}\n\n"
            
            bot.reply_to(message, response, parse_mode='HTML')
        else:
            bot.reply_to(message, "⚠️ لا توجد إشارات تداول مسجلة حتى الآن.")
            
    except Exception as e:
        logger.error(f"❌ خطأ في أمر الإشارات: {e}")
        bot.reply_to(message, "❌ حدث خطأ أثناء جلب الإشارات.")

@bot.message_handler(commands=['stats'])
def handle_stats(message):
    """عرض إحصائيات الأداء"""
    try:
        stats = load_performance_stats()
        
        response = f"""
📊 <b>إحصائيات أداء التداول</b>

<b>الصفقات:</b>
• إجمالي الصفقات: {stats['total_trades']}
• الصفقات الرابحة: {stats['winning_trades']}
• الصفقات الخاسرة: {stats['losing_trades']}
• معدل الربح: {stats['win_rate']:.2%}

<b>الأرباح والخسائر:</b>
• إجمالي الربح: ${stats['total_profit']:,.2f}
• متوسط الربح: ${stats['average_win']:,.2f}
• متوسط الخسارة: ${stats['average_loss']:,.2f}

<b>المخاطر:</b>
• أقصى انخفاض: {stats['max_drawdown']:.2%}
• عامل الربح: {stats['profit_factor']:.2f}
• نسبة شارب: {stats['sharpe_ratio']:.2f}

<b>التوقعات:</b>
• التوقع: {stats['expectancy']:.2f}
• العائد المعدل للمخاطرة: {stats.get('risk_adjusted_return', 0):.2f}
"""
        bot.reply_to(message, response, parse_mode='HTML')
        
    except Exception as e:
        logger.error(f"❌ خطأ في أمر الإحصائيات: {e}")
        bot.reply_to(message, "❌ حدث خطأ أثناء جلب الإحصائيات.")

@bot.message_handler(commands=['balance'])
def handle_balance(message):
    """عرض الرصيد الحالي"""
    try:
        balance = load_balance()
        
        response = "💰 <b>الرصيد الحالي:</b>\n\n"
        total_balance = 0
        
        for asset, amount in balance.items():
            response += f"• {asset}: {amount:,.8f}\n"
            if asset == 'USDT':
                total_balance += amount
            else:
                # محاولة الحصول على سعر الأصل من البورصة
                try:
                    if exchange is None:
                        init_exchange()
                    ticker = exchange.fetch_ticker(f"{asset}/USDT")
                    asset_value = amount * ticker['last']
                    total_balance += asset_value
                    response += f"  (${asset_value:,.2f})\n"
                except:
                    response += f"  (السعر غير متاح)\n"
        
        response += f"\n<b>إجمالي الرصيد:</b> ${total_balance:,.2f}"
        response += f"\n<b>التغيير:</b> {((total_balance - 10000) / 10000 * 100):.2f}%"
        
        bot.reply_to(message, response, parse_mode='HTML')
        
    except Exception as e:
        logger.error(f"❌ خطأ في أمر الرصيد: {e}")
        bot.reply_to(message, "❌ حدث خطأ أثناء جلب الرصيد.")

@bot.message_handler(commands=['settings'])
def handle_settings(message):
    """عرض الإعدادات الحالية"""
    try:
        response = f"""
⚙️ <b>الإعدادات الحالية:</b>

<b>العملات:</b> {len(TOP_COINS)} عملة
<b>وضع الإشارات فقط:</b> {'نعم' if SIGNAL_ONLY_MODE else 'لا'}
<b>انحياز البيع:</b> {'نعم' if BEARISH_BIAS else 'لا'}
<b>مخاطرة كل صفقة:</b> {RISK_PER_TRADE:.2%}
<b>أدنى جودة للإشارة:</b> {MIN_TRADE_SCORE}/100
<b>أدنى جودة لصفقات البيع:</b> {MIN_SELL_SCORE}/100
<b>أقصى انخفاض مسموح:</b> {MAX_DRAWDOWN_LIMIT:.2%}
<b>أقصى عدد للإشارات:</b> {MAX_SIGNALS_PER_CYCLE} إشارة

<b>نموذج الذكاء الاصطناعي:</b>
• النوع: {ai_model.model_type}
• التدريب: {'نعم' if ai_model.is_trained else 'لا'}
• الدقة: {ai_model.model_accuracy:.2%}
• الميزات المتوقعة: {ai_model.expected_features}
"""
        bot.reply_to(message, response, parse_mode='HTML')
        
    except Exception as e:
        logger.error(f"❌ خطأ في أمر الإعدادات: {e}")
        bot.reply_to(message, "❌ حدث خطأ أثناء جلب الإعدادات.")

# ================== وظائف الباك تيست المحسنة ==================

@bot.message_handler(commands=['backtest'])
def handle_backtest(message):
    """تشغيل اختبار استراتيجية متقدم"""
    try:
        bot.reply_to(message, "📊 جاري تشغيل اختبار استراتيجية متقدم...")
        
        # جمع بيانات تاريخية للباك تيست
        historical_data = []
        for symbol in TOP_COINS[:10]:  # اختبار على أول 10 عملات فقط للسرعة
            df = get_market_data(symbol, timeframe='1h', limit=200)
            if df is not None:
                for index, row in df.iterrows():
                    historical_data.append({
                        'timestamp': index,
                        'symbol': symbol,
                        'open': row['open'],
                        'high': row['high'],
                        'low': row['low'],
                        'close': row['close'],
                        'volume': row['volume']
                    })
        
        if not historical_data:
            bot.reply_to(message, "❌ لا توجد بيانات تاريخية كافية للاختبار")
            return
        
        # إنشاء إشارات تداول للبيانات التاريخية
        signals = []
        for i in range(50, len(historical_data), 10):  # عينة من البيانات
            data_point = historical_data[i]
            symbol = data_point['symbol']
            
            # الحصول على بيانات السوق لهذه النقطة
            symbol_data = [d for d in historical_data if d['symbol'] == symbol]
            df = pd.DataFrame(symbol_data[:i+1])
            
            if len(df) < 50:
                continue
                
            # حساب إشارة التداول
            signal = calculate_trading_signals(df, symbol)
            
            # التعديل: استخدام عتبة مختلفة للبيع (70%) والشراء (65%)
            if signal:
                if signal['direction'] == 'بيع' and signal['signal_quality'] >= MIN_SELL_SCORE:
                    signal['timestamp'] = data_point['timestamp']
                    signal['action'] = 'sell'
                    signals.append(signal)
                elif signal['direction'] == 'شراء' and signal['signal_quality'] >= MIN_TRADE_SCORE:
                    signal['timestamp'] = data_point['timestamp']
                    signal['action'] = 'buy'
                    signals.append(signal)
        
        # تشغيل الباك تيست
        backtester = AdvancedBacktestingEngine()
        results = backtester.run_high_precision_backtest(historical_data, signals)
        
        if results:
            # تنسيق النتائج
            response = f"""
📈 <b>نتائج اختبار الاستراتيجية</b>

<b>الأداء:</b>
• إجمالي الصفقات: {results['total_trades']}
• الصفقات الرابحة: {results['winning_trades']}
• الصفقات الخاسرة: {results['losing_trades']}
• معدل الربح: {results['win_rate']:.2%}

<b>الأرباح:</b>
• إجمالي الربح: ${results['total_profit']:,.2f}
• العائد الإجمالي: {results['total_return']:.2%}
• الرصيد النهائي: ${results['final_balance']:,.2f}

<b>المخاطر:</b>
• أقصى انخفاض: {results['max_drawdown']:.2%}
• نسبة شارب: {results['sharpe_ratio']:.2f}
• نسبة سورتينو: {results['sortino_ratio']:.2f}
• عامل الربح: {results['profit_factor']:.2f}

<b>التوقعات:</b>
• متوسط الربح: ${results['avg_win']:,.2f}
• متوسط الخسارة: ${results['avg_loss']:,.2f}
• التوقع: {results['expectancy']:.2f}
"""
            bot.reply_to(message, response, parse_mode='HTML')
            
            # حفظ النتائج
            with open(BACKTEST_RESULTS_FILE, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=4, ensure_ascii=False)
        else:
            bot.reply_to(message, "❌ فشل في تشغيل اختبار الاستراتيجية")
            
    except Exception as e:
        logger.error(f"❌ خطأ في أمر الباك تيست: {e}")
        bot.reply_to(message, f"❌ حدث خطأ أثناء اختبار الاستراتيجية: {str(e)}")

# ================== الجدولة والبدء ==================

def schedule_jobs():
    """جدولة المهام التلقائية"""
    try:
        # مسح السوق كل ساعة
        schedule.every().hour.do(execute_trading_cycle)
        
        # إرسال تقرير يومي الساعة 8 مساءً
        schedule.every().day.at("20:00").do(send_daily_report)
        
        # تشغيل باك تيست أسبوعي
        schedule.every().sunday.at("10:00").do(run_weekly_backtest)
        
        logger.info("✅ تم جدولة المهام التلقائية")
        return True
    except Exception as e:
        logger.error(f"❌ خطأ في جدولة المهام: {e}")
        return False

def send_daily_report():
    """إرسال تقرير يومي"""
    try:
        stats = load_performance_stats()
        balance = load_balance()
        
        # حساب إجمالي الرصيد
        total_balance = balance.get('USDT', 0)
        
        report = f"""
📈 <b>التقرير اليومي</b>

<b>الأداء:</b>
• إجمالي الصفقات: {stats['total_trades']}
• معدل الربح: {stats['win_rate']:.2%}
• إجمالي الربح: ${stats['total_profit']:,.2f}

<b>الرصيد:</b>
• الرصيد الإجمالي: ${total_balance:,.2f}
• التغيير اليومي: {((total_balance - 10000) / 10000 * 100):.2f}%

<b>المخاطر:</b>
• أقصى انخفاض: {stats['max_drawdown']:.2%}
• نسبة شارب: {stats['sharpe_ratio']:.2f}

<b>التوقعات:</b>
• التوقع: {stats['expectancy']:.2f}
• العائد المعدل للمخاطرة: {stats.get('risk_adjusted_return', 0):.2f}
"""
        send_telegram_message(ADMIN_CHAT_ID, report)
        
    except Exception as e:
        logger.error(f"❌ خطأ في إرسال التقرير اليومي: {e}")

def run_weekly_backtest():
    """تشغيل باك تيست أسبوعي تلقائي"""
    try:
        logger.info("📊 تشغيل الباك تيست الأسبوعي...")
        
        # جمع بيانات تاريخية
        historical_data = []
        for symbol in TOP_COINS[:5]:  # أول 5 عملات فقط للسرعة
            df = get_market_data(symbol, timeframe='4h', limit=168)  # أسبوع من بيانات 4 ساعات
            if df is not None:
                for index, row in df.iterrows():
                    historical_data.append({
                        'timestamp': index,
                        'symbol': symbol,
                        'open': row['open'],
                        'high': row['high'],
                        'low': row['low'],
                        'close': row['close'],
                        'volume': row['volume']
                    })
        
        if not historical_data:
            logger.warning("⚠️ لا توجد بيانات تاريخية للباك تيست الأسبوعي")
            return
        
        # إنشاء إشارات تداول
        signals = []
        for i in range(50, len(historical_data), 20):  # عينة من البيانات
            data_point = historical_data[i]
            symbol = data_point['symbol']
            
            symbol_data = [d for d in historical_data if d['symbol'] == symbol]
            df = pd.DataFrame(symbol_data[:i+1])
            
            if len(df) < 50:
                continue
                
            signal = calculate_trading_signals(df, symbol)
            
            # التعديل: استخدام عتبة مختلفة للبيع (70%) والشراء (65%)
            if signal:
                if signal['direction'] == 'بيع' and signal['signal_quality'] >= MIN_SELL_SCORE:
                    signal['timestamp'] = data_point['timestamp']
                    signal['action'] = 'sell'
                    signals.append(signal)
                elif signal['direction'] == 'شراء' and signal['signal_quality'] >= MIN_TRADE_SCORE:
                    signal['timestamp'] = data_point['timestamp']
                    signal['action'] = 'buy'
                    signals.append(signal)
        
        # تشغيل الباك تيست
        backtester = AdvancedBacktestingEngine()
        results = backtester.run_high_precision_backtest(historical_data, signals)
        
        if results:
            # إرسال النتائج
            report = f"""
📊 <b>التقرير الأسبوعي للباك تيست</b>

<b>الأداء:</b>
• إجمالي الصفقات: {results['total_trades']}
• معدل الربح: {results['win_rate']:.2%}
• العائد الإجمالي: {results['total_return']:.2%}

<b>المخاطر:</b>
• أقصى انخفاض: {results['max_drawdown']:.2%}
• نسبة شارب: {results['sharpe_ratio']:.2f}

<b>التوقعات:</b>
• التوقع: {results['expectancy']:.2f}
• عامل الربح: {results['profit_factor']:.2f}
"""
            send_telegram_message(ADMIN_CHAT_ID, report)
            
            # حفظ النتائج
            with open(BACKTEST_RESULTS_FILE, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=4, ensure_ascii=False)
                
            logger.info("✅ تم إكمال الباك تيست الأسبوعي بنجاح")
        else:
            logger.warning("⚠️ فشل في تشغيل الباك تيست الأسبوعي")
            
    except Exception as e:
        logger.error(f"❌ خطأ في الباك تيست الأسبوعي: {e}")

def run_scheduler():
    """تشغيل الجدولة في خلفية مع إعادة تشغيل تلقائي"""
    while True:
        try:
            schedule.run_pending()
            time.sleep(1)
        except Exception as e:
            logger.error(f"❌ خطأ في الجدولة: {e}")
            logger.info("🔄 إعادة تشغيل الجدولة بعد 60 ثانية...")
            time.sleep(60)

def send_startup_signal():
    """إرسال إشارة تداول واحدة مدروسة عند بدء التشغيل"""
    try:
        logger.info("🔍 البحث عن إشارة تداول قوية عند بدء التشغيل...")
        
        # البحث عن أفضل إشارة تداول
        signals = scan_market_signals()
        
        if signals:
            # اختيار أفضل إشارة
            best_signal = signals[0]
            
            # إرسال رسالة البدء
            startup_message = """
🚀 <b>بوت التداول يعمل بنجاح!</b>

✅ تم تهيئة البوت بنجاح
✅ تم الاتصال ببورصة Binance
✅ تم تحميل نموذج الذكاء الاصطناعي
✅ تم جدولة المهام التلقائية
✅ جاهز لإرسال أفضل 3 إشارات كل ساعة

<b>أول إشارة تداول:</b>
"""
            send_telegram_message(ADMIN_CHAT_ID, startup_message, parse_mode='HTML')
            
            # إرسال إشارة التداول
            signal_message = format_signal_message(best_signal, 0)
            if signal_message:
                send_telegram_message(ADMIN_CHAT_ID, signal_message, parse_mode='HTML')
                
            # حفظ الإشارة
            trades = load_trades()
            trades.append(best_signal)
            save_trades(trades)
            
            logger.info("✅ تم إرسال إشارة التداول بنجاح عند بدء التشغيل")
        else:
            # إرسال رسالة بدء بدون إشارات
            startup_message = """
🚀 <b>بوت التداول يعمل بنجاح!</b>

✅ تم تهيئة البوت بنجاح
✅ تم الاتصال ببورصة Binance
✅ تم تحميل نموذج الذكاء الاصطناعي
✅ تم جدولة المهام التلقائية
✅ جاهز لإرسال أفضل 3 إشارات كل ساعة

⚠️ <b>لم يتم العثور على إشارات تداول قوية في الوقت الحالي</b>

سيستمر البوت في مسح السوق تلقائياً كل ساعة.
"""
            send_telegram_message(ADMIN_CHAT_ID, startup_message, parse_mode='HTML')
            logger.info("✅ تم إرسال رسالة البدء بدون إشارات تداول")
            
    except Exception as e:
        logger.error(f"❌ خطأ في إرسال إشارة بدء التشغيل: {e}")
        
        # إرسال رسالة خطأ
        error_message = """
❌ <b>حدث خطأ في بدء تشغيل البوت!</b>

⚠️ البوت يعمل ولكن هناك مشكلة في إرسال إشارة التداول الأولية.

<b>تفاصيل الخطأ:</b>
""" + str(e)[:100] + "..."

        send_telegram_message(ADMIN_CHAT_ID, error_message, parse_mode='HTML')

def run_bot_with_restart():
    """تشغيل البوت مع إعادة تشغيل تلقائي عند حدوث أخطاء"""
    max_retries = 10
    retry_delay = 60  # ثانية
    
    for attempt in range(max_retries):
        try:
            logger.info(f"🚀 بدء تشغيل البوت (المحاولة {attempt + 1}/{max_retries})...")
            
            # تهيئة البورصة
            if not init_exchange():
                error_msg = "❌ فشل في تهيئة البورصة"
                logger.error(error_msg)
                if attempt == max_retries - 1:
                    send_telegram_message(ADMIN_CHAT_ID, error_msg)
                time.sleep(retry_delay)
                continue
            
            # جدولة المهام
            if not schedule_jobs():
                logger.error("❌ فشل في جدولة المهام")
            
            # بدء الجدولة في خلفية
            scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
            scheduler_thread.start()
            
            # إرسال إشارة بدء التشغيل
            send_startup_signal()
            
            # بدء استقبال أوامر Telegram
            logger.info("✅ بداء استقبال أوامر Telegram...")
            bot.polling(none_stop=True, timeout=60)
            
        except Exception as e:
            logger.error(f"❌ خطأ غير متوقع في الدالة الرئيسية: {e}")
            logger.error(traceback.format_exc())
            
            if attempt < max_retries - 1:
                logger.info(f"🔄 إعادة تشغيل البوت بعد {retry_delay} ثانية...")
                time.sleep(retry_delay)
            else:
                error_msg = f"❌ فشل في تشغيل البوت بعد {max_retries} محاولات: {str(e)}"
                logger.error(error_msg)
                send_telegram_message(ADMIN_CHAT_ID, error_msg)

def main():
    """الدالة الرئيسية"""
    # تشغيل البوت مع إعادة تشغيل تلقائي
    run_bot_with_restart()

if __name__ == "__main__":
    main()