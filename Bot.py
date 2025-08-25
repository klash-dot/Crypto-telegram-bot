import pandas as pd
import numpy as np
import telebot
import ccxt
from datetime import datetime
import time
import threading

# تهيئة البوت
TELEGRAM_BOT_TOKEN = '8362192432:AAEz-Sz0YBNNF4B3gLRRx67tVwTwR0cfBZ0'
BINANCE_API_KEY = 'CuTU5XNtvhAKyEZbfaxXnavpiR0nQLBPg1esLeyU2lskhtEoxK34wmu5UnHU0Kih'
BINANCE_SECRET_KEY = '311iAQSu5iVW9pWBrp25SAyajip0TcJmfIAwAjI3joaT5MssE26u4OpMXUTZcWRk'

bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN, parse_mode='HTML')

# حالة البوت
class BotState:
    def __init__(self):
        self.is_running = False
        self.last_signal_time = None
        self.performance = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_profit': 0.0,
            'daily_profit': 0.0,
            'monthly_profit': 0.0
        }
        self.settings = {
            'risk_per_trade': 1.0,
            'daily_risk_limit': 3.0,
            'max_drawdown': 8.0,
            'leverage': 3
        }
        self.active_trades = {}
        self.balance = 0.0
        self.chat_id = None

bot_state = BotState()

# رموز التداول
SYMBOLS = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT', 'ADA/USDT']
TIMEFRAMES = ['15m', '1h', '4h']

# اتصال بمنصة بينانس
class BinanceConnection:
    def __init__(self):
        self.exchange = None
        self.connect()
    
    def connect(self):
        """الاتصال بمنصة بينانس"""
        try:
            self.exchange = ccxt.binance({
                'apiKey': BINANCE_API_KEY,
                'secret': BINANCE_SECRET_KEY,
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'future',
                    'adjustForTimeDifference': True
                }
            })
            print("✅ تم الاتصال بمنصة بينانس بنجاح")
            
            # تحميل الأسواق
            self.exchange.load_markets()
            
        except Exception as e:
            print(f"❌ فشل الاتصال ببينانس: {e}")
    
    def get_balance(self):
        """الحصول على الرصيد"""
        try:
            balance = self.exchange.fetch_balance()
            return balance['USDT']['free']
        except Exception as e:
            print(f"❌ خطأ في جلب الرصيد: {e}")
            return 0.0
    
    def fetch_ohlcv(self, symbol, timeframe='15m', limit=100):
        """جلب بيانات الأسعار الحقيقية"""
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except Exception as e:
            print(f"❌ خطأ في جلب بيانات {symbol}: {e}")
            return None
    
    def fetch_ticker(self, symbol):
        """جلب بيانات السعر الحالي"""
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return ticker
        except Exception as e:
            print(f"❌ خطأ في جلب سعر {symbol}: {e}")
            return None

# التحليل الفني بدون talib
class TechnicalAnalyzer:
    def calculate_sma(self, df, period):
        """حساب المتوسط المتحرك البسيط"""
        return df['close'].rolling(window=period).mean()
    
    def calculate_ema(self, df, period):
        """حساب المتوسط المتحرك الأسي"""
        return df['close'].ewm(span=period, adjust=False).mean()
    
    def calculate_rsi(self, df, period=14):
        """حساب مؤشر القوة النسبية RSI"""
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_macd(self, df, fast=12, slow=26, signal=9):
        """حساب مؤشر MACD"""
        exp1 = df['close'].ewm(span=fast, adjust=False).mean()
        exp2 = df['close'].ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        histogram = macd - signal_line
        return macd, signal_line, histogram
    
    def calculate_bollinger_bands(self, df, period=20, std_dev=2):
        """حساب نطاقات بولينجر"""
        sma = df['close'].rolling(window=period).mean()
        std = df['close'].rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, sma, lower_band
    
    def calculate_atr(self, df, period=14):
        """حساب المدى الحقيقي المتوسط"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        atr = true_range.rolling(period).mean()
        return atr
    
    def calculate_stochastic(self, df, k_period=14, d_period=3):
        """حساب مؤشر ستوكاستيك"""
        low_min = df['low'].rolling(window=k_period).min()
        high_max = df['high'].rolling(window=k_period).max()
        stoch_k = 100 * ((df['close'] - low_min) / (high_max - low_min))
        stoch_d = stoch_k.rolling(window=d_period).mean()
        return stoch_k, stoch_d
    
    def calculate_all_indicators(self, df):
        """حساب جميع المؤشرات الفنية"""
        try:
            # المتوسطات المتحركة
            df['SMA_20'] = self.calculate_sma(df, 20)
            df['SMA_50'] = self.calculate_sma(df, 50)
            df['EMA_20'] = self.calculate_ema(df, 20)
            df['EMA_50'] = self.calculate_ema(df, 50)
            df['EMA_100'] = self.calculate_ema(df, 100)
            df['EMA_200'] = self.calculate_ema(df, 200)
            
            # MACD
            macd, macd_signal, macd_hist = self.calculate_macd(df)
            df['MACD'] = macd
            df['MACD_Signal'] = macd_signal
            df['MACD_Hist'] = macd_hist
            
            # RSI
            df['RSI'] = self.calculate_rsi(df)
            
            # Stochastic
            stoch_k, stoch_d = self.calculate_stochastic(df)
            df['Stoch_K'] = stoch_k
            df['Stoch_D'] = stoch_d
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(df)
            df['BB_Upper'] = bb_upper
            df['BB_Middle'] = bb_middle
            df['BB_Lower'] = bb_lower
            
            # ATR
            df['ATR'] = self.calculate_atr(df)
            
            return df.fillna(method='bfill')
            
        except Exception as e:
            print(f"❌ خطأ في حساب المؤشرات: {e}")
            return df

# استراتيجيات التداول
class TradingStrategies:
    def trend_following_strategy(self, df):
        """استراتيجية تتبع الاتجاه"""
        latest = df.iloc[-1]
        
        # شروط الشراء
        buy_conditions = (
            latest['close'] > latest['EMA_20'] > latest['EMA_50'] and
            latest['MACD'] > latest['MACD_Signal'] and
            latest['RSI'] > 50 and latest['RSI'] < 70 and
            latest['close'] > latest['BB_Middle']
        )
        
        # شروط البيع
        sell_conditions = (
            latest['close'] < latest['EMA_20'] < latest['EMA_50'] and
            latest['MACD'] < latest['MACD_Signal'] and
            latest['RSI'] < 50 and latest['RSI'] > 30 and
            latest['close'] < latest['BB_Middle']
        )
        
        if buy_conditions:
            return 'LONG', 0.75
        elif sell_conditions:
            return 'SHORT', 0.75
        
        return 'NEUTRAL', 0.5
    
    def mean_reversion_strategy(self, df):
        """استراتيجية العودة إلى المتوسط"""
        latest = df.iloc[-1]
        
        # شروط الشراء (تشبع بيعي)
        buy_conditions = (
            latest['RSI'] < 30 and
            latest['close'] < latest['BB_Lower'] and
            latest['Stoch_K'] < 20
        )
        
        # شروط البيع (تشبع شرائي)
        sell_conditions = (
            latest['RSI'] > 70 and
            latest['close'] > latest['BB_Upper'] and
            latest['Stoch_K'] > 80
        )
        
        if buy_conditions:
            return 'LONG', 0.65
        elif sell_conditions:
            return 'SHORT', 0.65
        
        return 'NEUTRAL', 0.5
    
    def analyze_all_strategies(self, df):
        """تحليل جميع الاستراتيجيات"""
        strategies = [
            self.trend_following_strategy(df),
            self.mean_reversion_strategy(df)
        ]
        
        # حساب متوسط الثقة
        buy_signals = [s for s in strategies if s[0] == 'LONG']
        sell_signals = [s for s in strategies if s[0] == 'SHORT']
        
        if len(buy_signals) >= 1:
            avg_confidence = sum(s[1] for s in buy_signals) / len(buy_signals)
            return 'LONG', avg_confidence
        elif len(sell_signals) >= 1:
            avg_confidence = sum(s[1] for s in sell_signals) / len(sell_signals)
            return 'SHORT', avg_confidence
        
        return 'NEUTRAL', 0.5

# نظام إدارة المخاطر
class RiskManager:
    def __init__(self, binance_conn):
        self.binance = binance_conn
    
    def calculate_position_size(self, entry_price, stop_loss_price):
        """حساب حجم المركز"""
        try:
            balance = self.binance.get_balance()
            if balance <= 0:
                return 0
                
            risk_amount = balance * (bot_state.settings['risk_per_trade'] / 100)
            price_difference = abs(entry_price - stop_loss_price)
            
            if price_difference == 0:
                return 0
                
            position_size = risk_amount / price_difference
            return round(position_size, 6)
        except Exception as e:
            print(f"❌ خطأ في حساب حجم المركز: {e}")
            return 0
    
    def calculate_stop_loss_take_profit(self, entry_price, direction, atr):
        """حساب وقف الخسارة وجني الأرباح"""
        try:
            # استخدام ATR لتحديد المسافة
            sl_distance = atr * 1.5
            rr_ratio = 2.0  # نسبة المكافأة إلى المخاطرة 2:1
            
            if direction == 'LONG':
                stop_loss = entry_price - sl_distance
                take_profit = entry_price + (sl_distance * rr_ratio)
            else:
                stop_loss = entry_price + sl_distance
                take_profit = entry_price - (sl_distance * rr_ratio)
                
            return round(stop_loss, 2), round(take_profit, 2)
        except Exception as e:
            print(f"❌ خطأ في حساب وقف الخسارة: {e}")
            return entry_price * 0.98, entry_price * 1.02

# النظام الرئيسي
class TradingSystem:
    def __init__(self):
        self.binance = BinanceConnection()
        self.technical_analyzer = TechnicalAnalyzer()
        self.strategies = TradingStrategies()
        self.risk_manager = RiskManager(self.binance)
    
    def generate_signals(self):
        """إنشاء إشارات التداول"""
        signals = []
        
        for symbol in SYMBOLS:
            try:
                # جلب البيانات الحقيقية من بينانس
                df = self.binance.fetch_ohlcv(symbol, '15m', 100)
                if df is None or len(df) < 50:
                    continue
                
                # التحليل الفني
                df = self.technical_analyzer.calculate_all_indicators(df)
                
                # تطبيق الاستراتيجيات
                direction, confidence = self.strategies.analyze_all_strategies(df)
                
                if direction != 'NEUTRAL' and confidence > 0.6:
                    signal = self.create_signal(symbol, '15m', direction, confidence, df)
                    if signal:
                        signals.append(signal)
                        
            except Exception as e:
                print(f"❌ خطأ في تحليل {symbol}: {e}")
        
        return signals
    
    def create_signal(self, symbol, timeframe, direction, confidence, df):
        """إنشاء إشارة تداول"""
        try:
            latest = df.iloc[-1]
            entry_price = latest['close']
            atr = latest['ATR'] if not pd.isna(latest['ATR']) else (latest['high'] - latest['low']) * 0.5
            
            stop_loss, take_profit = self.risk_manager.calculate_stop_loss_take_profit(
                entry_price, direction, atr
            )
            
            position_size = self.risk_manager.calculate_position_size(entry_price, stop_loss)
            
            if position_size <= 0:
                return None
            
            return {
                'symbol': symbol,
                'timeframe': timeframe,
                'direction': direction,
                'entry_price': round(entry_price, 2),
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'position_size': position_size,
                'confidence': confidence,
                'timestamp': datetime.now(),
                'atr': atr
            }
            
        except Exception as e:
            print(f"❌ خطأ في إنشاء الإشارة: {e}")
            return None

# تهيئة النظام
trading_system = TradingSystem()

def format_signal_message(signal):
    """تنسيق رسالة الإشارة"""
    emoji = "🟢" if signal['direction'] == 'LONG' else "🔴"
    direction_ar = "شراء" if signal['direction'] == 'LONG' else "بيع"
    
    message = f"""
{emoji} <b>إشارة تداول احترافية</b> {emoji}

🏦 <b>العملة:</b> {signal['symbol']}
🎯 <b>الاتجاه:</b> {direction_ar}
💰 <b>سعر الدخول:</b> ${signal['entry_price']:,.2f}
🛑 <b>وقف الخسارة:</b> ${signal['stop_loss']:,.2f}
🎯 <b>جني الربح:</b> ${signal['take_profit']:,.2f}
📈 <b>الإطار الزمني:</b> {signal['timeframe']}
📊 <b>الثقة:</b> {signal['confidence']*100:.1f}%
📦 <b>الحجم:</b> {signal['position_size']:.6f}
⏰ <b>الوقت:</b> {signal['timestamp'].strftime('%H:%M:%S')}

#إشارة_حقيقية #{signal['symbol'].replace('/', '')} #{direction_ar}
"""
    return message

def send_telegram_message(chat_id, message):
    """إرسال رسالة إلى التليجرام"""
    try:
        bot.send_message(chat_id, message)
        print(f"✅ تم إرسال إشارة إلى {chat_id}")
    except Exception as e:
        print(f"❌ فشل في إرسال الرسالة: {e}")

def generate_and_send_signals():
    """إنشاء وإرسال إشارات التداول"""
    if not bot_state.is_running or not bot_state.chat_id:
        return
    
    try:
        signals = trading_system.generate_signals()
        
        if signals:
            for signal in signals[:2]:  # إرسال أول إشارتين فقط
                message = format_signal_message(signal)
                send_telegram_message(bot_state.chat_id, message)
                
                bot_state.performance['total_trades'] += 1
                bot_state.last_signal_time = datetime.now()
                
                print(f"✅ تم إنشاء إشارة {signal['direction']} على {signal['symbol']}")
                time.sleep(1)
                
        else:
            print(f"⏰ {datetime.now().strftime('%H:%M:%S')} - لا توجد إشارات مناسبة")
            
    except Exception as e:
        print(f"❌ خطأ في النظام: {e}")

def run_bot_loop():
    """تشغيل حلقة البوت الرئيسية"""
    while True:
        if bot_state.is_running and bot_state.chat_id:
            generate_and_send_signals()
        time.sleep(60)  # الانتظار 60 ثانية بين كل دورة

# أوامر البوت
@bot.message_handler(commands=['start'])
def start_bot(message):
    """تشغيل البوت"""
    if bot_state.is_running:
        bot.reply_to(message, "✅ البوت يعمل بالفعل!")
        return
        
    bot_state.is_running = True
    bot_state.chat_id = message.chat.id
    
    # تحديث الرصيد
    try:
        balance = trading_system.binance.get_balance()
        bot_state.balance = balance
        balance_msg = f"💰 <b>الرصيد الحالي:</b> ${balance:,.2f}"
    except:
        balance_msg = "⚠️ <b>تعذر جلب الرصيد</b>"
    
    bot.reply_to(message, f"""
✅ <b>تم تشغيل بوت التداول الاحترافي!</b>

{balance_msg}
📊 <b>المخاطرة لكل صفقة:</b> {bot_state.settings['risk_per_trade']}%
⚡ <b>الرافعة:</b> {bot_state.settings['leverage']}x

🎯 <b>المميزات:</b>
• اتصال مباشر ببينانس
• تحليل فني متقدم
• إشارات كل دقيقة
• إدارة مخاطر احترافية

🏦 <b>العملات:</b> BTC, ETH, SOL, XRP, ADA
""")

@bot.message_handler(commands=['stop'])
def stop_bot(message):
    """إيقاف البوت"""
    if not bot_state.is_running:
        bot.reply_to(message, "⛔ البوت متوقف بالفعل!")
        return
        
    bot_state.is_running = False
    bot.reply_to(message, "⛔ تم إيقاف بوت التداول.")

@bot.message_handler(commands=['balance'])
def check_balance(message):
    """التحقق من الرصيد"""
    try:
        balance = trading_system.binance.get_balance()
        bot_state.balance = balance
        bot.reply_to(message, f"💰 <b>الرصيد الحالي:</b> ${balance:,.2f}")
    except:
        bot.reply_to(message, "❌ تعذر جلب الرصيد")

@bot.message_handler(commands=['stats'])
def show_stats(message):
    """عرض الإحصائيات"""
    stats_msg = f"""
📊 <b>إحصائيات البوت</b>

📈 <b>إجمالي الإشارات:</b> {bot_state.performance['total_trades']}
⏰ <b>آخر إشارة:</b> {bot_state.last_signal_time.strftime('%H:%M:%S') if bot_state.last_signal_time else 'N/A'}
💰 <b>الرصيد:</b> ${bot_state.balance:,.2f}
"""
    bot.reply_to(message, stats_msg)

if __name__ == "__main__":
    print("🚀 بدء تشغيل بوت التداول الاحترافي...")
    print("✅ البوت يعمل ببيانات حقيقية من بينانس")
    print("📊 تم استبدال جميع مؤشرات TA-Lib بدوال مبنية يدوياً")
    
    # بدء تشغيل حلقة البوت في خيط منفصل
    bot_thread = threading.Thread(target=run_bot_loop)
    bot_thread.daemon = True
    bot_thread.start()
    
    print("📊 البوت جاهز للعمل. استخدم /start في التليجرام")
    bot.infinity_polling()
