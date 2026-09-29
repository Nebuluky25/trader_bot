from datetime import datetime
import os
import numpy as np
import pandas as pd
from groq import Groq

# SDK de Alpaca
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

# -------------------------------------------------------------------
# CONFIGURACIÓN DE CREDENCIALES
# -------------------------------------------------------------------
API_KEY = os.environ.get("APCA_API_KEY_ID", "PKCP4J57RVFLIN6JIBLMGUP44J")
SECRET_KEY = os.environ.get(
    "APCA_API_SECRET_KEY", "7xDBtvUPZdqt9P6HYN8jHSeH4NA5HUzPXqnUos3dJ5ui"
)
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

CSV_FILE = "trades_history.csv"

TICKERS = {
    "BTCUSDT": "BTC-USD",
    "ETHUSDT": "ETH-USD",
    "NVDA": "NVDA",
    "SPY": "SPY",
}

# Inicializar cliente Alpaca
try:
  trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)
except Exception as e:
  trading_client = None
  print(f"⚠️ [Alpaca Warning] No se pudo inicializar el cliente: {e}")


# --- INDICADORES TÉCNICOS ---
def calculate_rsi(series, period=14):
  delta = series.diff()
  gain = delta.where(delta > 0, 0).rolling(window=period).mean()
  loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
  rs = gain / loss
  return 100 - (100 / (1 + rs))


def calculate_atr(df, period=14):
  high_low = df["High"] - df["Low"]
  high_close = np.abs(df["High"] - df["Close"].shift())
  low_close = np.abs(df["Low"] - df["Close"].shift())
  tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
  return tr.rolling(window=period).mean()


def obtener_datos_mercado():
  print("--- FASE 1: Descargando datos técnicos de Yahoo Finance ---")
  resumen = ""
  for symbol, yf_ticker in TICKERS.items():
    try:
      data = yf.download(yf_ticker, period="60d", interval="1d", progress=False)
      if data.empty:
        continue

      # Blindaje contra MultiIndex en yfinance
      if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

      # Asegurar que las series sean unidimensionales
      close = data["Close"]
      if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
      close = pd.Series(close.values.flatten(), index=close.index)

      high = data["High"]
      if isinstance(high, pd.DataFrame):
        high = high.iloc[:, 0]
      high = pd.Series(high.values.flatten(), index=high.index)

      low = data["Low"]
      if isinstance(low, pd.DataFrame):
        low = low.iloc[:, 0]
      low = pd.Series(low.values.flatten(), index=low.index)

      clean_df = pd.DataFrame({"High": high, "Low": low, "Close": close})

      last_price = float(close.iloc[-1])
      ema20 = float(close.ewm(span=20, adjust=False).mean().iloc[-1])
      sma50 = float(close.rolling(window=50).mean().iloc[-1])
      rsi14 = float(calculate_rsi(close).iloc[-1])
      atr14 = float(calculate_atr(clean_df).iloc[-1])

      resumen += (
          f"Activo: {symbol} | Precio: ${last_price:,.2f} | EMA20:"
          f" ${ema20:,.2f} | SMA50: ${sma50:,.2f} | RSI14: {rsi14:.2f} | ATR14:"
          f" ${atr14:,.2f}\n"
      )
    except Exception as e:
      print(f"Error procesando {symbol}: {e}")
  return resumen


# --- CONSULTA AUTOMÁTICA A GROQ ---
def consultar_grok(datos_texto):
  print("--- FASE 2: Consultando al CIO Agencial (Grok Llama 3) ---")
  if not GROQ_API_KEY:
    raise ValueError("❌ No se encontró la GROQ_API_KEY en el entorno.")

  client = Groq(api_key=GROQ_API_KEY)

  prompt = f"""
    Actúa como un CIO Agencial de Inversiones cuantitativo. Analiza los siguientes datos de mercado actuales:
    {datos_texto}

    Selecciona la MEJOR oportunidad de hoy (si la hay). Devuelve estrictamente una sola línea separada por comas con el siguiente formato exacto:
    TICKER,SEÑAL,PRECIO_ENTRADA,UNIDADES,STOP_LOSS,TAKE_PROFIT
    Ejemplo: BTCUSDT,LONG,65000.00,0.05,63000.00,70000.00
    Si ninguna oportunidad cumple criterios estrictos, responde únicamente la palabra: HOLD.
    No añadas explicaciones ni formato Markdown adicional.
    """

  completion = client.chat.completions.create(
      model="llama-3.1-8b-instant",
      messages=[{"role": "user", "content": prompt}],
      temperature=0.1,
  )

  return completion.choices[0].message.content.strip()


# --- ORDEN EN ALPACA Y CSV ---
def init_trade_log():
  if not os.path.exists(CSV_FILE):
    df = pd.DataFrame(columns=[
        "Fecha",
        "Ticker",
        "Tipo",
        "Precio_Entrada",
        "Precio_Salida",
        "Unidades",
        "Nocional_USD",
        "Stop_Loss",
        "Take_Profit",
        "Comision_USD",
        "PL_Neto_USD",
        "Estado",
    ])
    df.to_csv(CSV_FILE, index=False)


def enviar_orden_alpaca(symbol, qty, side_str):
  if not trading_client:
    print("⚠️ Cliente de Alpaca no disponible.")
    return None
  try:
    side = OrderSide.BUY if side_str.upper() in ["BUY", "LONG"] else OrderSide.SELL
    symbol_alpaca = (
        symbol.replace("USDT", "/USD") if "USDT" in symbol else symbol
    )

    market_order_data = MarketOrderRequest(
        symbol=symbol_alpaca, qty=qty, side=side, time_in_force=TimeInForce.GTC
    )
    order = trading_client.submit_order(order_data=market_order_data)
    print(
        f"🚀 [Alpaca] Orden {side.value} ejecutada para {symbol_alpaca} (ID:"
        f" {order.id})"
    )
    return order
  except Exception as e:
    print(f"❌ [Alpaca Error] {e}")
    return None

def procesar_senal(respuesta_grok, comision_pct=0.0005):
  print("--- FASE 3: Procesando respuesta y registrando ---")
  init_trade_log()

  # Si la respuesta es HOLD o no tiene formato de orden, registramos el HOLD
  if "HOLD" in respuesta_grok or "," not in respuesta_grok:
    print(f"ℹ️ Estado recibido de Grok: {respuesta_grok}. Registrando HOLD.")
    
    df = pd.read_csv(CSV_FILE)
    new_row = {
        "Fecha": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "Ticker": "SISTEMA",
        "Tipo": "HOLD",
        "Precio_Entrada": 0.0,
        "Precio_Salida": None,
        "Unidades": 0.0,
        "Nocional_USD": 0.0,
        "Stop_Loss": 0.0,
        "Take_Profit": 0.0,
        "Comision_USD": 0.0,
        "PL_Neto_USD": 0.0,
        "Estado": "SIN_OPERACION",
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(CSV_FILE, index=False)
    print("✅ [CSV Registrado] Estado HOLD guardado correctamente.")
    return

  # Si SÍ hay una orden de compra/venta válida, procesa normal:
  partes = [p.strip() for p in respuesta_grok.split(",")]
  if len(partes) < 6:
    print(f"⚠️ Formato inválido recibido de Grok: {respuesta_grok}")
    return

  ticker, senal, precio_str, unidades_str, sl_str, tp_str = partes[:6]

  try:
    precio_entrada = float(precio_str)
    unidades = float(unidades_str)
    stop_loss = float(sl_str)
    take_profit = float(tp_str)
  except ValueError as e:
    print(f"❌ Error al convertir los datos numéricos de Grok: {e}")
    return

  nocional = precio_entrada * unidades
  comision = nocional * comision_pct

  df = pd.read_csv(CSV_FILE)
  new_row = {
      "Fecha": datetime.now().strftime("%Y-%m-%d %H:%M"),
      "Ticker": ticker,
      "Tipo": senal,
      "Precio_Entrada": round(precio_entrada, 2),
      "Precio_Salida": None,
      "Unidades": round(unidades, 5),
      "Nocional_USD": round(nocional, 2),
      "Stop_Loss": round(stop_loss, 2),
      "Take_Profit": round(take_profit, 2),
      "Comision_USD": round(comision, 2),
      "PL_Neto_USD": round(-comision, 2),
      "Estado": "ABIERTA",
  }

  df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
  df.to_csv(CSV_FILE, index=False)
  print(f"✅ [CSV Registrado] {senal} en {ticker} @ ${precio_entrada:,.2f}")

  # Enviar orden real a Alpaca Paper
  enviar_orden_alpaca(ticker, unidades, "BUY" if senal == "LONG" else "SELL")


if __name__ == "__main__":
  datos = obtener_datos_mercado()
  if datos:
    resultado = consultar_grok(datos)
    procesar_senal(resultado)