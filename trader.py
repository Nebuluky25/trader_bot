from datetime import datetime
import os
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from groq import Groq
import yfinance as yf

# Alpaca se usa únicamente en modo paper mientras EXECUTE_ORDERS no sea true.
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

BASE_DIR = Path(__file__).resolve().parent
CSV_FILE = BASE_DIR / "trades_history.csv"
ENV_FILE = BASE_DIR / ".env"
load_dotenv(ENV_FILE)

CAPITAL_INICIAL = 5000.00
API_KEY = os.getenv("APCA_API_KEY_ID")
SECRET_KEY = os.getenv("APCA_API_SECRET_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
EXECUTE_ORDERS = os.getenv("EXECUTE_ORDERS", "false").strip().lower() == "true"

TICKERS = {
    "BTCUSDT": "BTC-USD",
    "ETHUSDT": "ETH-USD",
    "NVDA": "NVDA",
    "SPY": "SPY",
}


def get_alpaca_client(api_key, secret_key):
    if not api_key or not secret_key:
        return None
    try:
        return TradingClient(api_key=api_key, secret_key=secret_key, paper=True)
    except Exception:
        return None


trading_client = get_alpaca_client(API_KEY, SECRET_KEY)


def obtener_saldo_alpaca():
    if not trading_client:
        return 1000.0
    try:
        return float(trading_client.get_account().cash)
    except Exception:
        return 1000.0


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
    resumen_texto = ""
    detalles_activos = []

    for symbol, yf_ticker in TICKERS.items():
        try:
            data = yf.download(yf_ticker, period="60d", interval="1d", progress=False)
            if data.empty:
                continue

            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

            close = (pd.Series(data["Close"].values.flatten(), index=data.index)
                     if isinstance(data["Close"], pd.DataFrame) else data["Close"])
            high = (pd.Series(data["High"].values.flatten(), index=data.index)
                    if isinstance(data["High"], pd.DataFrame) else data["High"])
            low = (pd.Series(data["Low"].values.flatten(), index=data.index)
                   if isinstance(data["Low"], pd.DataFrame) else data["Low"])

            clean_df = pd.DataFrame({"High": high, "Low": low, "Close": close})
            last_price = float(close.iloc[-1])
            ema20 = float(close.ewm(span=20, adjust=False).mean().iloc[-1])
            sma50 = float(close.rolling(window=50).mean().iloc[-1])
            rsi14 = float(calculate_rsi(close).iloc[-1])
            atr14 = float(calculate_atr(clean_df).iloc[-1])

            resumen_texto += (
                f"Activo: {symbol} | Precio: ${last_price:,.2f} | EMA20: ${ema20:,.2f} "
                f"| SMA50: ${sma50:,.2f} | RSI14: {rsi14:.2f} | ATR14: ${atr14:,.2f}\n"
            )
            detalles_activos.append({
                "Ticker": symbol,
                "Precio": last_price,
                "RSI": rsi14,
                "EMA20": ema20,
                "SMA50": sma50,
            })
        except Exception as e:
            print(f"Error procesando {symbol}: {e}")

    return resumen_texto, detalles_activos


def consultar_grok(datos_texto):
    if not GROQ_API_KEY:
        raise ValueError("No se encontró GROQ_API_KEY en el entorno")

    saldo_disponible = obtener_saldo_alpaca()
    client = Groq(api_key=GROQ_API_KEY)
    prompt = f"""
Actúa como CIO cuantitativo. Analiza los datos actuales:
{datos_texto}

Capital disponible: ${saldo_disponible:,.2f}
Selecciona como máximo una oportunidad estricta o responde HOLD.
Devuelve exactamente una línea: TICKER,SEÑAL,PRECIO_ENTRADA,UNIDADES,STOP_LOSS,TAKE_PROFIT
Ejemplo: BTCUSDT,LONG,65000.00,0.005,63000.00,70000.00
No añadas Markdown ni explicaciones.
"""

    completion = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )
    return completion.choices[0].message.content.strip()


def init_trade_log():
    if not CSV_FILE.exists() or CSV_FILE.stat().st_size == 0:
        pd.DataFrame(columns=[
            "Fecha", "Ticker", "Tipo", "Precio_Entrada", "Precio_Salida",
            "Unidades", "Nocional_USD", "Stop_Loss", "Take_Profit",
            "Comision_USD", "PL_Neto_USD", "Estado",
        ]).to_csv(CSV_FILE, index=False)


def enviar_orden_alpaca(symbol, qty, side_str):
    if not EXECUTE_ORDERS:
        return None
    if not trading_client:
        return None

    try:
        side = OrderSide.BUY if side_str.upper() in ["BUY", "LONG"] else OrderSide.SELL
        symbol_alpaca = symbol.replace("USDT", "/USD") if "USDT" in symbol else symbol
        request = MarketOrderRequest(
            symbol=symbol_alpaca,
            qty=qty,
            side=side,
            time_in_force=TimeInForce.GTC,
        )
        return trading_client.submit_order(order_data=request)
    except Exception as e:
        print(f"Error de Alpaca: {e}")
        return None


def procesar_senal(respuesta_grok, detalles_activos, comision_pct=0.0005):
    init_trade_log()
    df = pd.read_csv(CSV_FILE)
    fecha_actual = datetime.now().strftime("%Y-%m-%d %H:%M")

    if "HOLD" in respuesta_grok or "," not in respuesta_grok:
        nuevas_filas = []
        for activo in detalles_activos:
            nuevas_filas.append({
                "Fecha": fecha_actual,
                "Ticker": activo["Ticker"],
                "Tipo": "HOLD",
                "Precio_Entrada": round(activo["Precio"], 2),
                "Precio_Salida": None,
                "Unidades": 0.0,
                "Nocional_USD": 0.0,
                "Stop_Loss": 0.0,
                "Take_Profit": 0.0,
                "Comision_USD": 0.0,
                "PL_Neto_USD": 0.0,
                "Estado": f"ESPERA (RSI: {activo['RSI']:.1f})",
            })
        if nuevas_filas:
            df = pd.concat([df, pd.DataFrame(nuevas_filas)], ignore_index=True)
            df.to_csv(CSV_FILE, index=False)
        return "Estado HOLD registrado: no se ejecutó ninguna orden."

    partes = [p.strip() for p in respuesta_grok.split(",")]
    if len(partes) < 6:
        return f"Formato inválido recibido de Groq: {respuesta_grok}"

    ticker, senal, precio_str, unidades_str, sl_str, tp_str = partes[:6]
    try:
        precio_entrada = float(precio_str)
        unidades = float(unidades_str)
        stop_loss = float(sl_str)
        take_profit = float(tp_str)
    except ValueError as e:
        return f"Error numérico en la respuesta de Groq: {e}"

    nocional = precio_entrada * unidades
    comision = nocional * comision_pct
    new_row = {
        "Fecha": fecha_actual,
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
        "Estado": "PROPUESTA - REQUIERE APROBACIÓN" if not EXECUTE_ORDERS else "ABIERTA",
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(CSV_FILE, index=False)

    if EXECUTE_ORDERS:
        enviar_orden_alpaca(ticker, unidades, "BUY" if senal == "LONG" else "SELL")
        return f"Operación registrada y ejecutada en paper trading: {senal} en {ticker}"
    return f"Propuesta registrada sin ejecutar: {senal} en {ticker}"


if __name__ == "__main__":
    print("--- ANÁLISIS AUTOMÁTICO: EJECUCIÓN DE ÓRDENES DESACTIVADA ---")
    datos, detalles = obtener_datos_mercado()
    if datos:
        try:
            respuesta = consultar_grok(datos)
            print(procesar_senal(respuesta, detalles))
        except Exception as e:
            print(f"Error en la ejecución: {e}")
    else:
        print("No se pudieron descargar datos de mercado.")
