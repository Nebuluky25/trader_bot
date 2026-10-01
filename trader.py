from datetime import datetime
import os
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from groq import Groq
import yfinance as yf

# SDK de Alpaca
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest


# ================================================================
# DETECTOR DE ENTORNO (WEB vs CONSOLA)
# ================================================================
def running_in_streamlit():
  try:
    from streamlit.runtime.scriptrunner import get_script_run_ctx

    return get_script_run_ctx() is not None
  except Exception:
    return False


if running_in_streamlit():
  import streamlit as st


# ================================================================
# CONFIGURACIÓN GENERAL Y CREDENCIALES
# ================================================================

BASE_DIR = Path(__file__).resolve().parent
CSV_FILE = BASE_DIR / "trades_history.csv"
ENV_FILE = BASE_DIR / ".env"

load_dotenv(ENV_FILE)

CAPITAL_INICIAL = 5000.00

API_KEY = os.getenv("ALPACA_API_KEY") or os.getenv(
    "APCA_API_KEY_ID", "PKCP4J57RVFLIN6JIBLMGUP44J"
)
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY") or os.getenv(
    "APCA_API_SECRET_KEY", "7xDBtvUPZdqt9P6HYN8jHSeH4NA5HUzPXqnUos3dJ5ui"
)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

TICKERS = {
    "BTCUSDT": "BTC-USD",
    "ETHUSDT": "ETH-USD",
    "NVDA": "NVDA",
    "SPY": "SPY",
}


# ================================================================
# CLIENTE ALPACA
# ================================================================

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
    account = trading_client.get_account()
    return float(account.cash)
  except Exception:
    return 1000.0


# ================================================================
# INDICADORES TÉCNICOS Y DATOS DE MERCADO
# ================================================================

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

      close = (
          pd.Series(data["Close"].values.flatten(), index=data.index)
          if isinstance(data["Close"], pd.DataFrame)
          else data["Close"]
      )
      high = (
          pd.Series(data["High"].values.flatten(), index=data.index)
          if isinstance(data["High"], pd.DataFrame)
          else data["High"]
      )
      low = (
          pd.Series(data["Low"].values.flatten(), index=data.index)
          if isinstance(data["Low"], pd.DataFrame)
          else data["Low"]
      )

      clean_df = pd.DataFrame({"High": high, "Low": low, "Close": close})

      last_price = float(close.iloc[-1])
      ema20 = float(close.ewm(span=20, adjust=False).mean().iloc[-1])
      sma50 = float(close.rolling(window=50).mean().iloc[-1])
      rsi14 = float(calculate_rsi(close).iloc[-1])
      atr14 = float(calculate_atr(clean_df).iloc[-1])

      resumen_texto += (
          f"Activo: {symbol} | Precio: ${last_price:,.2f} | EMA20:"
          f" ${ema20:,.2f} | SMA50: ${sma50:,.2f} | RSI14: {rsi14:.2f} | ATR14:"
          f" ${atr14:,.2f}\n"
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


# ================================================================
# CONSULTA A GROQ (CIO AGENCIAL)
# ================================================================

def consultar_grok(datos_texto):
  if not GROQ_API_KEY:
    raise ValueError("❌ No se encontró la GROQ_API_KEY en el entorno.")

  saldo_disponible = obtener_saldo_alpaca()
  client = Groq(api_key=GROQ_API_KEY)

  prompt = f"""
    Actúa como un CIO Agencial de Inversiones cuantitativo. Analiza los siguientes datos de mercado actuales:
    {datos_texto}

    Capital en efectivo disponible en cuenta: ${saldo_disponible:,.2f}. 
    Asegúrate de que el tamaño de las unidades solicitadas no supere este capital disponible.

    Selecciona la MEJOR oportunidad de hoy (si la hay). Devuelve estrictamente una sola línea separada por comas con el siguiente formato exacto:
    TICKER,SEÑAL,PRECIO_ENTRADA,UNIDADES,STOP_LOSS,TAKE_PROFIT
    Ejemplo: BTCUSDT,LONG,65000.00,0.005,63000.00,70000.00
    Si ninguna oportunidad cumple criterios estrictos, responde únicamente la palabra: HOLD.
    No añadas explicaciones ni formato Markdown adicional.
    """

  completion = client.chat.completions.create(
      model="openai/gpt-oss-120b",
      messages=[{"role": "user", "content": prompt}],
      temperature=0.1,
  )
  return completion.choices[0].message.content.strip()


# ================================================================
# GESTIÓN DE ÓRDENES Y CSV
# ================================================================

def init_trade_log():
  if not CSV_FILE.exists() or CSV_FILE.stat().st_size == 0:
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
    return None
  try:
    side = OrderSide.BUY if side_str.upper() in ["BUY", "LONG"] else OrderSide.SELL
    symbol_alpaca = (
        symbol.replace("USDT", "/USD") if "USDT" in symbol else symbol
    )
    market_order_data = MarketOrderRequest(
        symbol=symbol_alpaca, qty=qty, side=side, time_in_force=TimeInForce.GTC
    )
    return trading_client.submit_order(order_data=market_order_data)
  except Exception as e:
    print(f"❌ [Alpaca Error] {e}")
    return None


def procesar_senal(respuesta_grok, detalles_activos, comision_pct=0.0005):
  init_trade_log()
  df = pd.read_csv(CSV_FILE)
  fecha_actual = datetime.now().strftime("%Y-%m-%d %H:%M")

  if "HOLD" in respuesta_grok or "," not in respuesta_grok:
    # Registramos una fila por cada activo analizado con su precio actual y estado HOLD
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

    return (
        "ℹ️ Estado HOLD registrado: Se guardaron los precios de mercado"
        " analizados."
    )

  partes = [p.strip() for p in respuesta_grok.split(",")]
  if len(partes) < 6:
    return f"⚠️ Formato inválido recibido de Grok: {respuesta_grok}"

  ticker, senal, precio_str, unidades_str, sl_str, tp_str = partes[:6]

  try:
    precio_entrada = float(precio_str)
    unidades = float(unidades_str)
    stop_loss = float(sl_str)
    take_profit = float(tp_str)
  except ValueError as e:
    return f"❌ Error numérico en la respuesta de Grok: {e}"

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
      "Estado": "ABIERTA",
  }

  df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
  df.to_csv(CSV_FILE, index=False)

  enviar_orden_alpaca(ticker, unidades, "BUY" if senal == "LONG" else "SELL")
  return f"✅ Operación registrada y ejecutada: {senal} en {ticker}"


# ================================================================
# EJECUCIÓN SEGÚN EL ENTORNO (WEB vs TERMINAL)
# ================================================================

if running_in_streamlit():
  # --- INTERFAZ GRÁFICA STREAMLIT ---
  st.set_page_config(
      page_title="Groq Portfolio Master Dashboard",
      page_icon="📈",
      layout="wide",
  )

  st.title("📊 Groq Portfolio Master — Dashboard Cuantitativo & Trading Bot")

  st.sidebar.header("⚙️ Panel de Control")
  if st.sidebar.button("🚀 Ejecutar Análisis CIO (Groq + Alpaca)"):
    with st.spinner(
        "Analizando mercado, consultando a Groq y ejecutando orden..."
    ):
      datos_mercado, detalles_activos = obtener_datos_mercado()
      if datos_mercado:
        try:
          respuesta = consultar_grok(datos_mercado)
          msg = procesar_senal(respuesta, detalles_activos)
          st.sidebar.success(msg)
          st.rerun()
        except Exception as ex:
          st.sidebar.error(f"Error en la ejecución: {ex}")
      else:
        st.sidebar.error("No se pudieron descargar datos de mercado.")

  capital_actual = CAPITAL_INICIAL
  capital_inicial = CAPITAL_INICIAL
  pl_total = 0.0
  positions_data = []
  account_data_ok = False

  if trading_client:
    try:
      account = trading_client.get_account()
      capital_actual = float(account.equity)
      pl_total = capital_actual - capital_inicial
      account_data_ok = True

      positions = trading_client.get_all_positions()
      for position in positions:
        positions_data.append({
            "Ticker": position.symbol,
            "Cantidad": float(position.qty),
            "Precio Entrada": f"${float(position.avg_entry_price):,.2f}",
            "Precio Actual": f"${float(position.current_price):,.2f}",
            "P&L No Realizado ($)": f"${float(position.unrealized_pl):,.2f}",
            "Rendimiento (%)": f"{float(position.unrealized_plpc) * 100:.2f}%",
        })
    except Exception:
      pass

  df_csv = pd.DataFrame()
  if CSV_FILE.exists() and CSV_FILE.stat().st_size > 0:
    try:
      df_csv = pd.read_csv(CSV_FILE)
    except Exception:
      pass

  if not df_csv.empty:
    if "PL_Neto_USD" in df_csv.columns:
      df_csv["PL_Neto_USD"] = pd.to_numeric(
          df_csv["PL_Neto_USD"], errors="coerce"
      ).fillna(0.0)
    if "Fecha" in df_csv.columns:
      df_csv["Fecha"] = pd.to_datetime(df_csv["Fecha"], errors="coerce")

  total_ops = len(df_csv)
  if not account_data_ok:
    capital_inicial = CAPITAL_INICIAL
    if "PL_Neto_USD" in df_csv.columns:
      pl_total = float(df_csv["PL_Neto_USD"].sum())
    capital_actual = capital_inicial + pl_total

  operaciones_cerradas = (
      df_csv[df_csv["Estado"].astype("string").str.startswith("CERRADA", na=False)]
      if not df_csv.empty and "Estado" in df_csv.columns
      else pd.DataFrame()
  )
  win_rate = (
      (operaciones_cerradas["PL_Neto_USD"].gt(0).mean() * 100)
      if not operaciones_cerradas.empty
      else 0.0
  )

  col1, col2, col3, col4 = st.columns(4)
  col1.metric("Capital Actual (Equity)", f"${capital_actual:,.2f} USD")
  col2.metric(
      "P&L Total Acumulado",
      f"${pl_total:,.2f} USD",
      delta=f"${pl_total:,.2f} USD",
  )
  col3.metric("Registros Históricos", total_ops)
  col4.metric("Win Rate (CSV)", f"{win_rate:.1f}%")

  st.markdown("---")
  st.subheader("⚡ Posiciones Abiertas en Alpaca (Paper Trading)")
  if positions_data:
    st.dataframe(pd.DataFrame(positions_data), width="stretch", hide_index=True)
  elif account_data_ok:
    st.info("No hay posiciones abiertas en Alpaca.")
  else:
    st.info("Conexión con Alpaca en modo local / sin posiciones activas.")

  st.markdown("---")
  st.subheader("📈 Curva de Capital (Equity Curve)")
  if (
      not df_csv.empty
      and "PL_Neto_USD" in df_csv.columns
      and "Fecha" in df_csv.columns
  ):
    df_equity = df_csv.dropna(subset=["Fecha"]).copy()
    if not df_equity.empty:
      df_equity = df_equity.sort_values("Fecha")
      df_equity["Equity"] = CAPITAL_INICIAL + df_equity["PL_Neto_USD"].cumsum()

      fig, ax = plt.subplots(figsize=(12, 4))
      fig.patch.set_facecolor("#0e1117")
      ax.set_facecolor("#0e1117")

      ax.plot(
          df_equity["Fecha"],
          df_equity["Equity"],
          marker="o",
          color="#00FFAA",
          linewidth=2,
          label="Capital ($)",
      )
      ax.axhline(
          CAPITAL_INICIAL,
          color="#888888",
          linestyle="--",
          alpha=0.7,
          label="Capital Inicial ($5,000)",
      )

      ax.set_ylabel("Capital (USD)", color="white")
      ax.set_xlabel("Fecha", color="white")
      ax.tick_params(colors="white")
      for spine in ax.spines.values():
        spine.set_color("#555555")
      ax.grid(True, linestyle="--", alpha=0.15)
      ax.legend(facecolor="#0e1117", labelcolor="white")

      fig.autofmt_xdate()
      fig.tight_layout()
      st.pyplot(fig)
      plt.close(fig)
    else:
      st.info("No hay fechas válidas para representar la curva de capital.")
  else:
    st.info("Aún no hay suficiente historial para dibujar la curva de capital.")

  st.markdown("---")
  st.subheader("📋 Historial Completo y Precios Analizados (CSV)")
  if not df_csv.empty:
    st.dataframe(df_csv, width="stretch", hide_index=True)
  else:
    st.info("Todavía no hay registros guardados en el archivo CSV.")

else:
  # --- EJECUCIÓN AUTOMÁTICA DESDE CONSOLA (GitHub Actions / Terminal) ---
  if __name__ == "__main__":
    print("--- INICIANDO EJECUCIÓN AUTOMÁTICA DESDE CONSOLA ---")
    datos, detalles = obtener_datos_mercado()
    if datos:
      try:
        resultado = consultar_grok(datos)
        print(procesar_senal(resultado, detalles))
      except Exception as e:
        print(f"❌ Error en la ejecución automática: {e}")
    else:
      print("❌ No se pudieron descargar datos de mercado.")