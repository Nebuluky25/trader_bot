import os
import re
from datetime import datetime
import numpy as np
import pandas as pd
import yfinance as yf

# SDK de Alpaca
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

# -------------------------------------------------------------------
# CONFIGURACIÓN DE CREDENCIALES DE ALPACA PAPER TRADING ($0)
# -------------------------------------------------------------------
API_KEY = "PKCP4J57RVFLIN6JIBLMGUP44J"
SECRET_KEY = "7xDBtvUPZdqt9P6HYN8jHSeH4NA5HUzPXqnUos3dJ5ui"

# Conexión con Alpaca Paper Trading
try:
    trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)
except Exception as e:
    trading_client = None
    print(f"⚠️ Error al inicializar el cliente de Alpaca: {e}")

CSV_FILE = "trades_history.csv"

# Universo de activos de tu portafolio
TICKERS = {
    "BTCUSDT": "BTC-USD",
    "ETHUSDT": "ETH-USD",
    "NVDA": "NVDA",
    "SPY": "SPY",
}


# --- PARTE 1: DESCARGA Y CÁLCULO DE DATOS TÉCNICOS ---
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def calculate_atr(df, period=14):
    high_low = df["High"] - df["Low"]
    high_close = np.abs(df["High"] - df["Close"].shift())
    low_close = np.abs(df["Low"] - df["Close"].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


def fetch_and_format():
    now_str = datetime.now().strftime("%d-%m-%Y %H:%M")
    output = "=======================================================\n"
    output += f"📥 DATOS CUANTITATIVOS DIARIOS — {now_str}\n"
    output += "(Copia todo el bloque siguiente y pégalo en Grok)\n"
    output += "=======================================================\n\n"

    for symbol, yf_ticker in TICKERS.items():
        try:
            data = yf.download(
                yf_ticker, period="60d", interval="1d", progress=False
            )

            if data.empty:
                output += f"**{symbol}**: [ERROR - Sin datos devueltos por YFinance]\n"
                continue

            close = data["Close"].squeeze()
            volume = data["Volume"].squeeze()

            last_price = float(close.iloc[-1])
            ema20 = float(close.ewm(span=20, adjust=False).mean().iloc[-1])
            sma50 = float(close.rolling(window=50).mean().iloc[-1])
            rsi14 = float(calculate_rsi(close).iloc[-1])
            atr14 = float(calculate_atr(data).iloc[-1])

            vol_24h = float(volume.iloc[-1])
            vol_15d_avg = float(volume.tail(15).mean())
            vol_ratio = vol_24h / vol_15d_avg if vol_15d_avg > 0 else 1.0

            output += f"**{symbol}**\n"
            output += f"- Precio Actual: ${last_price:,.2f}\n"
            output += f"- EMA(20): ${ema20:,.2f} | SMA(50): ${sma50:,.2f}\n"
            output += f"- RSI(14): {rsi14:.2f}\n"
            output += f"- ATR(14): ${atr14:,.2f}\n"
            output += f"- Ratio Volumen (24h/15d): {vol_ratio:.2f}x\n\n"

        except Exception as e:
            output += f"**{symbol}**: Error al procesar ({str(e)})\n\n"

    return output


# --- PARTE 2: CONEXIÓN DE ÓRDENES CON ALPACA ---
def enviar_orden_alpaca(symbol, qty, side_str):
    """Ejecuta una orden simulada en Alpaca Paper Trading."""
    if not trading_client:
        print("⚠️ No hay conexión activa con Alpaca. Se omitió la orden en el broker.")
        return None

    try:
        side = OrderSide.BUY if side_str.upper() in ["BUY", "LONG"] else OrderSide.SELL
        
        # Mapeo de tickers (Alpaca requiere formato BTC/USD para crypto)
        symbol_alpaca = symbol.replace("USDT", "/USD") if "USDT" in symbol else symbol

        market_order_data = MarketOrderRequest(
            symbol=symbol_alpaca,
            qty=qty,
            side=side,
            time_in_force=TimeInForce.GTC,
        )

        order = trading_client.submit_order(order_data=market_order_data)
        print(f"🚀 [ALPACA] Orden {side.value} ejecutada con éxito para {symbol_alpaca} (ID: {order.id})")
        return order
    except Exception as e:
        print(f"❌ [ALPACA ERROR] No se pudo enviar la orden a Alpaca: {e}")
        return None


# --- PARTE 3: LECTURA DEL INFORME DE GROK Y GUARDADO EN CSV ---
def init_trade_log():
    """Crea la tabla en el archivo CSV si aún no existe."""
    if not os.path.exists(CSV_FILE):
        df = pd.DataFrame(
            columns=[
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
            ]
        )
        df.to_csv(CSV_FILE, index=False)


def parse_and_save_report(report_text, comision_pct=0.0005):
    """Parsea el informe copiado de Grok, guarda en el CSV y opera en Alpaca."""
    init_trade_log()

    pattern = re.compile(
        r"\|\s*([A-Z0-9]+)\s*\|\s*\$([\d,]+\.?\d*)\s*\|\s*(LONG|SHORT)\s*\|\s*[\d\.\/]+\s*\|\s*\$([\d,]+\.?\d*)\s*\(([\d,]+\.?\d*)[^\)]*\)\s*\|\s*\$([\d,]+\.?\d*)\s*\|\s*\$([\d,]+\.?\d*)\s*\|"
    )

    matches = pattern.findall(report_text)

    if not matches:
        print("\nℹ️ No se detectaron entradas activas (LONG/SHORT) en el texto pegado.")
        return

    df = pd.read_csv(CSV_FILE)
    operaciones_guardadas = 0

    for match in matches:
        ticker, precio_str, senal, nocional_str, unidades_str, sl_str, tp_str = match

        precio_entrada = float(precio_str.replace(",", ""))
        unidades = float(unidades_str.replace(",", ""))
        nocional = float(nocional_str.replace(",", ""))
        stop_loss = float(sl_str.replace(",", ""))
        take_profit = float(tp_str.replace(",", ""))
        comision = nocional * comision_pct

        # 1. Registrar localmente en CSV
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
        operaciones_guardadas += 1
        print(f"\n✅ [CSV REGISTRADO]: {senal} en {ticker} @ ${precio_entrada:,.2f} | Nocional: ${nocional:,.2f}")

        # 2. Transmitir orden a la cuenta de simulación de Alpaca
        enviar_orden_alpaca(symbol=ticker, qty=unidades, side_str="BUY" if senal == "LONG" else "SELL")

    df.to_csv(CSV_FILE, index=False)
    print(f"\n🎉 ¡Proceso completado! Se registraron {operaciones_guardadas} operación(es).")


def close_trade(ticker, precio_salida, motivo="TP", comision_pct=0.0005):
    """Cierra posiciones en el CSV y envía orden de cierre a Alpaca."""
    if not os.path.exists(CSV_FILE):
        print("❌ No existe el archivo CSV.")
        return

    df = pd.read_csv(CSV_FILE)
    open_trades = df[(df["Ticker"] == ticker) & (df["Estado"] == "ABIERTA")]

    if open_trades.empty:
        print(f"⚠️ No hay posiciones ABIERTAS para {ticker}.")
        return

    idx = open_trades.index[-1]
    row = df.loc[idx]

    precio_entrada = float(row["Precio_Entrada"])
    unidades = float(row["Unidades"])
    tipo = row["Tipo"]

    if tipo == "LONG":
        pl_bruto = (precio_salida - precio_entrada) * unidades
    else:
        pl_bruto = (precio_entrada - precio_salida) * unidades

    comision_entrada = float(row["Comision_USD"])
    comision_salida = (precio_salida * unidades) * comision_pct
    comision_total = comision_entrada + comision_salida
    pl_neto = pl_bruto - comision_total

    df.loc[idx, "Precio_Salida"] = round(precio_salida, 2)
    df.loc[idx, "Comision_USD"] = round(comision_total, 2)
    df.loc[idx, "PL_Neto_USD"] = round(pl_neto, 2)
    df.loc[idx, "Estado"] = f"CERRADA_{motivo.upper()}"

    df.to_csv(CSV_FILE, index=False)
    print(f"🏁 [CSV] {ticker} CERRADA: P&L Neto = ${pl_neto:,.2f} USD | Motivo: {motivo.upper()}")

    # Cerrar la posición contrapuesta en Alpaca
    enviar_orden_alpaca(symbol=ticker, qty=unidades, side_str="SELL" if tipo == "LONG" else "BUY")


# --- MENÚ DE INTERACCIÓN GENERAL ---
def main():
    while True:
        print("\n" + "=" * 55)
        print("🤖 BOT SISTEMÁTICO — MENÚ PRINCIPAL (+ ALPACA $0)")
        print("=" * 55)
        print("1. Obtener datos fríos de hoy (para pegar en Grok)")
        print("2. Pegar respuesta de Grok (para guardar en CSV y operar en Alpaca)")
        print("3. Registrar cierre de operación (TP / SL)")
        print("4. Salir")

        opcion = input("\nSelecciona una opción (1-4): ").strip()

        if opcion == "1":
            datos = fetch_and_format()
            print("\n" + datos)

        elif opcion == "2":
            print("\n👉 Pega la respuesta completa de Grok a continuación.")
            print("👉 Escribe 'FIN' en una nueva línea y presiona Enter al terminar:\n")

            lines = []
            while True:
                line = input()
                if line.strip().upper() == "FIN":
                    break
                lines.append(line)

            full_text = "\n".join(lines)
            parse_and_save_report(full_text)

        elif opcion == "3":
            ticker = input("Ticker (ej. BTCUSDT, NVDA): ").strip().upper()
            precio = float(input("Precio de salida ($): "))
            motivo = input("Motivo (TP / SL / MANUAL): ").strip().upper() or "TP"
            close_trade(ticker, precio, motivo)

        elif opcion == "4":
            print("👋 Sesión finalizada.")
            break


if __name__ == "__main__":
    main()