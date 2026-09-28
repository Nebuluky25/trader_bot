import os
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# SDK de Alpaca
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderStatus

# -------------------------------------------------------------------
# CONFIGURACIÓN DE PÁGINA
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Grok Portfolio Master Dashboard",
    page_icon="📈",
    layout="wide",
)

# -------------------------------------------------------------------
# CREDENCIALES DE ALPACA PAPER TRADING ($0)
# -------------------------------------------------------------------
API_KEY = "PKCP4J57RVFLIN6JIBLMGUP44J"
SECRET_KEY = "7xDBtvUPZdqt9P6HYN8jHSeH4NA5HUzPXqnUos3dJ5ui"

CSV_FILE = "trades_history.csv"


@st.cache_resource
def get_alpaca_client():
    """Inicializa la conexión con Alpaca Paper Trading."""
    try:
        return TradingClient(API_KEY, SECRET_KEY, paper=True)
    except Exception as e:
        st.sidebar.error(f"Error de conexión con Alpaca: {e}")
        return None


trading_client = get_alpaca_client()

# -------------------------------------------------------------------
# CABECERA
# -------------------------------------------------------------------
st.title("📊 Grok Portfolio Master — Dashboard Cuantitativo")

if trading_client:
    st.success("🟢 Conectado en tiempo real con Alpaca Paper Trading ($0)")
else:
    st.info("ℹ️ Modo Local (Configura tus claves API de Alpaca para sincronizar la cuenta real en simulación)")

# -------------------------------------------------------------------
# OBTENCIÓN DE DATOS (ALPACA vs CSV)
# -------------------------------------------------------------------
account_data = None
positions_data = []

if trading_client:
    try:
        # Obtenemos la cuenta y posiciones reales desde el servidor de Alpaca
        account = trading_client.get_account()
        capital_actual = float(account.equity)
        capital_inicial = 5000.00
        pl_total = capital_actual - capital_inicial
        
        # Posiciones abiertas
        positions = trading_client.get_all_positions()
        positions_data = [
            {
                "Ticker": p.symbol,
                "Cantidad": float(p.qty),
                "Precio Entrada": f"${float(p.avg_entry_price):,.2f}",
                "Precio Actual": f"${float(p.current_price):,.2f}",
                "P&L No Realizado ($)": f"${float(p.unrealized_pl):,.2f}",
                "Rendimiento (%)": f"{float(p.unrealized_plpc) * 100:.2f}%",
            }
            for p in positions
        ]
    except Exception as e:
        st.error(f"No se pudieron consultar los datos de Alpaca: {e}")
        trading_client = None

# Carga de respaldo local CSV
df_csv = pd.read_csv(CSV_FILE) if os.path.exists(CSV_FILE) else None

if not trading_client and (df_csv is None or df_csv.empty):
    st.warning("⚠️ No se encontró el archivo `trades_history.csv` ni conexión a Alpaca. Registra operaciones primero.")
    st.stop()

# -------------------------------------------------------------------
# MÉTRICAS PRINCIPALES
# -------------------------------------------------------------------
if not trading_client and df_csv is not None:
    capital_inicial = 5000.00
    pl_total = df_csv["PL_Neto_USD"].sum()
    capital_actual = capital_inicial + pl_total

operaciones_cerradas = df_csv[df_csv["Estado"].str.startswith("CERRADA")] if df_csv is not None and "Estado" in df_csv.columns else []
win_rate = (
    (operaciones_cerradas["PL_Neto_USD"] > 0).mean() * 100
    if len(operaciones_cerradas) > 0
    else 0.0
)
total_ops = len(df_csv) if df_csv is not None else 0

col1, col2, col3, col4 = st.columns(4)
col1.metric("Capital Actual (Equity)", f"${capital_actual:,.2f} USD")
col2.metric("P&L Total Acumulado", f"${pl_total:,.2f} USD", delta=f"{pl_total:,.2f} USD")
col3.metric("Operaciones Registradas", total_ops)
col4.metric("Win Rate (CSV)", f"{win_rate:.1f}%")

st.markdown("---")

# -------------------------------------------------------------------
# POSICIONES ABIERTAS EN TIEMPO REAL (ALPACA)
# -------------------------------------------------------------------
if positions_data:
    st.subheader("⚡ Posiciones Abiertas en Alpaca (Paper Trading)")
    st.dataframe(pd.DataFrame(positions_data), use_container_width=True)
    st.markdown("---")

# -------------------------------------------------------------------
# CURVA DE CAPITAL (EQUITY CURVE)
# -------------------------------------------------------------------
st.subheader("📈 Curva de Capital (Equity Curve)")

if df_csv is not None and not df_csv.empty:
    df_csv["Equity"] = capital_inicial + df_csv["PL_Neto_USD"].cumsum()

    fig, ax = plt.subplots(figsize=(10, 3.5))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')
    
    ax.plot(df_csv["Fecha"], df_csv["Equity"], marker="o", color="#00FFAA", linewidth=2, label="Capital ($)")
    ax.axhline(capital_inicial, color="#888888", linestyle="--", alpha=0.7, label="Capital Inicial ($5,000)")
    
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.set_ylabel("Capital (USD)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
else:
    st.info("Aún no hay suficiente historial en el CSV para trazar la curva de capital.")

# -------------------------------------------------------------------
# HISTORIAL LOCAL DE OPERACIONES
# -------------------------------------------------------------------
st.subheader("📋 Historial Local de Operaciones (CSV)")
if df_csv is not None and not df_csv.empty:
    st.dataframe(df_csv, use_container_width=True)