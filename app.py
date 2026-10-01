
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient


# ================================================================
# CONFIGURACIÓN GENERAL
# ================================================================

BASE_DIR = Path(__file__).resolve().parent
CSV_FILE = BASE_DIR / "trades_history.csv"
ENV_FILE = BASE_DIR / ".env"

# Cargar variables de entorno
load_dotenv(ENV_FILE)

CAPITAL_INICIAL = 5000.00


# ================================================================
# CONFIGURACIÓN DE STREAMLIT
# ================================================================

st.set_page_config(
    page_title="Groq Portfolio Master Dashboard",
    page_icon="📈",
    layout="wide",
)

st.title("📊 Groq Portfolio Master — Dashboard Cuantitativo")


# ================================================================
# CREDENCIALES DE ALPACA PAPER TRADING
# ================================================================

API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")


@st.cache_resource
def get_alpaca_client(api_key, secret_key):
    """Inicializa el cliente de Alpaca Paper Trading."""

    if not api_key or not secret_key:
        return None

    try:
        return TradingClient(
            api_key=api_key,
            secret_key=secret_key,
            paper=True,
        )
    except Exception as e:
        st.sidebar.error(f"Error al inicializar Alpaca: {e}")
        return None


trading_client = get_alpaca_client(API_KEY, SECRET_KEY)


# ================================================================
# ESTADO INICIAL
# ================================================================

capital_actual = CAPITAL_INICIAL
capital_inicial = CAPITAL_INICIAL
pl_total = 0.0

positions_data = []
account_data_ok = False


# ================================================================
# CONEXIÓN CON ALPACA
# ================================================================

if trading_client:
    try:
        account = trading_client.get_account()

        capital_actual = float(account.equity)
        pl_total = capital_actual - capital_inicial

        account_data_ok = True

        positions = trading_client.get_all_positions()

        for position in positions:
            positions_data.append(
                {
                    "Ticker": position.symbol,
                    "Cantidad": float(position.qty),
                    "Precio Entrada": (
                        f"${float(position.avg_entry_price):,.2f}"
                    ),
                    "Precio Actual": (
                        f"${float(position.current_price):,.2f}"
                    ),
                    "P&L No Realizado ($)": (
                        f"${float(position.unrealized_pl):,.2f}"
                    ),
                    "Rendimiento (%)": (
                        f"{float(position.unrealized_plpc) * 100:.2f}%"
                    ),
                }
            )

    except Exception as e:
        st.error(f"No se pudieron consultar los datos de Alpaca: {e}")
        trading_client = None


# ================================================================
# ESTADO DE CONEXIÓN
# ================================================================

if account_data_ok:
    st.success(
        "🟢 Conectado con Alpaca Paper Trading"
    )
elif not API_KEY or not SECRET_KEY:
    st.info(
        "ℹ️ Modo local: configura las claves de Alpaca en el archivo .env."
    )
else:
    st.warning(
        "⚠️ No se ha podido obtener información de Alpaca. "
        "Se intentará utilizar el historial local."
    )


# ================================================================
# CARGA DEL HISTORIAL CSV
# ================================================================

df_csv = None

if CSV_FILE.exists():
    try:
        if CSV_FILE.stat().st_size > 0:
            df_csv = pd.read_csv(CSV_FILE)

            if df_csv.empty:
                st.info("El archivo CSV está vacío.")

    except (pd.errors.EmptyDataError, pd.errors.ParserError) as e:
        st.error(f"Error al leer el archivo CSV: {e}")

    except OSError as e:
        st.error(f"No se pudo abrir el archivo CSV: {e}")


if df_csv is None:
    df_csv = pd.DataFrame()


# ================================================================
# PREPARACIÓN DE LOS DATOS
# ================================================================

if not df_csv.empty:
    # Convertir el beneficio neto a formato numérico
    if "PL_Neto_USD" in df_csv.columns:
        df_csv["PL_Neto_USD"] = pd.to_numeric(
            df_csv["PL_Neto_USD"],
            errors="coerce",
        ).fillna(0.0)

    # Convertir las fechas a formato datetime
    if "Fecha" in df_csv.columns:
        df_csv["Fecha"] = pd.to_datetime(
            df_csv["Fecha"],
            errors="coerce",
        )


# ================================================================
# CÁLCULO DE MÉTRICAS
# ================================================================

total_ops = len(df_csv)

if not account_data_ok:
    capital_inicial = CAPITAL_INICIAL

    if "PL_Neto_USD" in df_csv.columns:
        pl_total = float(df_csv["PL_Neto_USD"].sum())

    capital_actual = capital_inicial + pl_total


# Filtrar operaciones cerradas
if (
    not df_csv.empty
    and "Estado" in df_csv.columns
    and "PL_Neto_USD" in df_csv.columns
):
    operaciones_cerradas = df_csv[
        df_csv["Estado"]
        .astype("string")
        .str.startswith("CERRADA", na=False)
    ].copy()
else:
    operaciones_cerradas = pd.DataFrame()


# Calcular Win Rate
if not operaciones_cerradas.empty:
    win_rate = (
        operaciones_cerradas["PL_Neto_USD"].gt(0).mean() * 100
    )
else:
    win_rate = 0.0


# ================================================================
# MÉTRICAS PRINCIPALES
# ================================================================

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Capital Actual (Equity)",
    f"${capital_actual:,.2f} USD",
)

col2.metric(
    "P&L Total Acumulado",
    f"${pl_total:,.2f} USD",
    delta=f"${pl_total:,.2f} USD",
)

col3.metric(
    "Operaciones Registradas",
    total_ops,
)

col4.metric(
    "Win Rate (CSV)",
    f"{win_rate:.1f}%",
)

st.markdown("---")


# ================================================================
# POSICIONES ABIERTAS EN ALPACA
# ================================================================

st.subheader("⚡ Posiciones Abiertas en Alpaca (Paper Trading)")

if positions_data:
    df_positions = pd.DataFrame(positions_data)

    st.dataframe(
        df_positions,
        width="stretch",
        hide_index=True,
    )
elif account_data_ok:
    st.info("No hay posiciones abiertas en Alpaca.")
else:
    st.info(
        "Las posiciones en tiempo real no están disponibles "
        "sin conexión con Alpaca."
    )

st.markdown("---")


# ================================================================
# CURVA DE CAPITAL (EQUITY CURVE)
# ================================================================

st.subheader("📈 Curva de Capital (Equity Curve)")

if (
    not df_csv.empty
    and "PL_Neto_USD" in df_csv.columns
    and "Fecha" in df_csv.columns
):
    df_equity = df_csv.dropna(subset=["Fecha"]).copy()

    if not df_equity.empty:
        df_equity = df_equity.sort_values("Fecha")

        df_equity["Equity"] = (
            CAPITAL_INICIAL
            + df_equity["PL_Neto_USD"].cumsum()
        )

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

        ax.grid(
            True,
            linestyle="--",
            alpha=0.15,
        )

        ax.legend(
            facecolor="#0e1117",
            labelcolor="white",
        )

        fig.autofmt_xdate()
        fig.tight_layout()

        st.pyplot(fig)
        plt.close(fig)

    else:
        st.info(
            "No hay fechas válidas para representar la curva de capital."
        )

else:
    st.info(
        "Aún no hay suficiente historial con las columnas "
        "'Fecha' y 'PL_Neto_USD' para dibujar la curva de capital."
    )


# ================================================================
# HISTORIAL LOCAL DE OPERACIONES
# ================================================================

st.markdown("---")

st.subheader("📋 Historial Local de Operaciones (CSV)")

if not df_csv.empty:
    st.dataframe(
        df_csv,
        width="stretch",
        hide_index=True,
    )
else:
    st.info(
        "Todavía no hay operaciones registradas en el archivo CSV."
    )


# ================================================================
# INFORMACIÓN ADICIONAL
# ================================================================

st.markdown("---")

st.caption(
    "Dashboard de seguimiento de operaciones. "
    "Las métricas locales se calculan a partir del CSV y "
    "las métricas de cuenta se obtienen de Alpaca Paper Trading."
)