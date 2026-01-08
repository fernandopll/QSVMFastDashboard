import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import numpy as np
from itertools import combinations

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="Quantum Benchmark Dashboard",
    page_icon="⚛️",
    layout="wide"
)

# --- 1. FUNCIONES DE CARGA Y PROCESAMIENTO ---

@st.cache_data
def load_data(uploaded_files):
    dfs = {"test": [], "train": [], "total": []}

    for file in uploaded_files:
        filename = file.name
        if "Slave1" in filename:
            machine_name = "Slave 1"
        elif "Slave2" in filename:
            machine_name = "Slave 2"
        elif "Slave6" in filename:
            machine_name = "Slave 6"
        else:
            parts = filename.split('_')
            machine_name = parts[1] if len(parts) > 1 else filename.split('.')[0]

        xls = pd.ExcelFile(file)
        sheet_map = {name.lower(): name for name in xls.sheet_names}

        for key in dfs.keys():
            if key in sheet_map:
                df = pd.read_excel(file, sheet_name=sheet_map[key])
                df['Machine'] = machine_name
                df['Source File'] = filename
                dfs[key].append(df)

    return {k: pd.concat(v, ignore_index=True) if v else pd.DataFrame() for k, v in dfs.items()}


def get_time_columns(phase):
    if phase == "total":
        return ['Total Time', 'Penny Time Total', 'Resto Time Total', 'SVM iteraciones']
    else:
        return ['Penny Time', 'Resto Time']


# --- 2. SIDEBAR ---

st.sidebar.title("⚛️ Panel de Control")

uploaded_files = st.sidebar.file_uploader(
    "Subir excels",
    type=["xlsx"],
    accept_multiple_files=True
)

if not uploaded_files:
    st.stop()

data_dict = load_data(uploaded_files)

phase_input = st.sidebar.radio("Fase:", ["Test", "Train", "Total"], index=2, horizontal=True)
phase_key = phase_input.lower()
df_main = data_dict[phase_key]

if df_main.empty:
    st.stop()

# --- FILTROS GLOBALES ---
st.sidebar.subheader("Filtros Globales")

sel_machines = st.sidebar.multiselect("Machine", sorted(df_main['Machine'].unique()),
                                      default=sorted(df_main['Machine'].unique()))
sel_backends = st.sidebar.multiselect("Backend", sorted(df_main['Backend'].unique()),
                                      default=sorted(df_main['Backend'].unique()))
sel_qubits = st.sidebar.multiselect("Qubits", sorted(df_main['Qubits'].unique()),
                                    default=sorted(df_main['Qubits'].unique()))
sel_affinity = st.sidebar.multiselect("Affinity", sorted(df_main['Affinity'].unique()),
                                      default=sorted(df_main['Affinity'].unique()))
sel_cores = st.sidebar.multiselect("Cores", sorted(df_main['Cores'].unique()),
                                    default=sorted(df_main['Cores'].unique()))
#sel_blocktype = st.sidebar.multiselect("Block Type", sorted(df_main['Block Type'].unique()),
#                                    default=sorted(df_main['Block Type'].unique()))

df_filtered = df_main[
    (df_main['Machine'].isin(sel_machines))  &
    (df_main['Backend'].isin(sel_backends))  &
    (df_main['Qubits'].isin(sel_qubits))     &
    (df_main['Affinity'].isin(sel_affinity)) &
    (df_main['Cores'].isin(sel_cores))       
    #(df_main['Block Type'].isin(sel_blocktype))
]

# --- DASHBOARD ---
st.title(f"Dashboard de Rendimiento — {phase_input}")

time_cols = get_time_columns(phase_key)

tabs = st.tabs(["📊 Exploración Visual", "🧮 Estadística", "⏱️ Tiempos", "💾 Datos"])

# ================= TAB 1 =================
with tabs[0]:
    y_axis = st.selectbox("Métrica (Y)", time_cols)
    x_axis = st.selectbox("Eje X", ['Qubits', 'Backend', 'Machine', 'Mode', 'Block Type', 'Affinity', 'Cores'])
    #color_dim = st.selectbox("Color", ['Machine', 'Backend', 'Mode', 'Block Type', 'Affinity', 'Cores', 'Qubits'])
    color_dims = st.multiselect(
                "Color (una o dos variables)",
                ['Machine', 'Backend', 'Mode', 'Block Type', 'Affinity', 'Cores', 'Qubits'],
                default=['Machine'],
                max_selections=2
    )

    df_plot = df_filtered.copy()

    if len(color_dims) == 1:
        color_col = color_dims[0]

    elif len(color_dims) == 2:
        color_col = "ColorCombo"
        df_plot[color_col] = (
            df_plot[color_dims[0]].astype(str)
            + " | "
            + df_plot[color_dims[1]].astype(str)
        )

    else:
        color_col = None

    chart_type = st.radio(
        "Tipo de Gráfico",
        [
            "Boxplot",
            "Violin",
            "Barras",
            "Líneas",
            "Líneas (Log Y)",
            "Heatmap"
        ],
        horizontal=True
    )

    if chart_type == "Boxplot":
        fig = px.box(df_plot, x=x_axis, y=y_axis, color=color_dim, points="all")

    elif chart_type == "Violin":
        fig = px.violin(df_plot, x=x_axis, y=y_axis, color=color_dim, box=True, points="all")

    elif chart_type == "Barras":
        grp = [x_axis, color_dim] if x_axis != color_dim else [x_axis]
        df_stats = df_plot.groupby(grp)[y_axis].agg(['mean', 'std']).reset_index()
        fig = px.bar(df_stats, x=x_axis, y='mean', error_y='std', color=color_dim, barmode='group')

    elif chart_type == "Líneas":
        df_stats = df_plot.groupby([x_axis, color_dim])[y_axis].mean().reset_index()
        fig = px.line(df_stats, x=x_axis, y=y_axis, color=color_dim, markers=True)

    elif chart_type == "Líneas (Log Y)":
        df_log = df_plot[df_filtered[y_axis] > 0]
        df_stats = df_log.groupby([x_axis, color_dim])[y_axis].mean().reset_index()

        fig = go.Figure()
        for _, sub in df_stats.groupby([color_dim]):
            mode = "lines+markers" 
            fig.add_trace(go.Scatter(
                x=sub[x_axis],
                y=sub[y_axis],
                mode=mode,
                name=f"{sub[color_dim].iloc[0]}"
            ))

        fig.update_layout(
            yaxis_type="log",
            yaxis_title=f"{y_axis} (log)",
            title="Escalabilidad con escala logarítmica en Y"
        )

    elif chart_type == "Heatmap":
        pivot = df_plot.pivot_table(values=y_axis, index=x_axis, columns=color_dim, aggfunc='mean')
        fig = px.imshow(pivot, text_auto=".2f", aspect="auto")

    st.plotly_chart(fig, use_container_width=True)

# ================= TAB 2 =================
with tabs[1]:
    stat_target = st.selectbox("Variable Numérica", time_cols)
    stat_factor = st.selectbox(
        "Factor",
        ['Machine', 'Backend', 'Mode', 'Block Type', 'Qubits', 'Affinity', 'Cores']
    )

    if st.button("Ejecutar Análisis"):
        data = df_filtered[[stat_target, stat_factor]].dropna()
        groups = data[stat_factor].unique()
        group_data = [data[data[stat_factor] == g][stat_target] for g in groups]

        normal = all(len(g) >= 3 and stats.shapiro(g)[1] > 0.05 for g in group_data)
        homo = stats.levene(*group_data)[1] > 0.05 if len(group_data) > 1 else False

        if len(groups) == 2:
            if normal:
                _, p = stats.ttest_ind(*group_data, equal_var=homo)
                test = "T-Test"
            else:
                _, p = stats.mannwhitneyu(*group_data)
                test = "Mann-Whitney"
        else:
            if normal and homo:
                _, p = stats.f_oneway(*group_data)
                test = "ANOVA"
            else:
                _, p = stats.kruskal(*group_data)
                test = "Kruskal-Wallis"

        st.info(f"Test aplicado: **{test}**")
        st.metric("P-Value", f"{p:.4e}")

# ================= TAB 3 =================
with tabs[2]:
    stack_cols = ['Penny Time Total', 'Resto Time Total'] if phase_key == "total" else ['Penny Time', 'Resto Time']
    stack_group = st.selectbox(
        "Agrupar por",
        ['Backend', 'Machine', 'Qubits', 'Mode', 'Block Type', 'Affinity', 'Cores']
    )

    df_stack = df_filtered.groupby(stack_group)[stack_cols].mean().reset_index()
    df_melt = df_stack.melt(id_vars=stack_group, value_vars=stack_cols,
                            var_name="Componente", value_name="Segundos")

    fig = px.bar(df_melt, x=stack_group, y="Segundos", color="Componente", barmode="stack")
    st.plotly_chart(fig, use_container_width=True)

# ================= TAB 4 =================
with tabs[3]:
    st.dataframe(df_filtered, use_container_width=True)
    st.download_button(
        "Descargar CSV",
        df_filtered.to_csv(index=False),
        file_name=f"data_{phase_key}.csv"
    )
