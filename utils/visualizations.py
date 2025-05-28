
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import streamlit as st

def plot_distribution(df):
    fig = px.histogram(df, x='encargos', nbins=50, title='Distribuição de Encargos')
    st.plotly_chart(fig, use_container_width=True)

def plot_predictions(y_test, y_pred):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=y_test, mode='lines', name='Valores Reais'))
    fig.add_trace(go.Scatter(y=y_pred, mode='lines', name='Previsões'))
    fig.update_layout(title='Previsão vs Valor Real', xaxis_title='Amostras', yaxis_title='Encargos')
    return fig

def grafico_media_encargos_por_categoria(df, categoria):
    fig = px.bar(
        df.groupby(categoria)["encargos"].mean().reset_index(),
        x=categoria,
        y="encargos",
        title=f"Média de Encargos por {categoria.capitalize()}",
        labels={"encargos": "Média de Encargos"},
        color=categoria
    )
    return fig

def grafico_boxplot(df, categoria):
    fig = px.box(df, x=categoria, y="encargos", color=categoria,
                 title=f"Distribuição dos Encargos por {categoria.capitalize()}")
    return fig

def grafico_correlacao(df):
    numeric_df = df.select_dtypes(include='number')
    corr = numeric_df.corr()
    fig = ff.create_annotated_heatmap(
        z=corr.values,
        x=list(corr.columns),
        y=list(corr.index),
        annotation_text=corr.round(2).values,
        colorscale="Viridis"
    )
    fig.update_layout(title="Mapa de Correlação entre Variáveis")
    return fig

def grafico_dispersao(df, variavel):
    fig = px.scatter(
        df, x=variavel, y="encargos", color="fumante",
        title=f"Encargos vs {variavel.capitalize()}"
    )
    return fig

def plot_comparativo_modelos(y_test, y_pred_lr, y_pred_tree):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=y_test, mode='lines', name='Real', line=dict(color='blue')))
    fig.add_trace(go.Scatter(y=y_pred_lr, mode='lines', name='Regressão Linear', line=dict(color='red')))
    fig.add_trace(go.Scatter(y=y_pred_tree, mode='lines', name='Árvore de Decisão', line=dict(color='green')))
    fig.update_layout(
        title='Comparativo de Previsões',
        xaxis_title='Amostras',
        yaxis_title='Encargos',
        legend_title='Modelo'
    )
    return fig
