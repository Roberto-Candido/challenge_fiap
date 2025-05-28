import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

from utils.data_loader import load_data, preprocess_data
from services.model_service import train_and_evaluate_model, save_model, save_predictions
from utils.visualizations import (
    plot_distribution,
    plot_predictions,
    grafico_media_encargos_por_categoria,
    grafico_boxplot,
    grafico_correlacao,
    grafico_dispersao, plot_comparativo_modelos
)

st.set_page_config(page_title="Dashboard de Custos Médicos", layout="wide")

st.markdown("<h1 style='text-align: center; color: #4CAF50;'>📊 Dashboard de Custos Médicos</h1>", unsafe_allow_html=True)
st.markdown("---")

uploaded_file = st.file_uploader("📁 Carregue o arquivo CSV com os dados", type=["csv"])

if uploaded_file:
    df = load_data(uploaded_file)

    st.markdown("### 🔍 Estatísticas Descritivas")
    st.dataframe(df.describe(), use_container_width=True)

    st.markdown("### 📈 Distribuição dos Encargos")
    plot_distribution(df)

    st.markdown("### 📊 Análise por Categoria")
    with st.expander("Ver gráficos por categoria"):
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(grafico_media_encargos_por_categoria(df, "região"))
        with col2:
            st.plotly_chart(grafico_boxplot(df, "fumante"))

        st.plotly_chart(grafico_correlacao(df))
        st.plotly_chart(grafico_dispersao(df, "imc"))

    df_processed = preprocess_data(df)

    st.markdown("---")
    st.markdown("### 🧠 Treinamento do Modelo")

    model_type = st.selectbox("Selecione o modelo:", [
        "Regressão Linear",
        "Árvore de Decisão",
        "Regressão Linear + Árvore de Decisão"
    ])

    if st.button("🚀 Treinar e Avaliar Modelo"):
        X = df_processed.drop("encargos", axis=1)
        y = df_processed["encargos"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if model_type == "Regressão Linear + Árvore de Decisão":
            modelo_lr = LinearRegression().fit(X_train, y_train)
            modelo_tree = DecisionTreeRegressor().fit(X_train, y_train)
            y_pred_lr = modelo_lr.predict(X_test)
            y_pred_tree = modelo_tree.predict(X_test)

            st.markdown("### 📉 Comparativo de Previsões")
            fig = plot_comparativo_modelos(y_test, y_pred_lr, y_pred_tree)
            st.plotly_chart(fig, use_container_width=True)

        else:
            results = train_and_evaluate_model(df_processed, model_type)

            col1, col2, col3 = st.columns(3)
            col1.metric("R²", f"{results['metrics']['r2']:.2f}")
            col2.metric("MAE", f"R$ {results['metrics']['mae']:.2f}")
            col3.metric("MSE", f"R$ {results['metrics']['mse']:.2f}")

            st.markdown("### 🔬 Validação Estatística")
            st.write("**P-values:**", results["metrics"]["pvalues"])
            st.write("**Intervalos de Confiança:**", results["metrics"]["conf_intervals"])

            st.markdown("### 📉 Previsão vs Real")
            st.plotly_chart(plot_predictions(results["y_test"], results["y_pred"]), use_container_width=True)

            model_path = save_model(results["model"], model_type.lower().replace(' ', '_'))
            csv_path = save_predictions(results["y_test"], results["y_pred"])

            col4, col5 = st.columns(2)
            with col4:
                with open(model_path, "rb") as f:
                    st.download_button("📦 Baixar Modelo", f, file_name=model_path.split("/")[-1])
            with col5:
                with open(csv_path, "rb") as f:
                    st.download_button("📄 Baixar Resultados", f, file_name="predicoes.csv")
