
# 📊 Medical Costs Dashboard

Aplicativo interativo com Streamlit para prever e analisar os custos médicos de pacientes com base em variáveis como idade, IMC, número de filhos, hábito de fumar e região.

---

## Membro - José Roberto Cândido da Silva RM - 363845

## ✅ Funcionalidades

- Upload de arquivos CSV com dados de pacientes
- Visualização de estatísticas descritivas e gráficos interativos
- Treinamento de modelos de regressão:
  - Regressão Linear
  - Árvore de Decisão
  - Ambos ao mesmo tempo com gráfico comparativo
- Validação estatística com p-values e intervalos de confiança
- Download do modelo treinado e das previsões
- Armazenamento de relatórios no MongoDB

---

## 🚀 Como rodar o projeto

### 1. Clone o repositório
```bash
git clone https://github.com/Roberto-Candido/challenge_fiap.git
cd challenge_fiap
```

### 2. Crie e ative o ambiente virtual
#### Windows
```bash
python -m venv .venv
.venv\Scripts\activate
```

#### Linux/macOS
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Instale as dependências
```bash
pip install -r requirements.txt
```

### 4. (Opcional) Configure o MongoDB
Crie um arquivo `config.yaml` na raiz com:

```yaml
mongodb:
  host: "localhost"
  port: 27017
  database: "medical_costs"
  collection: "relatorios_modelo"
```

---

## ▶️ Execute o aplicativo
```bash
streamlit run app.py
```

---

## 📦 Estrutura de Diretórios

```
medical_costs_app/
├── app.py
├── config.yaml
├── requirements.txt
├── saved_models/
├── results/
├── logs/
├── services/
│   └── model_service.py
├── utils/
│   ├── data_loader.py
│   ├── visualizations.py
│   ├── logger.py
│   └── config_loader.py
```

---

## 📊 Exemplos de Gráficos

- Distribuição de encargos
- Média por região ou hábito
- Boxplot por categoria
- Mapa de correlação
- Comparação entre modelos preditivos

---

## 📌 Requisitos

- Python 3.8+
- Streamlit
- Scikit-learn
- Plotly
- Statsmodels
- Pymongo
- PyYAML

---

Desenvolvido com ❤️ para fins educacionais e exploratórios.
