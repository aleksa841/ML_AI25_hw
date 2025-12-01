import streamlit as st
import pandas as pd
import plotly.express as px
import pickle
from pathlib import Path
import phik

st.set_page_config(page_title='Приложение для предсказания цены авто', layout='wide')

MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_PATH = MODEL_DIR / 'ridge_model.pkl'

@st.cache_resource
def load_model():

    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)

    return model

try:
    model = load_model()
except Exception as e:
    st.error(f'Ошибка при загрузке модели: {e}, повторите попытку')
    st.stop()

# --- Чтение csv для EDA ---

st.title('🚗 Приложение для предсказания цены авто')
file_for_eda = st.file_uploader('Загрузите, пожалуйста, файл для EDA', type=['csv'])
if file_for_eda is None:
    st.info('Нужно обязательно загрузить файл')
    st.stop()

df = pd.read_csv(
            file_for_eda, 
            encoding='utf-8',
            on_bad_lines='skip'
        )

convert_types = {
                'name': 'string',
                'year': 'Int64',
                'selling_price': 'string',
                'km_driven': 'Int64',
                'fuel': 'string',
                'seller_type': 'string',
                'transmission': 'string',
                'owner': 'string',
                'mileage': 'float',
                'engine': 'float',
                'max_power': 'float',
                'torque': 'float',
                'seats': 'Int64',
                'max_torque_rpm': 'float'
            }

df = df.astype(convert_types)

# --- EDA часть ---

st.subheader('📊 EDA')
df_describe_num = df.describe(include='number')
df_describe_cat = df.describe(exclude='number')

st.write('#### Пример данных')
st.write(df.head())

st.write('#### Описательные статистики по колонкам')
tab1, tab2 = st.tabs(['Числовые', 'Категориальные'])
with tab1:
    st.dataframe(df_describe_num.style.format(precision=1))

with tab2:
    st.dataframe(df_describe_cat)

st.write('#### Взаимосвязь признаков с целевой переменной selling_price')
tabs = []
df_cols = df.drop('selling_price', axis=1).columns
num_cols = df.select_dtypes('number').columns
cat_cols = df.select_dtypes(exclude='number').columns
tabs = st.tabs(list(df_cols))

for col_name, tab in zip(df_cols, tabs):
    with tab:
        if col_name in num_cols:
            fig = px.scatter(
                df,
                x=col_name,
                y='selling_price',
                color='name',
            )
        else:
            fig = px.box(
                df,
                x=col_name,
                y='selling_price',
            )

        st.plotly_chart(fig, theme='streamlit')

st.write('#### Кореляция признаков (Phi)')
ph = df.phik_matrix()
st.dataframe(ph.style.background_gradient(cmap='Greens').format(precision=3))

# --- Модель: веса ---

st.subheader('✨Модель')
st.write('#### Веса модели по убыванию')
prep_part = model.named_steps['prep']
col_names_after_prep = model.named_steps['prep'].get_feature_names_out()
weights, intercept = model.named_steps['ridge_model'].coef_, model.named_steps['ridge_model'].intercept_
df_weights = pd.Series(dict(zip(col_names_after_prep, weights)))\
    .reset_index()\
    .rename({'index': 'feature',
             0: 'weight'}, axis=1)

df_weights.loc[len(df_weights)] = ['intercept', intercept]
df_weights = df_weights.sort_values(by='weight', key=lambda x: abs(x), ascending=False, ignore_index=True)

def color_w(val):
    if val > 0:
        return 'color: lightgreen; font-weight: bold' 
    elif val < 0:
        return 'color: red; font-weight: bold'
    else:
        return 'color: white; font-weight: bold'

st.dataframe(df_weights.style.applymap(color_w, subset=['weight']).format(precision=1), key=lambda x: abs(x))

# --- Форма для предсказания ---
st.write('#### Предсказать цену авто')
st.write('###### Вы можете заполнить форму или загрузить csv с признаками объектов')
with st.form('prediction_form'):
    col_cat, col_num = st.columns(2)
    input_data = {}

    with col_cat:
        st.write('Категориальные признаки')
        for col in cat_cols:
            unique_vals = df[col].unique()
            input_data[col] = st.selectbox(col, unique_vals)

    with col_num:
        st.write('Числовые признаки')
        for col in num_cols:
            if col == 'seats':
                min_val, max_val = df[col].min(), df[col].max()
                input_data[col] = st.slider(col, min_value=min_val, max_value=max_val)
            else:
                med_val = df[col].median()
                input_data[col] = st.number_input(col, value=med_val, step=1.0) 

    file_for_preds = st.file_uploader('ИЛИ загрузите CSV файл с данными для предсказания', type=['csv'])
    submitted = st.form_submit_button('Предсказать цену', use_container_width=True)

# --- Модель: предсказание ---

if submitted:
    try:
        if file_for_preds is not None:
            df_for_preds = pd.read_csv(
                file_for_preds, 
                encoding='utf-8',
                on_bad_lines='skip'
            )
            
            df_for_preds = df_for_preds.astype(convert_types)
            preds = model.predict(df_for_preds)

            st.success(f'💚 Сделано {len(preds)} предсказаний')
            preds_df = pd.DataFrame({
                'Наблюдение': range(len(preds)),
                'Предсказанная цена': preds.round(0).astype(int)
            })
            st.dataframe(preds_df, hide_index=True)

        else:
            input_df = pd.DataFrame(input_data, index=[0])
            preds = model.predict(input_df)[0]

            st.success(f'💚 Предсказанная цена  авто: {preds.round(0)}')
    except Exception as e:
        st.error(f'❌ Ошибка при попытке сделать предсказание: {e}')