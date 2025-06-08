import streamlit as st
import pandas as pd
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

@st.cache_data
def load_and_prepare_data():    
    
    df = pd.read_csv('new_dataset.csv')  
    
    def tokenize_track_name(track_name):
        fixed = track_name.lower().strip().replace(' ', '_').replace('(', '').replace(')', '').replace("'", '')
        tokens = word_tokenize(fixed)
        return tokens
    
    df['tokenized'] = df['track_name'].apply(tokenize_track_name)
    model = Word2Vec(sentences=df['tokenized'], vector_size=100, window=5, min_count=1, workers=4)
    
    numerical_features = ['acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'speechiness', 'tempo', 'valence']
    scaler = StandardScaler()
    df[numerical_features] = scaler.fit_transform(df[numerical_features])

    return df, model, numerical_features

df, model, numerical_features = load_and_prepare_data()
    
# Установка фона
st.markdown(
    """
    <style>
    .stApp {
        background-image: url('https://img.freepik.com/free-photo/beautiful-feathers-arrangement_23-2151436571.jpg');
        background-size: cover;
        background-position: center;
        color: white;  /* Цвет текста */
    }
    .stButton > button {
        background-color: rgba(255, 255, 255, 0.3);
        color: black;
        font-size: 16px;
        border-radius: 10px;
        transition: 0.3s;
    }
    .stButton > button:hover {
        background-color: rgba(255, 255, 255, 0.7);
        color: black;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Заголовок приложения
st.title('Система рекомендаций треков')

def display_recommendations_with_checkboxes():
    st.write("**Рекомендованные треки:**")
    
    # Синхронизация длины relevance_flags с количеством рекомендаций
    recommendations = st.session_state.recommendations
    if len(st.session_state.relevance_flags) < len(recommendations):
        st.session_state.relevance_flags.extend([False] * (len(recommendations) - len(st.session_state.relevance_flags)))
    elif len(st.session_state.relevance_flags) > len(recommendations):
        st.session_state.relevance_flags = st.session_state.relevance_flags[:len(recommendations)]
    
    # Отображение треков с флажками
    for i, (track, artist, score) in enumerate(recommendations):
        col1, col2 = st.columns([4, 1])
        col1.write(f"{track} - {artist} ({score})")
        # Управляем состоянием флажка
        st.session_state.relevance_flags[i] = col2.checkbox(
            label="", 
            value=st.session_state.relevance_flags[i],
            key=f"checkbox_{i}"
        )

# Инициализация переменных состояния
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = []
if 'relevance_flags' not in st.session_state:
    st.session_state.relevance_flags = []

num_recommendations = st.slider("Выберите количество рекомендаций", min_value=1, max_value=20, value=5)

def get_combined_recommendations(track, artist, df, model, num_recommendations):
    track_normalized = track.lower().strip().replace(' ', '_').replace('(', '').replace(')', '').replace("'", '')
    artist_normalized = artist.lower().strip().replace(' ', '_').replace('(', '').replace(')', '').replace("'", '')

    if track_normalized not in model.wv:
        st.error(f"Трек '{track}' не найден в модели.")
        return []

    by_name = model.wv.most_similar(track_normalized, topn=num_recommendations)
    by_name = [(similar[0], round(similar[1], 3)) for similar in by_name]

    df['track_name_normalized'] = df['track_name'].str.lower().str.strip().str.replace(' ', '_').str.replace('(', '').str.replace(')', '').str.replace("'", '')
    df['artist_name_normalized'] = df['artist_name'].str.lower().str.strip().str.replace(' ', '_').str.replace('(', '').str.replace(')', '').str.replace("'", '')

    track_index = df[(df['track_name_normalized'] == track_normalized) & (df['artist_name_normalized'] == artist_normalized)].index
    if len(track_index) == 0:
        st.warning(f"Трек '{track}' от '{artist}' не найден в датасете.")
        return []

    track_index = track_index[0]
    target_features = df.loc[track_index, numerical_features].values.reshape(1, -1)
    by_features = cosine_similarity(target_features, df[numerical_features]) 
    similar_indices = by_features[0].argsort()[-num_recommendations-1:-1][::-1]    
    by_features = [(df.iloc[i]['track_name'], round(by_features[0][i], 3)) for i in similar_indices]

    combined_recommendations = {name: similarity for name, similarity in by_name} 
    for name, similarity in by_features:
        if name not in combined_recommendations:
            combined_recommendations[name] = similarity

    genre_weight = 5 
    popularity_weight = 1.3 
    artist_weight = 10  
    
    final_scores = {}
    for name, sim in combined_recommendations.items():
        genre_score = 0
        popularity_score = 0
        artist_score = 0
        
        genre_row = df[df['track_name'] == name] 
        if not genre_row.empty:
            genre_score = genre_row['genre'].values[0] 
            genre_score = 1 if genre_score in df.loc[track_index, 'genre'] else 0  
        
        popularity_row = df[df['track_name'] == name] 
        if not popularity_row.empty:
            popularity_score = popularity_row['popularity'].values[0]  

        artist_row = df[df['track_name'] == name] 
        if not artist_row.empty:
            artist_score = artist_weight if artist_row['artist_name_normalized'].values[0] == artist_normalized else 0
        
        final_score = (sim * 0.5 + 
                       (genre_weight * genre_score) + 
                       (popularity_weight * popularity_score) + 
                       artist_score)
        final_scores[name] = min(final_score, 100)  

    sorted_recommendations = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)

    formatted_recommendations = []
    for name, score in sorted_recommendations[:num_recommendations]:
        artist_name = df[df['track_name'] == name]['artist_name'].values[0]  
        formatted_recommendations.append((name, artist_name, round(score, 1))) 

    return formatted_recommendations

col1, col2 = st.columns(2)
with col1:
    track = st.text_input('Введите название трека:', placeholder="например: Roar")
with col2:
    artist = st.text_input('Введите имя артиста:', placeholder="например: Katy Perry")

# Инициализация переменной recommendations
recommendations = []

if st.button('Получить рекомендации'):
    recommendations = get_combined_recommendations(track, artist, df, model, num_recommendations)
    if recommendations:
        st.session_state.recommendations = recommendations
        st.session_state.relevance_flags = [False] * len(recommendations)  # Сбрасываем флажки
    else:
        st.write('Рекомендации не найдены.')

# Отображение рекомендаций, если они есть
if st.session_state.recommendations:
    display_recommendations_with_checkboxes()

# Определение метрик
def calculate_precision_at_k(relevant_flags, k):
    relevant_at_k = sum(relevant_flags[:k])  # Количество релевантных рекомендаций в топ-K
    return relevant_at_k / k if k > 0 else 0

def calculate_mrr(relevant_flags):
    for i, flag in enumerate(relevant_flags):
        if flag:  # Первый релевантный элемент
            return 1 / (i + 1)
    return 0

# Функция для визуализации
def plot_metrics(precision_at_k, mrr):
    # Установим стиль Seaborn
    sns.set_theme(style="whitegrid")

    # Данные для графика
    metrics = ["Precision@K", "MRR"]
    values = [precision_at_k, mrr]
    colors = ["#1f77b4", "#ff7f0e"]

    # Создаем фигуру
    fig, ax = plt.subplots(figsize=(8, 6))

    # Построение графика
    bars = ax.bar(metrics, values, color=colors, edgecolor="black", linewidth=1.2)
    ax.set_ylim(0, 1.1)  # Устанавливаем диапазон по оси Y

    # Подписи над столбцами
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=14,
            color="black",
            fontweight="bold",
        )

    # Настройки для осей
    ax.set_title(
        "Результаты метрик рекомендационной системы",
        fontsize=18,
        fontweight="bold",
        color="#4a4a4a",
        pad=20,
    )
    ax.set_ylabel("Значение", fontsize=14, labelpad=10)
    ax.set_xlabel("Метрики", fontsize=14, labelpad=10)
    ax.tick_params(axis="both", which="major", labelsize=12)

    # Убираем верхнюю и правую границы графика
    sns.despine()

    # Возвращаем фигуру
    return fig

# Пояснения метрик
st.markdown("""
### Описание метрик:
- **Precision@K:** Показывает долю релевантных рекомендаций в первых K позициях. Чем выше Precision@K, тем точнее рекомендации.
- **MRR (Mean Reciprocal Rank):** Среднее обратное значение ранга первой релевантной рекомендации. Значение ближе к 1 означает, что релевантные рекомендации находятся в верхних позициях.
""")

if st.button('Показать метрики'):
    if any(st.session_state.relevance_flags):
        precision_at_k = calculate_precision_at_k(st.session_state.relevance_flags, num_recommendations)
        mrr = calculate_mrr(st.session_state.relevance_flags)
        
        st.write(f'**Precision@{num_recommendations}:** {precision_at_k:.2f}')
        st.write(f'**MRR:** {mrr:.2f}')
        
        fig_bar = plot_metrics(precision_at_k, mrr)
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.write('Отметьте релевантные рекомендации, чтобы рассчитать метрики.')
