

import numpy as np
import pandas as pd
import streamlit as st
import validators

from recommender import Recommender


@st.cache_data
def load_movies() -> pd.DataFrame:

    movies = pd.read_csv("./data/movies_imdb.csv")
    return movies


@st.cache_data
def get_random_movies_to_rate(num_movies: int = 5) -> pd.DataFrame:

    movies = load_movies()

    movies = movies.sort_values("imdb_rating", ascending=False).reset_index(drop=True)
    movies = movies[:100]

    select = np.random.choice(movies.index, size=num_movies, replace=False)

    return movies.iloc[select]


@st.cache_data
def get_movies() -> pd.DataFrame:

    movies = load_movies()
    movies = movies.sort_values("title").reset_index(drop=True)

    return movies


@st.cache_data
def get_movie_id_from_title(title_str: str) -> int:

    movies = load_movies()
    movies = movies[movies["title"] == title_str]["movie_id"]

    return int(movies.iloc[0])


def prepare_query_favourites() -> dict:


    data = get_movies()

    st.markdown(
        "오늘 무슨 영화 볼지 고민 중이신가요?"
        "**최애 영화 선택**  "
        "하시면 취향 바탕으로 추천해 드릴게요."
    )

    user_ratings = st.multiselect(
        "영화 선택을 자유롭게 해주세요",
        data["title"],
    )

    query = {}
    for title_selected in user_ratings:
        # Get movie ids
        mid = get_movie_id_from_title(title_selected)
        # Set rating to 5 for selected movies
        query[mid] = 5

    return query


def prepare_query_rating() -> dict:

    data = get_random_movies_to_rate(10)

    st.markdown(
        "어떤 영화를 봐야 할지 모르겠나요? 여기 10가지 영화를 무작위로 골라 봤어요."
        " **영화를 보시고 평점을 매기시면**  "
        "저희가 추천해 드릴게요."
    )

    query = {}
    for movie_id, title in zip(data["movie_id"], data["title"]):
        query[movie_id] = st.select_slider(title, options=[0, 1, 2, 3, 4, 5])

    return query


def recommender(rec_type: str = "fav") -> None:



    query = (
        prepare_query_rating() if rec_type == "rating" else prepare_query_favourites()
    )


    method_select = st.selectbox(
        "알고리즘을 선택해 주세요",
        ["Nearest Neighbors", "Non-negative matrix factorization"],
        key="method_selector_" + rec_type,
    )


    method = "neighbors" if method_select == "Nearest Neighbors" else "nmf"

    num_movies = st.slider(
        "몇개의 영화를 추천해 드릴까요?",
        min_value=1,
        max_value=10,
        value=5,
        key="num_movies_slider_" + rec_type,
    )


    if st.button("영화 추천!", key="button_" + rec_type):
        with st.spinner(f"Calculating recommendations using {method_select}..."):
            recommend = Recommender(query, method=method, k=num_movies)
            movie_ids, _ = recommend.recommend()

        with st.spinner("Fetching movie information from IMDB..."):
            st.write("Recommended movies using Nearest Neighbors:\n")
            for movie_id in movie_ids:
                display_movie(movie_id)


def display_movie(movie_id: int) -> None:

    movies = load_movies()
    movie = movies[movies["movie_id"] == movie_id].copy()

    col1, col2 = st.columns([1, 4])

    with col1:
        if validators.url(str(movie["cover_url"].iloc[0])):
            st.image(movie["cover_url"].iloc[0])

    with col2:
        if not pd.isnull(movie["title"].iloc[0]) and not pd.isnull(
            movie["year"].iloc[0]
        ):
            st.header(f"{movie['title'].iloc[0]} ({movie['year'].iloc[0]})")
        if not pd.isnull(movie["imdb_rating"].iloc[0]):
            st.markdown(f"**IMDB-rating:** {movie['imdb_rating'].iloc[0]}/10")
        if not pd.isnull(movie["genre"].iloc[0]):
            st.markdown(f"**Genres:** {', '.join(movie['genre'].iloc[0].split(' | '))}")
        if not pd.isnull(movie["director"].iloc[0]):
            st.markdown(
                f"**Director(s):** {', '.join(movie['director'].iloc[0].split('|'))}"
            )
        if not pd.isnull(movie["cast"].iloc[0]):
            st.markdown(
                f"**Cast:** {', '.join(movie['cast'].iloc[0].split('|')[:10])}, ..."
            )
        if not pd.isnull(movie["plot"].iloc[0]):
            st.markdown(f"{movie['plot'].iloc[0]}")
        if validators.url(str(movie["url"].iloc[0])):
            st.markdown(f"[Read more on imdb.com]({movie['url'].iloc[0]})")
    st.divider()


# 제목
st.set_page_config(page_title="영화 추천해 드릴게요")

# 이미지
st.image("data/movies.jpg")


st.title("오늘 어떤 영화 보실래요?")
st.subheader("영화 추천 해드릴게요")

tab1, tab2 = st.tabs(["최애 영화", "나만의 평점"])

with tab1:
    recommender("fav")

with tab2:
    recommender("rating")
