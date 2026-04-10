from pathlib import Path

import pandas as pd
import streamlit as st

from autoint import CAT_COLS, load_artifact_model, predict_top_k


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_ROOT = PROJECT_ROOT / "model" / "baseline_tf_halfday"


def resolve_best_artifact(model_root: Path) -> Path:
    result_files = sorted(model_root.glob("batch_results_*.csv"))
    if result_files:
        results_df = pd.read_csv(result_files[-1])
        sort_cols = [col for col in ["valid_ndcg_at_10", "valid_hitrate_at_10", "valid_auc"] if col in results_df.columns]
        if sort_cols:
            best_row = results_df.sort_values(sort_cols, ascending=[False] * len(sort_cols)).iloc[0]
        else:
            best_row = results_df.iloc[0]
        return PROJECT_ROOT / str(best_row["artifact_dir"])

    candidate_dirs = sorted(path.parent for path in model_root.glob("*/config.json"))
    if not candidate_dirs:
        raise FileNotFoundError("사용 가능한 모델 artifact 를 찾지 못했습니다.")
    return candidate_dirs[0]


# streamlit run streamlit/show_st.py
@st.cache_resource
def load_data():
    """
    앱에서 보여줄 필요 데이터를 가져오는 함수입니다.
    - 사용자, 영화, 평점 데이터를 가져옵니다.
    - 노트북에서 저장한 최적 모델도 불러옵니다.
    """
    artifact_dir = resolve_best_artifact(MODEL_ROOT)
    model, encoder_maps, _ = load_artifact_model(artifact_dir)

    source_df = pd.read_csv(DATA_DIR / "movielens_rcmm_v2.csv")
    movies_df = pd.read_csv(DATA_DIR / "movies_prepro.csv")
    users_df = pd.read_csv(DATA_DIR / "users_prepro.csv")
    ratings_df = pd.read_csv(DATA_DIR / "ratings_prepro.csv")

    supported_user_ids = {int(user_id) for user_id in encoder_maps["user_id"].keys()}
    supported_movie_ids = {int(movie_id) for movie_id in encoder_maps["movie_id"].keys()}

    users_df = users_df[users_df["user_id"].isin(supported_user_ids)].copy()
    movies_df = movies_df[movies_df["movie_id"].isin(supported_movie_ids)].copy()
    ratings_df = ratings_df[
        ratings_df["user_id"].isin(supported_user_ids) & ratings_df["movie_id"].isin(supported_movie_ids)
    ].copy()
    source_df = source_df[
        source_df["user_id"].isin(supported_user_ids) & source_df["movie_id"].isin(supported_movie_ids)
    ].copy()

    movie_meta_df = (
        source_df[["movie_id", "movie_decade", "movie_year", "genre1", "genre2", "genre3"]]
        .fillna("no")
        .drop_duplicates(subset="movie_id")
        .merge(movies_df[["movie_id", "title"]], on="movie_id", how="left")
    )
    user_meta_df = (
        source_df[["user_id", "gender", "age", "occupation", "zip"]]
        .fillna("no")
        .drop_duplicates(subset="user_id")
    )

    return users_df, movies_df, ratings_df, source_df, movie_meta_df, user_meta_df, model, encoder_maps


def get_user_seen_movies(ratings_df):
    """
    사용자가 과거에 보았던 영화 리스트를 가져옵니다.
    """
    return ratings_df.groupby("user_id")["movie_id"].apply(list).reset_index()


def get_user_non_seed_dict(movies_df, user_df, user_seen_movies):
    """
    사용자가 보지 않았던 영화 리스트를 가져옵니다.
    """
    unique_movies = set(movies_df["movie_id"].unique().tolist())
    user_seen_map = dict(zip(user_seen_movies["user_id"], user_seen_movies["movie_id"]))
    user_non_seen_dict = {}

    for user in user_df["user_id"].unique():
        user_seen_movie_list = user_seen_map.get(user, [])
        user_non_seen_dict[user] = sorted(unique_movies - set(user_seen_movie_list))

    return user_non_seen_dict


def get_user_info(user_id):
    """
    사용자 정보를 가져옵니다.
    """
    return users_df[users_df["user_id"] == user_id]


def get_user_past_interactions(user_id):
    """
    사용자 평점 데이터 중 4점 이상(선호했다는 정보)만 가져옵니다.
    """
    return ratings_df[(ratings_df["user_id"] == user_id) & (ratings_df["rating"] >= 4)].merge(
        movies_df, on="movie_id", how="left"
    )


def get_recom(user, user_non_seen_dict, user_df, movies_df, r_year, r_month, model, encoder_maps):
    """
    아래와 같은 순서로 추천 결과를 가져옵니다.
    1. streamlit에서 입력 받은 타겟 월, 연도, 사용자 정보를 받아옴
    2. 사용자가 보지 않았던 정보 추출
    3. model input으로 넣을 수 있는 형태로 데이터프레임 구성
    4. 새 모델 인코딩 규칙을 적용
    5. 모델 predict 수행
    """
    user_non_seen_movie = user_non_seen_dict.get(user, [])
    user_info = user_df[user_df["user_id"] == user]

    if not user_non_seen_movie or user_info.empty:
        return movies_df.iloc[0:0].copy()

    user_row = user_info.iloc[0]
    rating_year = int(r_year)
    rating_month = int(r_month)
    rating_decade = f"{rating_year - (rating_year % 10)}s"

    candidate_df = movie_meta_df[movie_meta_df["movie_id"].isin(user_non_seen_movie)].copy()
    candidate_df["user_id"] = int(user)
    candidate_df["rating_year"] = rating_year
    candidate_df["rating_month"] = rating_month
    candidate_df["rating_decade"] = rating_decade
    candidate_df["gender"] = user_row["gender"]
    candidate_df["age"] = user_row["age"]
    candidate_df["occupation"] = user_row["occupation"]
    candidate_df["zip"] = user_row["zip"]
    candidate_df = candidate_df[CAT_COLS]

    ranked_df = predict_top_k(model, candidate_df, encoder_maps, top_k=10)
    return ranked_df[["movie_id"]].merge(movies_df, on="movie_id", how="left")


# 데이터 로드
users_df, movies_df, ratings_df, source_df, movie_meta_df, user_meta_df, model, encoder_maps = load_data()
supported_user_ids = set(users_df["user_id"].tolist())
user_seen_movies = get_user_seen_movies(ratings_df)
user_non_seen_dict = get_user_non_seed_dict(movies_df, users_df, user_seen_movies)

# 타이틀
st.title("영화 추천 결과 살펴보기")

st.header("사용자 정보를 넣어주세요.")
user_id = st.number_input(
    "사용자 ID 입력",
    min_value=int(users_df["user_id"].min()),
    max_value=int(users_df["user_id"].max()),
    value=int(users_df["user_id"].min()),
)
r_year = st.number_input(
    "추천 타겟 연도 입력",
    min_value=int(source_df["rating_year"].min()),
    max_value=int(source_df["rating_year"].max()),
    value=int(source_df["rating_year"].min()),
)
r_month = st.number_input(
    "추천 타겟 월 입력",
    min_value=int(source_df["rating_month"].min()),
    max_value=int(source_df["rating_month"].max()),
    value=int(source_df["rating_month"].min()),
)


# streamlit run streamlit/show_st.py --client.showErrorDetails=false
if st.button("추천 결과 보기"):
    user_id = int(user_id)

    if user_id not in supported_user_ids:
        st.error("학습된 모델에서 지원하지 않는 사용자 ID입니다.")
    else:
        st.write("사용자 기본 정보")
        user_info = get_user_info(user_id)
        st.dataframe(user_info)

        st.write("샤용자가 과거에 봤던 이력(평점 4점 이상)")
        user_interactions = get_user_past_interactions(user_id)
        st.dataframe(user_interactions)

        st.write("추천 결과")
        recommendations = get_recom(
            user_id,
            user_non_seen_dict,
            users_df,
            movies_df,
            int(r_year),
            int(r_month),
            model,
            encoder_maps,
        )
        st.dataframe(recommendations)
