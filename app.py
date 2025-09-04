import streamlit as st
import requests
import time
import numpy as np
import pandas as pd

from movie_recommender import (
    load_movielens_100k,
    make_train_test,
    build_ui_matrix,
    predict_user_based,
    predict_item_based,
    predict_svd,
    recommend_top_n,
    precision_at_k
)
TMDB_API_KEY = st.secrets["TMDB_API_KEY"]

st.set_page_config(
    page_title="CineAI - Movie Recommendation System",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

:root {
    --primary-color: #6366f1;
    --primary-dark: #4f46e5;
    --secondary-color: #8b5cf6;
    --accent-color: #f59e0b;
    --success-color: #10b981;
    --background-dark: #0a0a0a;
    --background-card: #111111;
    --background-glass: rgba(255, 255, 255, 0.05);
    --text-primary: #ffffff;
    --text-secondary: #a1a1aa;
    --border-color: rgba(255, 255, 255, 0.1);
    --shadow-glow: 0 0 20px rgba(99, 102, 241, 0.3);
}

* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    background: var(--background-dark);
    color: var(--text-primary);
    overflow-x: hidden;
}

/* Animated Background */
.main::before {
    content: '';
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: 
        radial-gradient(circle at 20% 80%, rgba(99, 102, 241, 0.15) 0%, transparent 50%),
        radial-gradient(circle at 80% 20%, rgba(139, 92, 246, 0.15) 0%, transparent 50%),
        radial-gradient(circle at 40% 40%, rgba(245, 158, 11, 0.1) 0%, transparent 50%);
    animation: backgroundShift 20s ease-in-out infinite;
    pointer-events: none;
    z-index: -1;
}

@keyframes backgroundShift {
    0%, 100% { transform: translate(0, 0) rotate(0deg); }
    33% { transform: translate(-10px, -20px) rotate(1deg); }
    66% { transform: translate(20px, 10px) rotate(-1deg); }
}

/* Header Animations */
.main-header {
    text-align: center;
    padding: 60px 20px;
    position: relative;
    overflow: hidden;
}

.main-title {
    font-size: clamp(2.5rem, 5vw, 4rem);
    font-weight: 800;
    background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 50%, var(--accent-color) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 20px;
    animation: titleGlow 3s ease-in-out infinite alternate;
    letter-spacing: -0.02em;
}

@keyframes titleGlow {
    0% { filter: drop-shadow(0 0 10px rgba(99, 102, 241, 0.5)); }
    100% { filter: drop-shadow(0 0 20px rgba(139, 92, 246, 0.8)); }
}

.main-subtitle {
    font-size: 1.3rem;
    color: var(--text-secondary);
    font-weight: 400;
    animation: slideUp 1s ease-out 0.5s both;
    line-height: 1.6;
}

@keyframes slideUp {
    from { opacity: 0; transform: translateY(30px); }
    to { opacity: 1; transform: translateY(0); }
}

/* Glass Morphism Sidebar */
.sidebar .sidebar-content {
    background: var(--background-glass);
    backdrop-filter: blur(20px);
    border-right: 1px solid var(--border-color);
}

.sidebar .element-container {
    animation: fadeInRight 0.8s ease-out;
}

@keyframes fadeInRight {
    from { opacity: 0; transform: translateX(-20px); }
    to { opacity: 1; transform: translateX(0); }
}

/* Enhanced Buttons */
.stButton > button {
    background: linear-gradient(135deg, var(--primary-color), var(--primary-dark));
    color: white;
    font-weight: 600;
    font-size: 1.1rem;
    padding: 16px 32px;
    border-radius: 12px;
    border: none;
    cursor: pointer;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
    box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4);
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.stButton > button:hover {
    transform: translateY(-2px) scale(1.02);
    box-shadow: var(--shadow-glow);
    background: linear-gradient(135deg, var(--primary-dark), var(--secondary-color));
}

.stButton > button:active {
    transform: translateY(0) scale(0.98);
}

.stButton > button::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
    transition: left 0.5s;
}

.stButton > button:hover::before {
    left: 100%;
}

/* Movie Cards with Advanced Animations */
.movies-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 24px;
    padding: 40px 20px;
    animation: gridFadeIn 1s ease-out;
}

@keyframes gridFadeIn {
    from { opacity: 0; transform: translateY(40px); }
    to { opacity: 1; transform: translateY(0); }
}

.movie-card {
    background: var(--background-card);
    border-radius: 16px;
    padding: 0;
    overflow: hidden;
    position: relative;
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    border: 1px solid var(--border-color);
    animation: cardSlideIn 0.8s ease-out;
    animation-fill-mode: both;
}

.movie-card:nth-child(1) { animation-delay: 0.1s; }
.movie-card:nth-child(2) { animation-delay: 0.2s; }
.movie-card:nth-child(3) { animation-delay: 0.3s; }
.movie-card:nth-child(4) { animation-delay: 0.4s; }
.movie-card:nth-child(5) { animation-delay: 0.5s; }

@keyframes cardSlideIn {
    from { 
        opacity: 0; 
        transform: translateY(60px) scale(0.9); 
    }
    to { 
        opacity: 1; 
        transform: translateY(0) scale(1); 
    }
}

.movie-card:hover {
    transform: translateY(-12px) scale(1.03);
    box-shadow: 
        0 20px 40px rgba(0, 0, 0, 0.3),
        0 0 30px rgba(99, 102, 241, 0.2);
    border-color: var(--primary-color);
}

.movie-poster-container {
    position: relative;
    overflow: hidden;
}

.movie-poster {
    width: 100%;
    height: 320px;
    object-fit: cover;
    transition: transform 0.4s ease;
}

.movie-card:hover .movie-poster {
    transform: scale(1.1);
}

.movie-rank {
    position: absolute;
    top: 12px;
    left: 12px;
    background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
    color: white;
    width: 32px;
    height: 32px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    font-size: 0.9rem;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    animation: rankPulse 2s ease-in-out infinite;
}

@keyframes rankPulse {
    0%, 100% { transform: scale(1); }
    50% { transform: scale(1.1); }
}

.movie-info {
    padding: 20px;
}

.movie-title {
    font-size: 1.1rem;
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: 8px;
    line-height: 1.4;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
}

.movie-meta {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 0.9rem;
    color: var(--text-secondary);
}

.movie-rating {
    color: var(--accent-color);
    font-weight: 600;
    display: flex;
    align-items: center;
    gap: 4px;
}

.movie-rating::before {
    content: '⭐';
    animation: starTwinkle 2s ease-in-out infinite;
}

@keyframes starTwinkle {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.7; }
}

/* Loading Animation */
.loading-container {
    text-align: center;
    padding: 60px 20px;
}

.loading-spinner {
    width: 60px;
    height: 60px;
    border: 3px solid var(--border-color);
    border-top: 3px solid var(--primary-color);
    border-radius: 50%;
    animation: spin 1s linear infinite;
    margin: 0 auto 20px;
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

.loading-text {
    color: var(--text-secondary);
    font-size: 1.1rem;
    animation: loadingDots 1.5s ease-in-out infinite;
}

@keyframes loadingDots {
    0%, 20% { opacity: 0.2; }
    50% { opacity: 1; }
    100% { opacity: 0.2; }
}

/* Success Message */
.success-message {
    background: linear-gradient(135deg, var(--success-color), #059669);
    color: white;
    padding: 16px 24px;
    border-radius: 12px;
    text-align: center;
    font-weight: 600;
    margin: 20px 0;
    animation: successSlide 0.8s ease-out;
    box-shadow: 0 8px 25px rgba(16, 185, 129, 0.3);
}

@keyframes successSlide {
    from { opacity: 0; transform: translateY(-20px); }
    to { opacity: 1; transform: translateY(0); }
}

/* Section Headers */
.section-header {
    text-align: center;
    margin: 60px 0 40px;
    animation: headerFade 1s ease-out;
}

.section-title {
    font-size: 2.5rem;
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 12px;
    background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

@keyframes headerFade {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}

/* Sidebar Enhancements */
.stSelectbox > div > div {
    background: var(--background-card);
    border: 1px solid var(--border-color);
    border-radius: 8px;
    transition: all 0.3s ease;
}

.stSelectbox > div > div:hover {
    border-color: var(--primary-color);
    box-shadow: 0 0 10px rgba(99, 102, 241, 0.2);
}

.stSlider > div > div > div {
    background: var(--primary-color);
}

.stNumberInput > div > div {
    background: var(--background-card);
    border: 1px solid var(--border-color);
    border-radius: 8px;
    transition: all 0.3s ease;
}

.stNumberInput > div > div:hover {
    border-color: var(--primary-color);
}

/* Responsive Design */
@media (max-width: 768px) {
    .main-title {
        font-size: 2.5rem;
    }
    
    .movies-grid {
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 16px;
        padding: 20px 10px;
    }
    
    .movie-poster {
        height: 260px;
    }
    
    .movie-info {
        padding: 16px;
    }
}

/* Scrollbar Styling */
::-webkit-scrollbar {
    width: 8px;
}

::-webkit-scrollbar-track {
    background: var(--background-dark);
}

::-webkit-scrollbar-thumb {
    background: var(--primary-color);
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: var(--primary-dark);
}
</style>
""", unsafe_allow_html=True)

def get_movie_poster(title: str) -> str:
    """Fetch movie poster from TMDB API by title (with year fallback)."""
    try:
        # Extract title and year
        if "(" in title and ")" in title:
            clean_title = title.split("(")[0].strip()
            year = title.split("(")[-1].split(")")[0].strip()
        else:
            clean_title, year = title, None

        url = "https://api.themoviedb.org/3/search/movie"
        params = {"api_key": TMDB_API_KEY, "query": clean_title}
        response = requests.get(url, params=params).json()

        if response.get("results"):
            poster_path = response["results"][0].get("poster_path")
            if poster_path:
                return f"https://image.tmdb.org/t/p/w500{poster_path}"

        # Try with year if available
        if year:
            params = {"api_key": TMDB_API_KEY, "query": clean_title, "year": year}
            response = requests.get(url, params=params).json()
            if response.get("results"):
                poster_path = response["results"][0].get("poster_path")
                if poster_path:
                    return f"https://image.tmdb.org/t/p/w500{poster_path}"

        # Fallback placeholder
        return f"https://via.placeholder.com/300x450/1a1a1a/6366f1?text={clean_title.replace(' ', '+')}"

    except Exception as e:
        st.error(f"Poster fetch error for '{title}': {e}")
        return "https://via.placeholder.com/300x450/1a1a1a/ff4b4b?text=Error"

def display_movie_card(title: str, rank: int) -> None:
    poster_url = get_movie_poster(title)
    
    # Extract year
    year = "N/A"
    if "(" in title and ")" in title:
        try:
            year = title.split("(")[-1].split(")")[0]
        except:
            pass
    
    # Generate random rating for demo
    rating = round(np.random.uniform(3.5, 4.8), 1)

    st.markdown(f"""
    <div class="movie-card">
        <div class="movie-poster-container">
            <img src="{poster_url}" class="movie-poster" alt="{title}" loading="lazy">
            <div class="movie-rank">{rank}</div>
        </div>
        <div class="movie-info">
            <div class="movie-title">{title}</div>
            <div class="movie-meta">
                <span>{year}</span>
                <div class="movie-rating">{rating}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def run_recommender(data_dir, method, user_id, k, rank, top_n, prec_k, threshold):
    try:
        ratings, movies = load_movielens_100k(data_dir)
        train_df, test_df = make_train_test(ratings, test_size_per_user=5)
        train_mat, uid_to_ix, iid_to_ix, ix_to_uid, ix_to_iid = build_ui_matrix(train_df)

        if user_id not in uid_to_ix:
            st.error(f"User ID {user_id} not found. Try IDs between 1–943.")
            return None, None

        uix = uid_to_ix[user_id]

        # Predict based on method
        if method == "usercf":
            pred = predict_user_based(train_mat, k=k)
        elif method == "itemcf":
            pred = predict_item_based(train_mat, k=k)
        elif method == "svd":
            pred = predict_svd(train_mat, rank=rank)

        # Get recommendations
        rec_indices = recommend_top_n(pred, train_mat, uix, top_n=top_n)
        movie_ids = [ix_to_iid[i] for i in rec_indices]
        recommended_movies = []
        for mid in movie_ids:
            row = movies[movies["movie_id"] == mid]
            if not row.empty:
                recommended_movies.append(row.iloc[0]["title"])

        # Calculate precision
        precision = precision_at_k(pred, train_mat, test_df, uid_to_ix, iid_to_ix, k=prec_k, threshold=threshold)

        return recommended_movies, precision
    except Exception as e:
        st.error(f"Error running recommender: {str(e)}")
        return None, None

def main():
    st.markdown("""
    <div class="main-header">
        <h1 class="main-title">Movie Recommendation System</h1>
        <p class="main-subtitle">Discover your next favorite movie with AI-powered recommendations</p>
    </div>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("### Configuration")
        
        st.markdown("#### Algorithm")
        method = st.selectbox(
            "Recommendation Method",
            ["usercf", "itemcf", "svd"],
            format_func=lambda x: {
                "usercf": "User-Based Collaborative Filtering",
                "itemcf": "Item-Based Collaborative Filtering", 
                "svd": "SVD Matrix Factorization"
            }[x]
        )
        
        st.markdown("#### User Settings")
        user_id = st.number_input("User ID", min_value=1, max_value=943, value=1, step=1)
        
        st.markdown("#### Model Parameters")
        k = st.slider("K (neighbors)", 5, 50, 20)
        rank = st.slider("SVD Rank", 5, 100, 20)
        
        st.markdown("#### Output Settings")
        top_n = st.slider("Top-N Recommendations", 5, 20, 10)
        prec_k = st.slider("Precision@K", 5, 20, 10)
        threshold = st.slider("Rating Threshold", 1.0, 5.0, 3.5, 0.5)

    # Main content with enhanced button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Discover Movies", use_container_width=True):
            # Enhanced loading animation
            loading_placeholder = st.empty()
            with loading_placeholder:
                st.markdown("""
                <div class="loading-container">
                    <div class="loading-spinner"></div>
                    <div class="loading-text">Finding your perfect movies...</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Simulate processing time for better UX
            time.sleep(3)
            
            recommended_movies, precision = run_recommender(
                "./ml-100k", method, user_id, k, rank, top_n, prec_k, threshold
            )
            
            loading_placeholder.empty()
            
            if recommended_movies:
                # Success message with animation
                st.markdown(f"""
                <div class="success-message">
                    Precision@{prec_k}: {precision:.4f} | Found {len(recommended_movies)} perfect matches!
                </div>
                """, unsafe_allow_html=True)
                
                # Section header
                st.markdown("""
                <div class="section-header">
                    <h2 class="section-title">Your Personal Recommendations</h2>
                </div>
                """, unsafe_allow_html=True)
                
                # Movies grid with enhanced cards
                st.markdown('<div class="movies-grid">', unsafe_allow_html=True)
                cols = st.columns(5)
                for i, title in enumerate(recommended_movies, 1):
                    with cols[(i-1) % 5]:
                        display_movie_card(title, i)
                st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":

    main()
