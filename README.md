# Movie-Recommendation-System

A personalized movie recommendation system built with **Streamlit** and **TMDB API**. This project uses collaborative filtering techniques (User-based, Item-based, and SVD) to suggest movies and provides movie details fetched from **The Movie Database (TMDB)**.

Live App: https://movie-recommendation-system-mjc8awkzejmjxsqxrschf9.streamlit.app/

---

## Features

- **Search & Recommend Movies** using collaborative filtering.
-  **Multiple Algorithms Supported**:  
  - User-based filtering  
  - Item-based filtering  
  - Singular Value Decomposition (SVD)
-  **Movie Details from TMDB API**: posters, ratings, release dates, etc.
-  **Evaluation Metrics**: Precision@K for recommendation performance.
-  **Streamlit Web App** for interactive use.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/movie-recommendation-system.git
```
2. Change directory
```bash
cd movie-recommendation-system
```
3.Install dependencies
```bash
pip install -r requirements.txt
```
4.Set your TMDB API key in config.py
```bash
TMDB_API_KEY = "your_api_key_here"
```
5.Run the App
```bash
streamlit run app.py
```


