-- 1. KPI Summary
SELECT COUNT(*) AS total_titles,
       SUM(CASE WHEN type = 'Movie' THEN 1 ELSE 0 END) AS total_movies,
       SUM(CASE WHEN type = 'TV Show' THEN 1 ELSE 0 END) AS total_shows
FROM netflix_titles;

-- 2. Genre Rankings
SELECT primary_genre, COUNT(*) AS count
FROM netflix_titles
GROUP BY primary_genre
ORDER BY count DESC LIMIT 15;

-- 3. Country Rankings
SELECT country, COUNT(*) AS count
FROM netflix_titles WHERE country NOT IN ('Unknown','')
GROUP BY country ORDER BY count DESC LIMIT 15;

-- 4. Year-wise Trend
SELECT added_year, type, COUNT(*) AS titles_added
FROM netflix_titles WHERE added_year IS NOT NULL
GROUP BY added_year, type ORDER BY added_year;

-- 5. Ratings Distribution
SELECT rating, audience_segment, COUNT(*) AS count
FROM netflix_titles GROUP BY rating, audience_segment
ORDER BY count DESC;