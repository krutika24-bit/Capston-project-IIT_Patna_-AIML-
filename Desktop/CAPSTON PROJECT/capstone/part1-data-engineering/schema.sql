-- ============================================================
-- Schema for reviews table (DuckDB)
-- Source: Women's Clothing E-Commerce Reviews
-- ============================================================

-- Column descriptions and types

/*
  clothing_id: INTEGER  -- Unique identifier for the clothing item
  age: INTEGER  -- Age of the reviewer (clipped to 18-100)
  title: VARCHAR  -- Title of the review (may be null)
  review_text: VARCHAR  -- Text content of the review
  rating: INTEGER  -- Rating given by the reviewer (1 to 5)
  recommended_ind: INTEGER  -- Whether the reviewer recommends the product (0/1)
  positive_feedback_count: INTEGER  -- Number of positive feedback votes on the review
  division_name: VARCHAR  -- Product division (General, General Petite, Initmates)
  department_name: VARCHAR  -- Product department (Tops, Dresses, Bottoms, etc.)
  class_name: VARCHAR  -- Product class (Dresses, Knits, Blouses, etc.)
  review_length: INTEGER  -- Derived: word count of review_text
  age_bucket: VARCHAR  -- Derived: age category bucket (<25, 25-34, 35-44, 45-54, 55+)
*/

-- DuckDB CREATE TABLE statement
CREATE TABLE IF NOT EXISTS reviews (
    clothing_id INTEGER,
    age INTEGER,
    title VARCHAR,
    review_text VARCHAR,
    rating INTEGER,
    recommended_ind INTEGER,
    positive_feedback_count INTEGER,
    division_name VARCHAR,
    department_name VARCHAR,
    class_name VARCHAR,
    review_length INTEGER,
    age_bucket VARCHAR
);