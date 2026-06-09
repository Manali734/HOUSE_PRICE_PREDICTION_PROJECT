CREATE DATABASE house_price_db;

USE house_price_db;
CREATE TABLE predictions(

id INT AUTO_INCREMENT PRIMARY KEY,

area INT,

bedrooms INT,

bathrooms INT,

stories INT,

parking INT,

predicted_price DOUBLE

);
SELECT * FROM predictions;