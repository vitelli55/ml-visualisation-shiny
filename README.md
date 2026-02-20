# Machine Learning Algorithms Visualization (Shiny for Python)
Interactive Shiny app that demonstrates how machine learning algorithms work by visualizing them.

Live app: https://019c7b40-716d-cef8-3dcb-eae324145e63.share.connect.posit.cloud/

## About

This project focuses on building intuition for machine learning by showing how models update in real time through interactive visualizations. I build this project as my submission for the Maynooth Data Science Society 2025/2026 Shiny competition.

All algorithms are implemented from scratch. This means no ML libraries were used for the models. However, I only used scikit-learn for `make_blobs` to generate isotropic Gaussian blobs for K-Means clustering visualization. The models themselves were made with just *numpy*.

## Current Features
* Interactive sliders and inputs
* Step-by-step training visualization
* Randomise data

## Future Features
* Import data from csv file
* Visually display the error for Linear Regression
* Ability to hover over points for more information in KMeans & Linear Regression
* Add Decision Trees
* Add explanations for each model
* Improve UI

## Purpose

My goal with this project is to:
* Strengthen my foundational ML understanding
* Build algorithms from first principles
* Hopefully help others to visualise how these algorithms work.

## References

This project was developed as a learning exercise.  
Models and mathematical explanations were studied from:
- *Data Science from Scratch* by Joel Grus  
- YouTube videos on machine learning in Python, such as *NeuralNine* and *Andy McDonald*
