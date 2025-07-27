# Rat-A-Turnip Price Predictor

![](https://github.com/mizhare/turnip-weekly-predictor/blob/main/static/prediction-gif.gif)

### Overview

This is a turnip price predictor project inspired by the game _Animal Crossing: New Horizons_ from Nintendo. I found it to be a rich and rewarding full-stack project, that would help me solidify some of the Machine Learning knowledge I'm currently studying. On the back-end, I used **Flask** to handle the prediction logic and serve the app, while the front-end was built using **HTML** and **CSS** for layout and styling.

It was especially enjoyable to integrate all the components from model training and serialization with **joblib**, to input handling, user interface design, and displaying forecast results in a playful, themed way with some touches from my personal branding (🐭). For deployment, I used **Render** to bring the application online, making it accessible as a simple and responsive web app.

Even though the project is still a work in progress, the hands-on experience of bringing machine learning into a web context and taking it all the way to deployment has been both challenging and exciting experience.
### How to use

Access this [link](https://turnip-predictor.onrender.com) and enter the turnip prices for your current week (ideally at least 4 prices) to improve the accuracy of the predicted pattern.  If you know the previous week's pattern, you can select it using the button above. (Keep in mind that it can influence the prediction for the current week.)


### Machine learning

**Classifier Model**  
I chose a **Random Forest Classifier** to predict the weekly price patterns because it handles small-to-medium datasets well, captures non-linear relationships, and is relatively easy to interpret. It performed consistently across different types of patterns, such as _small spike_, _large spike_, _fluctuating_, and _decreasing_.

**Regression Model**  
For forecasting specific future prices (e.g., Friday and Saturday), I trained multiple **Gaussian Process Regressors**, one for each time period. I picked GPR due to its ability to return not just predictions but also confidence intervals, which adds more context and trust to the results.


### Challenges

One of the main challenges was dealing with **incomplete user input**. Since the prediction model requires at least a few price points to identify the correct pattern, I had to design a way to handle missing values without breaking the model. I solved this by using **imputation techniques** during preprocessing and making the interface flexible enough to accept partial inputs.

Another difficulty was related to the **similarity between different price patterns**, especially early in the week. Several patterns in the game like **fluctuating** and **small_spike** can begin with similar price drops or rises, making it hard for the model to distinguish them based on limited data. To address this, I manually annotated several weeks of historical price data and trained a **Random Forest classifier** to learn subtle differences. The goal was to teach the model how to "see ahead" using early cues, even when the patterns look alike at first.
