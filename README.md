
# 🏏 SmartInnings – IPL Win Predictor Using Machine Learning

SmartInnings is an interactive, AI-powered web application that predicts the outcome of IPL matches in real time. It allows users to input live match conditions — such as the score, overs completed, and wickets lost — and returns win probabilities for both teams using a machine learning model trained on 15+ years of IPL match data.

> ⚡️ Built with: Python · XGBoost · Flask · HTML/CSS/JavaScript

---

## 🎯 Features

✅ Predicts the probability of either team winning during the **second innings** of an IPL match  
✅ Uses live match inputs: batting team, bowling team, venue, current score, target, overs, wickets  
✅ Built on an **XGBoost classifier** trained on IPL ball-by-ball data from **2008–2023**  
✅ Clean, animated web interface using **Flask + JS**  
✅ Tracks the **last 5 predictions** to compare match scenarios  
✅ Designed for students, fans, fantasy players, and analysts alike

---

## 🧠 How It Works

### Inputs:
- Batting Team
- Bowling Team
- Venue
- Target Score
- Current Score
- Overs Completed
- Wickets Fallen

### Engineered Features:
- Runs Left
- Balls Left
- Wickets Remaining
- Required Run Rate (RRR)
- Current Run Rate (CRR)
- Encoded Teams & Venue

### Model:
- Algorithm: `XGBoostClassifier`
- Accuracy: ~79%
- Output: Win probability (%) for the batting team

---



