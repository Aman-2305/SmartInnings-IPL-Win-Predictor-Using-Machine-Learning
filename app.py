from flask import Flask, render_template, request, session
import numpy as np
import joblib

app = Flask(__name__)
app.secret_key = 'supersecretkey'

model = joblib.load('models/model.pkl')
encoders = joblib.load('models/encoders.pkl')

team_mapping = {
    'Delhi Daredevils': 'Delhi Capitals',
    'Gujarat Lions': 'Gujarat Titans',
    'Kings XI Punjab': 'Punjab Kings',
    'Rising Pune Supergiant': 'Rising Pune Supergiants',
}

exclude_teams = ['Kochi Tuskers Kerala', 'Pune Warriors', 'Deccan Chargers']

all_teams = encoders['Bat First'].classes_.tolist()
normalized_teams = set()
for team in all_teams:
    if team in exclude_teams:
        continue
    normalized_name = team_mapping.get(team, team)
    normalized_teams.add(normalized_name)
teams = sorted(normalized_teams)

venues = sorted(encoders['Venue'].classes_.tolist())

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        team1 = request.form['team1']
        team2 = request.form['team2']
        venue = request.form['venue']
        target = int(request.form['target_score'])
        score = int(request.form['current_score'])
        overs = float(request.form['overs_completed'])
        wickets = int(request.form['wickets_down'])

        team1 = team_mapping.get(team1, team1)
        team2 = team_mapping.get(team2, team2)

        team1_encoded = encoders['Bat First'].transform([team1])[0]
        team2_encoded = encoders['Bat Second'].transform([team2])[0]
        venue_encoded = encoders['Venue'].transform([venue])[0]

        runs_left = target - score
        balls_left = 120 - int(overs * 6)
        wickets_left = 10 - wickets
        crr = score / overs if overs > 0 else 0
        rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0

        input_data = np.array([[team1_encoded, team2_encoded, venue_encoded,
                                runs_left, balls_left, wickets_left,
                                target, crr, rrr]])

        result = model.predict_proba(input_data)[0]
        win_percent_team1 = round(result[1] * 100, 2)
        win_percent_team2 = round(result[0] * 100, 2)

        prediction = {
            'team1': team1,
            'team2': team2,
            'team1_win_prob': win_percent_team1,
            'team2_win_prob': win_percent_team2
        }

        past = session.get('past_predictions', [])
        past = [prediction] + past[:4]
        session['past_predictions'] = past

        return render_template('index.html', prediction=prediction, teams=teams, venues=venues, past_predictions=past, team_mapping=team_mapping)

    return render_template('index.html', teams=teams, venues=venues, team_mapping=team_mapping)

if __name__ == '__main__':
    app.run(debug=True)

