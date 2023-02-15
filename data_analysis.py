from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# Route for the homepage


@app.route('/')
def index():
    return render_template('index.html')

# Route for the data analysis


@app.route('/analyze', methods=['POST'])
def analyze():
    # Get the uploaded file from the form
    file = request.files['file']

    # Read the file into a Pandas DataFrame
    df = pd.read_csv(file)

    # Split the data into training and testing sets
    X = df.drop('target', axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Train a linear regression model on the training set
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Test the model on the testing set
    score = model.score(X_test, y_test)

    # Return the score to the results page
    return str(score)


if __name__ == '__main__':
    app.run(debug=True)
