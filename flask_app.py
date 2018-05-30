from flask import Flask, request, redirect, url_for, render_template
import numpy as np
import recommendation as recom

app = Flask(__name__)
app.config["DEBUG"] = True
app.secret_key = 'This is really unique and secret'

## FUNCTIONS ##




##

course_ids = open("/home/daviddemeij/mysite/course_ids.txt", 'r')
courses = []
for course in course_ids:
    courses.append(str(course).rstrip('\r\n'))

movie_ids = open("/home/daviddemeij/mysite/movie_ids3.txt", 'r')
movies = ["Toy Story (1995)", "GoldenEye (1995)", "Four Rooms (1995)", "Get Shorty (1995)", "Copycat (1995)", "Shanghai Triad (1995)", "Twelve Monkeys (1995)", "Babe (1995)", "Dead Man Walking (1995)",
"Richard III (1995)", "Seven (Se7en) (1995)", "Usual Suspects, The (1995)", "Mighty Aphrodite (1995)", "Postino, Il (1994)", "Mr. Holland's Opus (1995)", "French Twist (Gazon maudit) (1995)",
"From Dusk Till Dawn (1996)", "White Balloon, The (1995)", "Antonia's Line (1995)", "Angels and Insects (1995)",
"Muppet Treasure Island (1996)", "Braveheart (1995)", "Taxi Driver (1976)", "Rumble in the Bronx (1995)", "Birdcage, The (1996)",
"Brothers McMullen, The (1995)", "Bad Boys (1995)", "Apollo 13 (1995)", "Batman Forever (1995)", "Belle de jour (1967)", "Crimson Tide (1995)", "Crumb (1994)",
"Desperado (1995)", "Doom Generation, The (1995)", "Free Willy 2: The Adventure Home (1995)", "Mad Love (1995)", "Nadja (1994)", "Net, The (1995)", "Strange Days (1995)"]


@app.route("/", methods=['POST', 'GET'])
def index():
    return render_template("main_page.html")

@app.route("/hmi-courses", methods = ['POST', 'GET'])
def hmi():
    link = "/hmi-courses"
    if request.method == "GET":
        return render_template("get_ratings.html", items=courses, link=link, methods=['POST'])
    else:
        user_ratings = []
        for course in courses:
            user_ratings.append(request.form[course])
        predictions = recom.recommendation_courses(user_ratings)
        predictions = predictions.T.tolist()
        predictions = np.round(predictions[0],1).tolist()
        ratings_ordered = sorted(zip(predictions, courses), reverse=True)
        return render_template("recommend.html", link=link, ratings = ratings_ordered)

@app.route("/movies", methods = ['POST', 'GET'])
def mov():
    if request.method == "GET":
        return render_template("get_ratings.html", items=movies, methods=['POST'])
    else:
        user_ratings = []
        for movie in movies:
            user_ratings.append(request.form[movie])
        predictions = recom.recommendation_movies(user_ratings)
        predictions = predictions.T.tolist()
        predictions = np.round(predictions[0],1).tolist()
        ratings_ordered = sorted(zip(predictions, movies), reverse=True)
        return render_template("recommend.html", ratings = ratings_ordered)


@app.route("/gameployer", methods = ['POST', 'GET'])
def gameployer():
    return render_template('gameployer.html')

@app.route("/gameployer/create", methods = ['POST', 'GET'])
def create():
    return render_template('create.html')

@app.route("/gameployer/apply", methods = ['POST', 'GET'])
def apply():
    return render_template('game.html')

@app.route("/web", methods = ['POST', 'GET'])
def web():
    return render_template('web_app.html')