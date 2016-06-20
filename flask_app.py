from flask import Flask, request, redirect, url_for, render_template
import numpy as np
import scipy.io as sio
import random
import time
from numpy import *
import numpy as np
from scipy import optimize
import copy

app = Flask(__name__)
app.config["DEBUG"] = True
app.secret_key = 'This is really unique and secret'

## FUNCTIONS ##

def normalize_ratings(ratings, did_rate):
    num_movies = ratings.shape[0]
    
    ratings_mean = zeros(shape = (num_movies, 1))
    ratings_norm = zeros(shape = ratings.shape)
    
    for i in range(num_movies): 
        # Get all the indexes where there is a 1
        idx = where(did_rate[i] == 1)
        #  Calculate mean rating of ith movie only from user's that gave a rating
        ratings_mean[i] = mean(ratings[i, idx])
        ratings_norm[i, idx] = ratings[i, idx] - ratings_mean[i]
    
    return ratings_norm, ratings_mean

def unroll_params(X_and_theta, num_users, num_movies, num_features):
	# Retrieve the X and theta matrixes from X_and_theta, based on their dimensions (num_features, num_movies, num_movies)
	# --------------------------------------------------------------------------------------------------------------
	# Get the first 30 (10 * 3) rows in the 48 X 1 column vector
	first_30 = X_and_theta[:num_movies * num_features]
	# Reshape this column vector into a 10 X 3 matrix
	X = first_30.reshape((num_features, num_movies)).transpose()
	# Get the rest of the 18 the numbers, after the first 30
	last_18 = X_and_theta[num_movies * num_features:]
	# Reshape this column vector into a 6 X 3 matrix
	theta = last_18.reshape(num_features, num_users ).transpose()
	return X, theta

def calculate_gradient(X_and_theta, ratings, did_rate, num_users, num_movies, num_features, reg_param):
	X, theta = unroll_params(X_and_theta, num_users, num_movies, num_features)
	# we multiply by did_rate because we only want to consider observations for which a rating was given
	difference = X.dot( theta.T ) * did_rate - ratings
	X_grad = difference.dot( theta ) + reg_param * X
	theta_grad = difference.T.dot( X ) + reg_param * theta
	# wrap the gradients back into a column vector 
	return r_[X_grad.T.flatten(), theta_grad.T.flatten()]

def calculate_cost(X_and_theta, ratings, did_rate, num_users, num_movies, num_features, reg_param):
	X, theta = unroll_params(X_and_theta, num_users, num_movies, num_features)
	# we multiply (element-wise) by did_rate because we only want to consider observations for which a rating was given
	cost = sum( (X.dot( theta.T ) * did_rate - ratings) ** 2 ) / 2
	regularization = (reg_param / 2) * (sum( theta**2 ) + sum(X**2))
	return cost + regularization

def recommendation(user_rat):
    dataFile = sio.loadmat("course_ratings.mat")
    original_ratings = dataFile['course_ratings']
    #Let's make some ratings.
    ratings = copy.copy(original_ratings)
    num_movies,num_users = ratings.shape
    users_ratings = np.array(user_rat)[:,np.newaxis]
    users_ratings=np.zeros(users_ratings.shape)
    for i in range(len(user_rat)):
        users_ratings[i]=user_rat[i]
    
    ratings = append(users_ratings, ratings, axis = 1)
    did_rate = (ratings != 0) * 1
    #Normalize ratings
    ratings, ratings_mean = normalize_ratings(ratings, did_rate)
    num_users = ratings.shape[1]
    num_features = 10
    movie_features = random.randn( num_movies, num_features )
    user_prefs = random.randn( num_users, num_features )
    initial_X_and_theta = r_[movie_features.T.flatten(), user_prefs.T.flatten()]
    reg_param = 2
    minimized_cost_and_optimal_params = optimize.fmin_cg(
        calculate_cost,
        fprime=calculate_gradient,
        x0=initial_X_and_theta, 
        args=(ratings, did_rate, num_users, num_movies, num_features, reg_param),
        maxiter=100,
        disp=False,
        full_output=True ) 
    
    cost, optimal_movie_features_and_user_prefs = minimized_cost_and_optimal_params[1], minimized_cost_and_optimal_params[0]
    # unroll once again
    movie_features, user_prefs = unroll_params(optimal_movie_features_and_user_prefs, num_users, num_movies, num_features)

    # Make some predictions (movie recommendations). Dot product
    all_predictions = movie_features.dot( user_prefs.T )

    # add back the ratings_mean column vector to my (our) predictions
    predictions_for_user = all_predictions[:, 0:1] + ratings_mean

    #print "predictions:",np.round(predictions_for_user,1)
    #print "cost",cost
    return predictions_for_user
##

course_ids = open("/home/daviddemeij/mysite/course_ids.txt", 'r')
courses = []
for course in course_ids:
    courses.append(str(course).rstrip('\r\n'))

comments = []

@app.route("/", methods = ['POST', 'GET'])
def index():
    return render_template("main_page.html", course_ids=courses, methods=['POST'])

@app.route('/recommend', methods=['POST', 'GET'])
def recommend():
    user_ratings = []
    for course in courses:
        user_ratings.append(request.form[course])
    predictions = recommendation(user_ratings)
    predictions = np.round(predictions,1).tolist()
    
    return render_template("recommend.html", ratings = zip(predictions,courses))

