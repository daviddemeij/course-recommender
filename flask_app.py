from flask import Flask, request, redirect, url_for, render_template
import numpy as np
import scipy.io as sio

import random
def learn():
    data = sio.loadmat("/home/daviddemeij/mysite/course_ratings.mat")
    ratings = data['course_ratings'].T
    num_users = 26
    num_features = 7
    num_courses = 22
    Lambda = 1
    R = ratings>=1
    X = np.random.rand(num_courses,num_features) - 0.5
    Theta = np.random.rand(num_users,num_features) - 0.5
    def normalizeRatings(Y,R):
        avgRatingPerCourse = np.sum(Y, 0) / (np.sum(R, 0) + 0.0)
    return (Y - R * avgRatingPerCourse), avgRatingPerCourse

    def costFunction(X,Y,Theta, R, Lambda):
        cost = 0.5*np.sum((X.dot(Theta.T).T-Y)**2)
        cost = cost +  (Lambda/2)*(np.sum(Theta**2)+np.sum(X**2))
        return cost

    def gradientFunction(X, Y, Theta, R, Lambda):
        X_grad = (R*(X.dot(Theta.T).T-Y)).T.dot(Theta) + Lambda*X
        Theta_grad = (R*(X.dot(Theta.T).T-Y)).dot(X) + Lambda*Theta
        return X_grad, Theta_grad

    avgRatingPerCourse = []
    Y, avgRatingPerCourse = normalizeRatings(ratings,R)

    maxit=2000
    import time
    def gradDesc(X, Y, R, Theta, Lambda, eta=1e-3):
        cost = costFunction(X,Y,Theta, R, Lambda)            # Compute the error function
        errs = np.zeros(maxit)       # Keep track of the error function
        ts = np.zeros(maxit)         # Keep track of time stamps
        start = time.time()
        X_best = X
        Theta_best=Theta
        bestCost = 100000

        newCost = cost + 1.
        for n in range(maxit):
            X_grad, Theta_grad = gradientFunction(X,Y,Theta, R, Lambda)

            #if (n % 2 == 0):
            X -= X_grad * eta     # Update the weights
            #else:
            Theta -= Theta_grad * eta

            #print " X Gradient:", X_grad
            #print " Theta Gradient", Theta_grad
            pastCost = newCost
            newCost = costFunction(X,Y,Theta, R, Lambda)
            print ("##", n, "err:", newCost)

            if newCost < bestCost:
                bestCost = newCost
                X_best = X
                Theta_best = Theta
            #if pastCost-newCost < 1e-5:                # If we couldn't decrease the error anymore,
            #    return X, Theta            # just give up

            errs[n] = newCost                           # Keep track of how the errors evolved
            ts[n] = time.time()-start

        return X_best, Theta_best
    X_new,Theta_new = gradDesc(X,Y,R,Theta,Lambda)

    p = X_new.dot(Theta_new.T)
    print (avgRatingPerCourse)
    print (np.round(p[:,3] + avgRatingPerCourse,1))
    print (ratings[3])

app = Flask(__name__)
app.config["DEBUG"] = True
app.secret_key = 'This is really unique and secret'

##@app.route('/')
# def hello_person():
#     course_ids = open("/home/daviddemeij/mysite/course_ids.txt", 'r')
#     text = ""
#     for course_name in course_ids:
#         text = text+("<tr><td>"+str(course_name)+"</td>\
#         <td><input type=\"radio\" name=\""+str(course_name).rstrip('\r\n')+"\" value=\"0\" checked></td>\
#         <td><input type=\"radio\" name=\""+str(course_name).rstrip('\r\n')+"\" value=\"1\" ></td>\
#         <td><input type=\"radio\" name=\""+str(course_name).rstrip('\r\n')+"\" value=\"2\" ></td>\
#         <td><input type=\"radio\" name=\""+str(course_name).rstrip('\r\n')+"\" value=\"3\" ></td>\
#         <td><input type=\"radio\" name=\""+str(course_name).rstrip('\r\n')+"\" value=\"4\" ></td>\
#         <td><input type=\"radio\" name=\""+str(course_name).rstrip('\r\n')+"\" value=\"5\" ></td>")
#     return """<form action="%s" method="POST"><table border=1><tr>
#         <td>Course name</td><td>N/A</td><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td></tr>""" % (url_for('greet'),) +text+"""
#          </table>
#   <input type="submit" value="Go!" />
#     </form>


#         """
course_ids = open("/home/daviddemeij/mysite/course_ids.txt", 'r')
comments = []

@app.route("/", methods = ['POST', 'GET'])
def index():
    if request.method == "GET":
        return render_template("main_page.html", course_ids=course_ids)

    comments.append(request.form["contents"])
    return redirect(url_for('index'))

@app.route('/greet', methods=['POST'])
def greet():
    greeting = random.choice(["Hiya", "Hallo", "Hola", "Ola", "Salut", "Privet", "Konnichiwa", "Ni hao"])
    return """
        <p>%s, %s!</p>
        <p><a href="%s">Back to start</a></p>
        """ % (greeting, request.form["Internship"], url_for('hello_person'))

