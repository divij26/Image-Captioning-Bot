from flask import Flask, render_template, redirect

app = Flask(__name__)

friends = ["I", "Me", "Myself"]

num = 5


@app.route('/')
def Hello():
    return render_template("index.html", friends=friends)


@app.route('/about')
def about():
    return "<h1> About Page </h1>"


@app.route('/home')
def home():
    return redirect('/')


if __name__ == "__main__":
    app.run(debug=True)
