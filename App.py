from flask import Flask, render_template, redirect, request
import ImageCaptions


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

@app.route("/", methods = ["POST"])
def submit_data():
    if request.method == "POST":
        f = request.files["userfile"]
        path = "./static/{}".format(f.filename) + f.filename
        f.save(path)
        caption = ImageCaptions.caption_image(f)

        res_dic = {
            "image": path,
            "caption": caption
        }

    return render_template("index.html", result_dic = res_dic)


if __name__ == "__main__":
    app.run(debug=True)
