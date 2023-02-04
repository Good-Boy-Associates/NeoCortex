import os

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename


app = Flask(__name__, static_folder="static", template_folder="templates")
app.config['UPLOAD_FOLDER'] = "./static/img"


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == 'POST':
      f = request.files['file']
      filename = secure_filename(f.filename)
      print(filename)

      save_path = (os.path.join(app.config['UPLOAD_FOLDER'], filename))
      f.save(save_path)
      return render_template("index.html", upload_img=filename)
    return render_template("index.html")

app.run(debug=True)
