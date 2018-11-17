from flask import Flask, render_template, request

app = Flask(__name__)

def CekAduan(text):
    return True

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == "POST":
        text = request.form['aduan']
        aduan = []
        aduan.append(text)
        aduan.append('ya' if CekAduan(text) else 'bukan')
        aduans = []
        aduans.append(aduan)
        return render_template('index.html', aduans = aduans)
    return render_template('index.html')