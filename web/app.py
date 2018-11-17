from flask import Flask, render_template, request
from aduan import cek_aduan

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == "POST":
        text = request.form['aduan']
        aduan = []
        aduan.append(text)
        aduan.append('ya' if cek_aduan(text) else 'bukan')
        aduans = []
        aduans.append(aduan)
        return render_template('index.html', aduans = aduans)
    return render_template('index.html')