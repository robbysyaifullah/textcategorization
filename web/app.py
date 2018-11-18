from flask import Flask, render_template, request
from aduan import cek_aduan, deteksi_pos_tag

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == "POST":
        text = request.form['aduan']
        aduan = []
        aduan.append(text)
        is_aduan = cek_aduan(text)
        aduan.append('ya' if is_aduan else 'bukan')
        aduans = []
        aduans.append(aduan)
        if is_aduan:
            nouns, verbs, angkas = deteksi_pos_tag(text)
            # make the three array of same length
            arrays = ( nouns, verbs, angkas )
            max_arr_length = max(len(nouns), len(verbs), len(angkas))
            for arr in arrays:
                for i in range(len(arr), max_arr_length):
                    arr.append(" ")
            postags = []
            for i in range(0, max_arr_length):
                postags.append((nouns[i], verbs[i], angkas[i]))
            return render_template(
                'index.html', 
                aduans = aduans,
                postags = postags
            )
        else:
            return render_template('index.html', aduans = aduans)
    return render_template('index.html')