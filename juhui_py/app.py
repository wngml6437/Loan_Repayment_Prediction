from flask import Flask, render_template, url_for, request, redirect, flash
import find
app = Flask(__name__)

@app.route('/')
def index() :
    return render_template('index.html')

@app.route('/search/')
def search() :
    return render_template('search.html')

@app.route('/result/', methods = ["POST"])
def result() :
    error = None
    try:
        customer_id = request.form['customer_id']
        result = find.find_result(int(customer_id))

        if result == 1 :
            return render_template('result1.html')
        elif result == 0 :
            return render_template('result0.html')
    except:
        error = 'Please Input Valid Value..!'
        return render_template('search.html', error=error)


if __name__ == '__main__' :
    app.run(debug=True)
