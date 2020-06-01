# coding=utf-8
from __future__ import unicode_literals
from flask import Flask, render_template, request, redirect, url_for
import urllib
#import unirest #In an attempt to update the library, unirest requires Python 2.7
                #which is getting updated to Python 3 which means we can now use requests instead
import difflib
from goose3 import Goose
import io
import os
import sys, csv
import tensorflow as tf
import requests
import json
import re
#import importlib #Used to reload modules
#importlib.reload(sys)
#sys.setdefaultencoding('utf8')
csv.field_size_limit(sys.maxsize)

import defend
import nltk
#nltk.download('punkt')

application = Flask(__name__)

SAVED_MODEL_DIR = './static/saved_models'
EMBEDDINGS_PATH = 'saved_models/glove.6B.100d.txt'
platform = 'politifact'
SAVED_MODEL_FILENAME = platform + '_Defend_model.h5'

h = defend.Defend(platform)
h.load_weights(saved_model_dir=SAVED_MODEL_DIR, saved_model_filename=SAVED_MODEL_FILENAME)
h.activation_maps('abc', ['1', '2', '3'])

graph = tf.get_default_graph()

newsdic = {}

file_title = './static/' + platform + '_title_no_ignore.tsv'
with open(file_title) as tsvfile:
    reader = csv.reader(tsvfile, delimiter=str(u'\t'))
    for row in reader:
        id = row[0]
        newsdic[id] = {}
        newsdic[row[0]]['title'] = row[1]

file_comment_our = './static/' + platform + '_comment_no_ignore.tsv'
with open(file_comment_our) as tsvfile:
    reader = csv.reader(tsvfile, delimiter=str(u'\t'))
    for row in reader:
        if row[0] in newsdic.keys():
            newsdic[row[0]]['comment_our'] = row[1].split("::")

file_content = './static/' + platform + '_content_no_ignore.tsv'
with open(file_content) as tsvfile:
    reader = csv.reader(tsvfile, delimiter=str(u'\t'))
    for row in reader:
        if row[0] in newsdic.keys():
            newsdic[row[0]]['content'] = row[2]
            newsdic[row[0]]['label'] = row[1]


def url2str(_title):
    newtitle = _title
    if _title.startswith('twitter.com') or _title.startswith('https://twitter.com'):
        url = json.loads(requests.get("https://publish.twitter.com/oembed?url=" + _title).content)
        url = url['html']

        #remove special characters that will cause it to crash
        remove_emojis = re.compile("["
                                   u"\U0001F600-\U0001F64F"  # emoticons
                                   u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                   u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                   u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                   "]+", flags=re.UNICODE)
        url = remove_emojis.sub(r'', url)
        url = url.replace('*', ' ')

        newtitle = re.sub(r"<([^>]*)>", "", url)
    return newtitle.encode('ascii', 'ignore')[:50]


def check(_title):
    newtitle = url2str(_title)
    url = newtitle
    url = urllib.parse.quote_plus(url)
    response = requests.get(
        "https://api-hoaxy.p.rapidapi.com/articles?sort_by=relevant&use_lucene_syntax=true&query=" + url,
        headers={
            "X-RapidAPI-Key": "API-KEY"
        }
    ) #TODO: Replace API-KEY with a variable with file content with an api key
    a = response.json()
    articles = a['articles']
    canonical_url = a['articles'][0]['canonical_url']
    title = a['articles'][0]['title']
    score = 0
    for a in articles:
        tmpscore = difflib.SequenceMatcher(None, a['title'], newtitle).quick_ratio()
        if tmpscore > score:
            canonical_url = a['canonical_url']
            title = a['title']
            id = a['id']
            score = tmpscore
    g = Goose()
    article = g.extract(url=canonical_url)
    article = article.cleaned_text.split("\n")
    article = list(filter(None, article))

    label = 0
    found = 0

    for key in newsdic.keys():
        if difflib.SequenceMatcher(None, newsdic[key]['title'], title).quick_ratio() > 0.8 or difflib.SequenceMatcher(None, newsdic[key]['title'], _title).quick_ratio() > 0.8:
            if difflib.SequenceMatcher(None, newsdic[key]['title'], _title).quick_ratio() > 0.8:
                title = _title
            # comment_our = newsdic[key]['comment_our']
            # for i in range(len(comment_our)):
            #     c_val.append(comment_our[i].split('\t')[0])
            #     if i < 5:
            #         comment += "<td>User" + str(i + 1) + "</td><td>" + comment_our[i].split('\t')[
            #             0] + "</td><td class=\"text-nowrap\">" + str(
            #             round(float(comment_our[i].split('\t')[1].replace('\n', '')), 4)) + "</td></tr>"
            comment = ''
            sentence = ''
            article = "<p><strong>Source: </strong><a href=\"" + canonical_url + "\">" + canonical_url + "</a></p>" + '<p>' + newsdic[key]['content'] + '</p>'
            if newsdic[key]['label'] == "1":
                label = 1

            x_val = [nltk.tokenize.sent_tokenize(newsdic[key]['content'])]
            c_val = [newsdic[key]['comment_our']]
            found = 1
            break

    if found == 0:
        x_val = [article]
        c_val = [' ']
        article = "<p><strong>Source: </strong><a href=\"" + canonical_url + "\">" + canonical_url + "</a></p>" + "<br>".join(
            [a for a in article if a != ""])

        comment = "<td></td><td>The comments required to run the algorithm are not enough!</td><td class=\"text-nowrap\"></td></tr>"
        sentence = ''

    global graph
    with graph.as_default():
        global h
        res_comment_weight, res_sentence_weight = h.activation_maps(x_val, c_val)
    sorted_comment_weight = sorted(res_comment_weight[0], key=lambda tup: tup[1], reverse=True)

    for i in range(len(sorted_comment_weight)):
        if i < 5:
            comment += "<td>User" + str(i + 1) + "</td><td>" + " ".join(sorted_comment_weight[i][
                                                                            0]) + "</td><td class=\"text-nowrap\">" + str(
                round(sorted_comment_weight[i][1], 6)) + "</td></tr>"

    for pos, v in enumerate(x_val[0]):
        if pos < len(res_sentence_weight[0]):
            sentence += "<tr><td name=\"td2\">" + str(pos + 1) + "</td><td name=\"td0\">" + v + "</td>\
                                                          <td class=\"text-right\" name=\"td1\">\
                                                            <span class=\"badge badge-default\">" + str(
                round(res_sentence_weight[0][pos][1], 8)) + "</span>\
                                                          </td>\
                                                        </tr>"
    confidence = round(sum([a[1] for a in res_sentence_weight[0]]) * 100, 1)
    if confidence < 50:
        confidence = 100 - confidence
    elif confidence > 100:
        confidence = 95

    if label == 1:
        if confidence > 90:
            image = "<img src=\"/static/demo/brand/f.jpg\" class=\"header-brand-img\" alt=\"tabler logo\">"
        elif confidence > 70:
            image = "<img src=\"/static/demo/brand/mf.jpg\" class=\"header-brand-img\" alt=\"tabler logo\">"
        else:
            image = "<img src=\"/static/demo/brand/ht.jpg\" class=\"header-brand-img\" alt=\"tabler logo\">"
        label = "<div class=\"card-alert alert alert-danger mb-0\">\
                    Disputed by FOO with " + str(confidence) + "% confidence  \
                    <a href=\"#\" onClick=\"alert(\'We have received your feedback. Thank you!\')\" class=\"btn outline-info\"><i class=\"fe fe-flag\"></i> Report detection error</a>\
                    </div>"
    else:
        if confidence > 90:
            image = "<img src=\"/static/demo/brand/t.jpg\" class=\"header-brand-img\" alt=\"tabler logo\">"
        elif confidence > 70:
            image = "<img src=\"/static/demo/brand/mt.jpg\" class=\"header-brand-img\" alt=\"tabler logo\">"
        else:
            image = "<img src=\"/static/demo/brand/ht.jpg\" class=\"header-brand-img\" alt=\"tabler logo\">"
        label = "<div class=\"card-alert alert alert-success mb-0\">\
                    Verified by FOO with " + str(confidence) + "% confidence  \
                    <a href=\"#\" onClick=\"alert(\'We have received your feedback. Thank you!\')\" class=\"btn outline-info\"><i class=\"fe fe-flag\"></i> Report detection error</a>\
                    </div>"

    return url, image + article, title, comment, sentence, label

title = "Tom Price: “It’s Better For Our Budget If Cancer Patients Die More Quickly”"
url, article, title, comment, sentence, label = check(title)


@application.route('/')
def index():
    return render_template('index.html', url=url, newscontent=article, title=title, comment=comment, sentence=sentence, label=label)


@application.route('/<name>', methods=['GET'])
def link(name):
    url, article, title, comment, sentence, label = check(name)
    return render_template('index.html', url=url, newscontent=article, title=title, comment=comment, sentence=sentence, label=label)


@application.route('/demo', methods=['POST'])
def demo():
    newtitle = request.form['url']
    newtitle = url2str(newtitle)
    return redirect(url_for('link', name=newtitle, _anchor='exactlocation'))
    # url, article, title, comment, sentence, label = check(newtitle)
    # return render_template('index.html', url=url, newscontent=article, title=title, comment=comment, sentence=sentence, label=label)


@application.route('/index')
def foobar():
    return render_template('./docs/index.html')


if __name__ == '__main__':
    application.run(debug=False, threaded=False, host='0.0.0.0', port=80)
