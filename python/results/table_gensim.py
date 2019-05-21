import plotly.plotly as py
import plotly.graph_objs as go



trace = go.Table(
    header=dict(values=['A Scores', 'B Scores'],
                line = dict(color='#7D7F80'),
                fill = dict(color='#a1c3d1'),
                align = ['left'] * 5),
    cells=dict(values=[[100, 90, 80, 90],
                       [95, 85, 75, 95]],
               line = dict(color='#7D7F80'),
               fill = dict(color='#EDFAFF'),
               align = ['left'] * 5))

layout = dict(width=500, height=300)
data = [trace]
fig = dict(data=data, layout=layout)
py.iplot(fig, filename = 'styled_table')





data :





Gensim

conv+lstm

300

* (10 epocas de w2v) = 0.478   (Estable por 10 epocas de la red)

* (50 epocas de w2v) = 

* (100 epocas de w2v) = 

* (300 epocas de w2v) = 




Gensim

lstm

300

* (10 epocas de w2v) = 

* (50 epocas de w2v) = 

* (100 epocas de w2v) = 

* (300 epocas de w2v) = 

