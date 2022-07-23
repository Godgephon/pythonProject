import plotly.express as px

fig = px.line(x=['a', 'b', 'c'], y=[1, 3, 2], title='sample figure')
fig.show()