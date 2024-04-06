import pandas as pd
import configparser as config
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import tweepy

cfg = config.ConfigParser()
cfg.read('config.ini')

#reading config.ini
api_key = cfg['twitter']['api_key']
api_key_secret = cfg['twitter']['api_key_secret']
access_token = cfg['twitter']['access_token']
access_token_secret = cfg['twitter']['access_token_secret']

#authentication
auth = tweepy.OAuthHandler(api_key,api_key_secret)
auth.set_access_token(access_token,access_token_secret)

api = tweepy.API(auth)

keywords = '#WearAMask -filter:retweets'
limit = 200

tweets = tweepy.Cursor(api.search_tweets, q=keywords, lang='en',
    count=100, tweet_mode='extended').items(limit)

#creating dataframe
columns = ['User','Tweet','Entities']
data = []
for tweet in tweets:
    data.append([tweet.user.screen_name,tweet.full_text,tweet.entities])
df = pd.DataFrame(data,columns=columns)

#exporting as a json file
result = df.to_json('.\TwitterData.json')

hashtagslist = []
for dict in df['Entities']:
    for ht in dict['hashtags']:
        hashtagslist.append(ht['text'])

for i in range(len(hashtagslist)):
    hashtagslist[i] = hashtagslist[i].lower()

unique_hashtags = list(set(hashtagslist))

tweets_dictionary = {}
iterative = 0
for tweet in df['Tweet']:
    iterative += 1
    for uht in unique_hashtags:
        if '#'+(uht) in tweet:
            if str(iterative) in tweets_dictionary.keys():
                tweets_dictionary[str(iterative)].append(uht)
            else:
                tweets_dictionary[str(iterative)] = [uht]

df1 = pd.DataFrame(columns = unique_hashtags, index = unique_hashtags)
df1[:] = int(0)
for value in tweets_dictionary.values():
    for uht1 in unique_hashtags:
        for uht2 in unique_hashtags:
            if uht1 in value and uht2 in value:
                df1[uht1][uht2] += 1
                df1[uht2][uht1] += 1

edge_list = []
for index, row in df1.iterrows():
    i=0
    for col in row:
        weight = float(col)/400
        edge_list.append((index,df1.columns[i],weight))
        i+=1

updated_edge_list = [x for x in edge_list if not x[2] == 0.0]

node_list = []
for i in unique_hashtags:
    for e in updated_edge_list:
        if i == e[0] and i == e[1]:
           node_list.append((i, e[2]*5))
for i in node_list:
    if i[1] == 0.0:
        node_list.remove(i)
# for i in node_list:
#     if i[1] < 5.0:
#         node_list.remove(i)

for i in updated_edge_list:
    if i[0] == i[1]:
        updated_edge_list.remove(i)

plt.subplots(figsize=(10,10))

G = nx.Graph()
#G = nx.barabasi_albert_graph(1000,2,20532)
for i in sorted(node_list):
    G.add_node(i[0],size=i[1])
G.add_weighted_edges_from(updated_edge_list)

node_order = nx.nodes(G)

updated_node_order = []
for i in node_order:
    for x in node_list:
        if x[0] == i:
            updated_node_order.append(x)

test = nx.get_edge_attributes(G,'weight')
updated_edges_2 = []
for i in nx.edges(G):
    for x in test:
        if i[0] == x[0] and i[1] == x[1]:
            updated_edges_2.append(test[x])

node_scalar = 400
edge_scalar = 10

sizes = [x[1]*node_scalar for x in updated_node_order]
widths = [x*edge_scalar for x in updated_edges_2]

pos = nx.spring_layout(G, k=0.5, iterations = 100, seed=1)
#pos = nx.get_node_attributes(G, "pos")

nx.draw(G, pos, with_labels=False, font_size = 6,
    node_size=sizes, width = widths)

degree_sequence = sorted((d for n, d in G.degree()),reverse=True)
dmax = max(degree_sequence)

fig = plt.figure("Degree of word co-occurrence network", figsize=(8, 8))
axgrid = fig.add_gridspec(5,4)

ax0 = fig.add_subplot(axgrid[0:3, :])
Gcc = G.subgraph(sorted(nx.connected_components(G), key=len, reverse=True)[0])
pos = nx.spring_layout(Gcc, seed=10396953)
nx.draw_networkx_nodes(Gcc, pos, ax=ax0, node_size=20)
nx.draw_networkx_edges(Gcc, pos, ax=ax0, alpha=0.4)
ax0.set_title("Connected components of G")
ax0.set_axis_off()

ax1 = fig.add_subplot(axgrid[3:, :2])
ax1.plot(degree_sequence, "b-", marker="o")
ax1.set_title("Degree Rank Plot")
ax1.set_ylabel("Degree")
ax1.set_xlabel("Rank")

ax2 = fig.add_subplot(axgrid[3:, 2:])
ax2.bar(*np.unique(degree_sequence, return_counts=True))
ax2.set_title("Degree histogram")
ax2.set_xlabel("Degree")
ax2.set_ylabel("# of Nodes")

fig.tight_layout()
plt.show()

dia = nx.diameter(Gcc)
print("The diameter of the connected graph is: ",dia)

dc = nx.degree_centrality(G)
print(list(dc.keys())[list(dc.values()).index(max(dc.values()))],max(dc.values()))

bc = nx.betweenness_centrality(G)
print(list(bc.keys())[list(bc.values()).index(max(bc.values()))],max(bc.values()))