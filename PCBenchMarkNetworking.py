import os
os.environ["MODIN_ENGINE"] = "dask"
os.environ["MODIN_CPUS"] = "32"

import modin.pandas as pd
import numpy as np
from collections import namedtuple
from dotenv import load_dotenv
import networkx as nx
import matplotlib.pyplot as plt
import mysql.connector
import csv
from distributed import Client


load_dotenv()

config_players = {
  'user': os.getenv('user_id'),
  'password': os.getenv('password_id'),
  'host': os.getenv('host_ip'),
  'database': os.getenv('DB_NAME_PLAYERS')
}


def execute_sql(sql, insert=False, param=None):
    conn = mysql.connector.connect(**config_players)
    mycursor = conn.cursor(buffered=True, dictionary=True)
    
    mycursor.execute(sql, param)
    
    if insert:
        conn.commit()
        mycursor.close()
        conn.close()
        return

    rows = mycursor.fetchall()

    with open("players.csv", "w") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([i[0] for i in mycursor.description]) # write headers
        Record = namedtuple('Record', rows[0].keys())
        csv_writer.writerows([Record(*r.values()) for r in rows])

    mycursor.close()
    conn.close()
    return 


def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]
        
        
def process_names():
    df = None
        
    sqlLocation = ("""
    SELECT
        pl.name,
        p.Prediction,
        p.Predicted_confidence,
        rp.region_id,
        rp.timestamp
    FROM Players pl
    JOIN Reports rp ON (rp.reportedID = pl.id)
    JOIN Predictions p ON (p.id = pl.id)
    LIMIT 1000000
    """)

    execute_sql(sqlLocation, insert=False, param=None)

    return


def main():
    print("Here we goo!!")
    df = pd.read_csv("players.csv")
    df['timestamp'] = pd.to_datetime(df['timestamp'], format="%Y-%m-%d %H:%M:%S").sort_values()
    print(df)

    df_color = df[['name','Prediction']].copy()
    df_color.drop_duplicates(inplace=True)
    mask = df_color['Prediction'] == 'Real_Player'
    df_color['color'] = np.where(mask, 'green','red') 
    df_color.drop(columns='Prediction',inplace=True)


    colmask = df_color['color'] == 'green'
    real = df_color[colmask]['color'].count()
    bot = df_color[~colmask]['color'].count()
    print("Real_Player:", real)
    print("Bot:", bot)
    print("Percent: ", (round((bot)/(real+bot),5)*100),"%")

    df_transformed = df.copy()
    del df
    mask = df_transformed['Prediction'] == 'Real_Player'

    df_transformed['Predicted_confidence'] = np.where(mask, df_transformed['Predicted_confidence'], df_transformed['Predicted_confidence'] * -1)
    df_transformed.drop(columns=['Prediction'], inplace=True)

    df_clean = pd.merge(df_transformed, df_transformed, left_on=['region_id','timestamp'], right_on=['region_id','timestamp'])
    df_clean.drop(columns=['region_id','timestamp'], inplace=True)

    mask = df_clean['name_x'] == df_clean['name_y']
    df_clean = df_clean[~mask]
    print(df_clean)
    df_clean.to_csv("df_clean.csv")

    df_process = df_clean.groupby(by=['name_x','name_y']).sum().reset_index()
    df_process.drop(columns=['name_y','Predicted_confidence_x'],inplace=True)
    df_process = df_process.groupby(by='name_x').sum().reset_index()
    df_process.rename(columns={'name_x':'name','Predicted_confidence_y':'community_score'},inplace=True)

    df_process.sort_values(by='community_score')
    print(df_process)

    df_process.to_csv("df_process.csv")

    df_graph = df_clean.groupby(by=['name_x','name_y']).count().reset_index()
    df_graph.drop(columns='Predicted_confidence_x',inplace=True)
    df_graph = df_graph.rename(columns={'Predicted_confidence_y':'size'})

    cols = ['name_x','name_y']

    df_graph[cols] = np.sort(df_graph[cols].values,axis=1)
    df_graph = df_graph.drop_duplicates()
    df_graph.sort_values(by='size',ascending=False)
    print(df_graph)

    df_graph.to_csv("df_graph.csv")

    K = nx.from_pandas_edgelist(df_graph, 'name_x', 'name_y')

    print("Finished")

    groups = list(K.subgraph(c) for c in nx.connected_components(K))

    nameList = []

    for index, group in enumerate(groups):
        
        if not (len(group) > 1):
            continue
            
        for name in group:
            nameList.append({'name':name, 'group_number':index})
            
    df_groups = pd.DataFrame(nameList)
    print(df_groups)

    df_groups.to_csv("df_groups.csv")


    """
        ncv = network community value
    """

    df_ncv = pd.merge(df_groups, df_process, how='inner',left_on='name',right_on='name')
    del df_process
    df_ncv.sort_values(by='community_score')

    df_ncv_count = df_ncv.groupby('group_number').count().drop('community_score',axis=1)
    df_ncv_max = df_ncv.groupby('group_number').max().drop('name',axis=1)
    df_ncv_min = df_ncv.groupby('group_number').min().drop('name',axis=1)

    df_ncv_full = pd.DataFrame()

    df_ncv_full['name_count'] = df_ncv_count['name']
    df_ncv_full['max'] = df_ncv_max['community_score']
    df_ncv_full['min'] = df_ncv_min['community_score']
    print(df_ncv_full)

    m1 = df_ncv_full['max'] < 0
    m2 = df_ncv_full['min'] < 0
    mask = m1&m2
    df_ncv_masked = df_ncv_full[mask].sort_values(by='name_count',ascending=False)
    del df_ncv_full

    for group_index in df_ncv_masked.index.values:
        mask = df_ncv['group_number'] == group_index
        print(df_ncv[mask])

    ## for taking apart the gordian knot - nonsense for now please ignore ;d

    #m1 = df_ncv['group_number'] == 0
    #m2 = df_ncv[mask]['community_score'] < 0
    #mask = m1&m2
    #df_ncv[mask].sort_values(by='community_score')


    mask = df_graph['size'] > 50
    df_minor_graph = df_graph[mask].sort_values(by='size',ascending=0)

    df_name_conf = df_transformed[['name','Predicted_confidence']].copy() # produces singles name list with conf
    df_name_conf.drop_duplicates(inplace=True)

if __name__ == "__main__":
    client = Client()

    process_names()
    main()