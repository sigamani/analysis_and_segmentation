import pandas as pd
import numpy as np


def get_averages(segment_df, total_avgs):

    binary_df = pd.get_dummies(segment_df)
    segment_avgs = np.mean(binary_df, axis=0)
    top_features_indices = np.argsort(segment_avgs)[::-1]

    cluster_key_features = [segment_df.columns[tf]
                            for tf in top_features_indices]
    cluster_avgs = [segment_avgs[tf] for tf in top_features_indices]
    total_avgs = [total_avgs[tf] for tf in top_features_indices]

    final_avgs = []
    for i in range(len(total_avgs)):
        a = abs((cluster_avgs[i] - total_avgs[i]))
        final_avgs.append(a)

    d = {"Segment": cluster_key_features,
         'SegmentAvg': cluster_avgs,
         'TotalAvg': total_avgs}

    d = pd.DataFrame(data=d)
    df = d.sort_values(by=['SegmentAvg'], ascending=False)
   # d = df[df.Segment.str.contains('Checked', case=True)]
    return d


def get_top_features(data, segment):
    segments_labels = data.clusters
    segments_fts = pd.get_dummies(data.drop('clusters', axis=1))

    total_avgs = segments_fts.mean(axis=0)

    clusters = segments_labels.unique()
    segments_dict = {cluster: segments_fts[segments_labels == cluster] for cluster in clusters}

    segment_ = segments_dict.get(segment)
    segment_fts = get_averages(segment_, total_avgs)
    return segment_fts 
