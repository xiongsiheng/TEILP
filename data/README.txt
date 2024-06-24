# For wiki_time_pred_eval_rm_idx.json and YAGO_time_pred_eval_rm_idx.json:
# Some events in wiki and YAGO have missing time (either start time or end time or both). We only consider known-time events for evaluation.

# For gdelt100:
# It is a subset of the original gdelt dataset. We choose the top 100 entities that have the most events. We also delete repeated events (all the elements, i.e., subject, relation, object, time, are the same).