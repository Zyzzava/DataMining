

def get_recommendations(target_user, training_dict):
    #grab songs of target user in training set
    target_tracks = training_dict[target_user]

    track_scores = {}

    #calculate jaccard similarity between target user and all other users
    for other_user, other_tracks in training_dict.items():
        #dont compare user to themselves
        if other_user == target_user:
            continue

        #calculate jaccard similarity
        intersection = len(target_tracks & other_tracks)
        if intersection == 0:
            continue
        union = len(target_tracks | other_tracks)
        jaccard_sim = intersection / union

        #score cadidate tracks based on jaccard similarity
        for track in other_tracks:
            if track not in target_tracks:
                if track not in track_scores:
                    track_scores[track] = 0
                track_scores[track] += jaccard_sim

    # sort tracks by score
    ranked_tracks = sorted(track_scores.items(), key=lambda x: x[1], reverse=True)

    # return track names
    final_predictions = [track_tuple[0] for track_tuple in ranked_tracks]
    return final_predictions


