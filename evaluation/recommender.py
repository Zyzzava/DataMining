from collections import defaultdict

def get_recommendations(target_user, training_dict, track_to_users_index):
    #grab songs of target user in training set
    target_tracks = training_dict[target_user]

    #use track_to_users_index to find other users who listened to the same tracks as target user
    potential_neighbors = set()
    for track in target_tracks:
        potential_neighbors.update(track_to_users_index.get(track, set()))

    #remove target user from potential neighbors
    potential_neighbors.discard(target_user)

    track_scores = defaultdict(float)
    #calculate jaccard similarity between uses who share tracks
    for other_user in potential_neighbors:      
        other_tracks = training_dict[other_user]

        #calculate jaccard similarity
        intersection = len(target_tracks & other_tracks)
        union = len(target_tracks | other_tracks)
        jaccard_sim = intersection / union

        #score cadidate tracks based on jaccard similarity
        new_tracks = other_tracks - target_tracks
        for track in new_tracks:
            track_scores[track] += jaccard_sim

    # sort tracks by score and return top recommendations
    ranked_tracks = sorted(track_scores.items(), key=lambda x: x[1], reverse=True)
    return [track_tuple[0] for track_tuple in ranked_tracks]


