import random

def create_train_test_dict(cluster_data):
    training_dict = {}
    testing_dict = {}

    #group all tracks by user
    grouped_users = cluster_data.groupby('user_id')['trackname'].unique()

    for user, tracks in grouped_users.items():
        tracks_list = list(tracks)

        # Skip users with less than 3 tracks
        if len(tracks_list) < 3:
            continue  

        # shuffle and find split index
        random.shuffle(tracks_list)
        split_idx = int(len(tracks_list) * (2/3))

        #slice and make sets for train and test
        training_dict[user] = set(tracks_list[:split_idx])
        testing_dict[user] = set(tracks_list[split_idx:])   

    
    print(f"  -> Successfully split {len(training_dict)} users into train/test sets.")
    return training_dict, testing_dict