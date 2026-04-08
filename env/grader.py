def compute_score(task, action):

    # handle if action is int
    if isinstance(action, int):
        categories = ["spam", "work", "personal"]
        predicted = categories[action]
    else:
        predicted = action.predicted_category

    true_label = task[1]

    if predicted == true_label:
        return 0.9
    elif predicted in ["spam", "work", "personal"]:
        return 0.5
    else:
        return 0.1