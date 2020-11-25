def get_user_index(users,user_id,user_id_col='user_id'):
    res=users[users[user_id_col] == user_id]
    if len(res) > 1:
        print("Ambiguous: found")
    elif len(res) == 0:
        print('not found')
    else:
        return res.index[0]
    
def get_user_ID(users,index,user_id_col='user_id'):
    return (users.iloc[index])[user_id_col]
