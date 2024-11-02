import pandas as pd # type: ignore
import helpers 


config = helpers.load_config("../../configs/candidate_config.yaml")


# User data
user_data = pd.DataFrame({
    'user_id': [1, 2, 3, 4, 5],
    'age': [25, 34, 45, 23, 30],
    'location': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix']
})

# Item data
item_data = pd.DataFrame({
    'item_id': [101, 102, 103, 104, 105],
    'category': ['Electronics', 'Books', 'Fashion', 'Sports', 'Books'],
    'brand': ['BrandA', 'BrandB', 'BrandC', 'BrandD', 'BrandB'],
    'price': [199.99, 15.99, 49.99, 29.99, 12.99]
})

# User-Item Interaction data (ratings or clicks)
interaction_data = pd.DataFrame({
    'user_id': [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
    'item_id': [101, 102, 102, 103, 101, 104, 105, 101, 104, 105],
    'rating': [5, 4, 3, 4, 5, 2, 3, 4, 5, 1]
})

# Access paths from the YAML configuration
user_data_path = config['data']['user_data_path']
item_data_path = config['data']['item_data_path']
interaction_data_path = config['data']['interaction_data_path']

print(user_data_path)
print(item_data_path)
print(interaction_data_path)


# user_data.to_csv('users.csv', index=False)
# item_data.to_csv('items.csv', index=False)
# interaction_data.to_csv('interactions.csv', index=False)


# # Load the data (assuming it's saved as CSV files)
# user_data = pd.read_csv('users.csv')
# item_data = pd.read_csv('items.csv')
# interaction_data = pd.read_csv('interactions.csv')

# # Preview the data
# print("Users Data:\n", user_data)
# print("\nItems Data:\n", item_data)
# print("\nInteraction Data:\n", interaction_data)
