def find_result(customer_id) :
    import pandas as pd
    data = pd.read_csv("test.csv")
    result = data[data['id'] == customer_id]['target'].values[0]

    return result

# customer_id = 2
# data[data['id'] == customer_id]['target'].values[0]
