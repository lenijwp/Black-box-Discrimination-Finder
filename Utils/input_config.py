
class Dat:
    """
    Configuration of dataset Census Income
    """

    # the size of total features
    params = 13

    # the valid religion of each feature
    input_bounds = []
    input_bounds.append([1, 9])
    input_bounds.append([0, 7])
    input_bounds.append([0, 39]) #69 for THEMIS
    input_bounds.append([0, 15])
    input_bounds.append([0, 6])
    input_bounds.append([0, 13])
    input_bounds.append([0, 5])
    input_bounds.append([0, 4])
    input_bounds.append([0, 1])
    input_bounds.append([0, 99])
    input_bounds.append([0, 39])
    input_bounds.append([0, 99])
    input_bounds.append([0, 39])

    # the name of each feature
    feature_name = ["age", "workclass", "fnlwgt", "education", 
    "marital_status", "occupation", "relationship", "race", "gender", "capital_gain",
    "capital_loss", "hours_per_week", "native_country"]

    # the name of each class
    class_name = ["low", "high"]
    data_path = 'Demo/Datasets/census'
    model_path = 'Demo/Targetmodels/census_model1.h5'
    # specify the categorical features with their indices
    categorical_features = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    protected_attri={8:'gender'}  ##protected_attribute information
    pro_at=[8]



class census:
    """
    Configuration of dataset Census Income
    """

    # the size of total features
    params = 13

    # the valid religion of each feature
    input_bounds = []
    input_bounds.append([1, 9])
    input_bounds.append([0, 7])
    input_bounds.append([0, 39]) #69 for THEMIS
    input_bounds.append([0, 15])
    input_bounds.append([0, 6])
    input_bounds.append([0, 13])
    input_bounds.append([0, 5])
    input_bounds.append([0, 4])
    input_bounds.append([0, 1])
    input_bounds.append([0, 99])
    input_bounds.append([0, 39])
    input_bounds.append([0, 99])
    input_bounds.append([0, 39])

    # the name of each feature
    feature_name = ["age", "workclass", "fnlwgt", "education", 
    "marital_status", "occupation", "relationship", "race", "gender", "capital_gain",
    "capital_loss", "hours_per_week", "native_country"]

    # the name of each class
    class_name = ["low", "high"]
    data_path = 'Demo/Datasets/census'
    model_path = 'Demo/Targetmodels/census_model1.h5'
    # specify the categorical features with their indices
    categorical_features = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    protected_attri={8:'gender'}  ##protected_attribute information
    pro_at=[8]

class credit:
    """
    Configuration of dataset German Credit
    """

    # the size of total features
    params = 20

    # the valid religion of each feature
    input_bounds = []
    input_bounds.append([0, 3])
    input_bounds.append([1, 80])
    input_bounds.append([0, 4])
    input_bounds.append([0, 10])
    input_bounds.append([1, 200])
    input_bounds.append([0, 4])
    input_bounds.append([0, 4])
    input_bounds.append([1, 4])
    input_bounds.append([0, 1])
    input_bounds.append([0, 2])
    input_bounds.append([1, 4])
    input_bounds.append([0, 3])
    input_bounds.append([1, 8])
    input_bounds.append([0, 2])
    input_bounds.append([0, 2])
    input_bounds.append([1, 4])
    input_bounds.append([0, 3])
    input_bounds.append([1, 2])
    input_bounds.append([0, 1])
    input_bounds.append([0, 1])

    # the name of each feature
    feature_name = ["checking_status", "duration", "credit_history", "purpose", "credit_amount", "savings_status", "employment", "installment_commitment", "sex", "other_parties",
                                                                      "residence", "property_magnitude", "age", "other_payment_plans", "housing", "existing_credits", "job", "num_dependents", "own_telephone", "foreign_worker"]
 
    # the name of each class
    class_name = ["bad", "good"]

    data_path = 'Demo/Datasets/credit'
    model_path = 'Demo/Targetmodels/credit_model1.h5'
    # specify the categorical features with their indices
    categorical_features = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,17, 18, 19]
    protected_attri={12:'age'}  ##protected_attribute information
    pro_at=[12]


class bank:
    """
    Configuration of dataset Bank Marketing
    """

    # the size of total features
    params = 16

    # the valid religion of each feature
    input_bounds = []
    input_bounds.append([1, 9])
    input_bounds.append([0, 11])
    input_bounds.append([0, 2])
    input_bounds.append([0, 3])
    input_bounds.append([0, 1])
    input_bounds.append([-20, 179])
    input_bounds.append([0, 1])
    input_bounds.append([0, 1])
    input_bounds.append([0, 2])
    input_bounds.append([1, 31])
    input_bounds.append([0, 11])
    input_bounds.append([0, 99])
    input_bounds.append([1, 63])
    input_bounds.append([-1, 39])
    input_bounds.append([0, 1])
    input_bounds.append([0, 3])

    # the name of each feature
    feature_name = ["age", "job", "marital", "education", "default", "balance", "housing", "loan", "contact", "day",
                                                                      "month", "duration", "campaign", "pdays", "previous", "poutcome"]

    # the name of each class



    class_name = ["no", "yes"]
    data_path = 'Demo/Datasets/bank'
    model_path = 'Demo/Targetmodels/bank_model1.h5'
    # specify the categorical features with their indices
    categorical_features = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    protected_attri={0:'age'}  ##protected_attribute information
    pro_at=[0]