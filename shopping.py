import csv
import sys
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):




    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """

    months = {'Jan': 0,
              'Feb': 1,
              'Mar': 2,
              'Apr': 3,
              'May': 4,
              'June': 5,
              'Jul': 6,
              'Aug': 7,
              'Sep': 8,
              'Oct': 9,
              'Nov': 10,
              'Dec': 11}

    def convert_to_binary(x):
        if x == "Returning_visitor" or x == "True" or x is True:
            return 1
        else:

            return 0

    file = pd.read_csv(filename)
    dataframe_l = file['Revenue'] #gets only the revenue columns
    dataframe_e = file.drop(columns = "Revenue") #gets all the columns besides revenue

    dataframe_e = dataframe_e.replace(months)

    #converts values to either 0 or 1
    dataframe_e["Weekend"] = dataframe_e["Weekend"].apply(convert_to_binary)
    dataframe_e["VisitorType"] = dataframe_e["VisitorType"].apply(convert_to_binary)
    dataframe_l = dataframe_l.apply(convert_to_binary)


    evidence = dataframe_e.values.tolist()
    labels = dataframe_l.values.tolist()

    #Month = 10, Visitor_Type = 15, Weekend = 16
    #Month = {Jan: 1, Feb : 2...Dec:12}
    #Visitor_Type = {Returning: 1, Non_returning: 0}
    #Weekend = {Yes: 1, No: 0}
    #Other indexed are floats


    return (evidence,labels)


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """

    divide = int(TEST_SIZE * len(evidence)) #can also use(len(label)) as they are the same length

    training_e = evidence[divide:]
    training_l = labels[divide:]

    model = KNeighborsClassifier(n_neighbors = 1)


    return model.fit(training_e,training_l)


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificity).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """

    #labels = labels in the testing set which are labels[:divide]
    #predictions = predictions predicted by model

    sens = []
    spec = []

    for (label, pred) in list(zip(labels,predictions)):
        if label == 1 and pred == 1:
            sens.append(1)
        elif label == 1 and pred == 0:
            sens.append(0)
        elif label == 0 and pred == 0:
            spec.append(1)
        elif label == 0 and pred == 1:
            spec.append(0)

    return (sum(sens)/len(sens), sum(spec)/len(spec))

if __name__ == "__main__":
    main()
